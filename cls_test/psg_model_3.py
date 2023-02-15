import copy
import math
import torch
import torch.nn as nn
from model import loss_fn
from partseg_hf5 import ShapeNetPart


# 2 functions from dgcnn
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if not dim9:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    local_device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=local_device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # (batch_size, 2*num_dims, num_points, k)


def get_patch_feature(patch, patch_encoder):
    return patch_encoder(patch)


def get_patches_feature(patches, patch_encoder):
    num_patches = patches.shape[1]
    patch_features = get_patch_feature(torch.squeeze(patches[:, 0, :, :]), patch_encoder)
    for i in range(1, num_patches):
        current_patch_feature = get_patch_feature(torch.squeeze(patches[:, i, :, :]), patch_encoder)
        patch_features = torch.cat((patch_features, current_patch_feature), dim=1)
    return patch_features


def get_points_feature(data, point_encoder):
    return point_encoder(data)


def new_loss_fn(x, y, proj):
    l_loss = 0.0
    for i in range(x.shape[2]):
        l_loss += loss_fn(proj(x[:, :, i]), y[:, :, i])
    return l_loss / x.shape[2]


def get_loss(
        sample_1,
        sample_2,
        concat_patch_feature,
        projector,
        o_en,
        t_en,
        o_x_attn,
        t_x_attn):
    # [B, N, 256], N = 2048
    po_f1 = torch.squeeze(get_points_feature(sample_1, o_en))
    po_f2 = torch.squeeze(get_points_feature(sample_2, t_en))
    po_f3 = torch.squeeze(get_points_feature(sample_2, o_en))
    po_f4 = torch.squeeze(get_points_feature(sample_1, t_en))

    # B, 256, N
    x_attn_1 = o_x_attn(po_f1, concat_patch_feature)
    x_attn_2 = t_x_attn(po_f2, concat_patch_feature)
    x_attn_3 = o_x_attn(po_f3, concat_patch_feature)
    x_attn_4 = t_x_attn(po_f4, concat_patch_feature)

    l1 = new_loss_fn(x_attn_1, x_attn_2, projector)
    l2 = new_loss_fn(x_attn_3, x_attn_4, projector)

    return l1 + l2


@torch.no_grad()
def momentum_update(online, target, tao=0.99):
    if target is None:
        target = copy.deepcopy(online)
    else:
        for online_params, target_params in zip(online.parameters(), target.parameters()):
            target_weight, online_weight = target_params.data, online_params.data
            target_params.data = target_weight * tao + (1 - tao) * online_weight
    for parameter in target.parameters():
        parameter.requires_grad = False
    return target


class SA_MH_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, f, p = x.shape
        x_q = self.q_conv(x)
        x_q = x_q.reshape(bs, 4, -1, p).permute(0, 1, 3, 2)
        x_k = self.k_conv(x)
        x_k = x_k.reshape(bs, 4, -1, p)
        xy = torch.matmul(x_q, x_k)
        xy = self.softmax(xy / math.sqrt(x_k.shape[-2]))

        x_v = self.v_conv(x)
        x_v = x_v.reshape(bs, 4, -1, p).permute(0, 1, 3, 2)
        xyz = torch.matmul(xy, x_v)
        xyz = xyz.permute(0, 1, 3, 2).reshape(bs, -1, p)
        xyz = self.act(self.after_norm(self.trans_conv(xyz - x)))
        xyz = x + xyz
        return xyz


class SegXAttn(nn.Module):
    def __init__(self, q_in=256, q_out=256, k_in=1024):
        super().__init__()
        self.q_in = q_in
        self.q_out = q_out
        self.k_in = k_in

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, q_out, 1, bias=False)
        self.v_conv = nn.Conv1d(k_in, q_in, 1, bias=False)

        self.trans_conv = nn.Conv1d(q_in, q_in, 1)
        self.after_norm = nn.BatchNorm1d(q_in)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_tensor, kv_tensor):
        # print('q_tensor: ', q_tensor.shape)
        # N, 256 ---> N, 256
        x_q = self.q_conv(q_tensor)
        # print('x_q: ', x_q.shape)
        # 16, 1024 ---> 16, 256
        x_k = self.k_conv(kv_tensor.permute(0, 2, 1))
        # print('x_k: ', x_k.shape)
        # 16, 1024 ---> 16, 256
        x_v = self.v_conv(kv_tensor.permute(0, 2, 1))
        # print('x_v: ', x_v.shape)
        # N, 16
        energy = torch.matmul(x_q.permute(0, 2, 1), x_k)
        # print('energy: ', energy.shape)
        attention = self.softmax(energy / math.sqrt(x_k.shape[-2]))
        # attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.matmul(attention, x_v.permute(0, 2, 1))
        # print('x_r shape: ', x_r.shape)
        res = (q_tensor - x_r.permute(0, 2, 1))
        # print('res: ', res.shape)
        x_r = self.act(self.after_norm(self.trans_conv(res)))
        # print('last x_r:', x_r.shape)
        x_r = x_r + q_tensor

        return x_r


class Encoder_Head(nn.Module):
    def __init__(self, k=40, part_num=50):
        super(Encoder_Head, self).__init__()
        self.k = k
        self.part_num = part_num
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.sa1 = SA_MH_Layer(128)
        self.sa2 = SA_MH_Layer(128)
        self.sa3 = SA_MH_Layer(128)
        self.sa4 = SA_MH_Layer(128)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B,N,3] -> [B,3,N]
        # from DGCNN EdgeConv
        # 1L
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # 2L
        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # # 3L
        # x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        # x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)
        x = torch.cat((x1, x2), dim=1)  # (batch_size, 64*2, num_points)
        # print('--test in encoder, before sa layer, x: ', x.shape)
        # b, 128, 2048
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        # 128 * 4 = 512, [B, 512, N]
        x = torch.concat((x1, x2, x3, x4), dim=1)
        # 512 --> 256, [B, 256, N]
        x = self.conv_fuse(x)
        # x_max = torch.max(x, 2)[0]  # [B, 1024]
        # x_max = torch.max(x, 2)[0]  # [B, 1024]
        return x


class Encoder_Patch(nn.Module):
    def __init__(self, encoder_head):
        super(Encoder_Patch, self).__init__()
        self.encoder_head = encoder_head
        self.conv_fuse = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.shape[0]
        pointwise_feature = self.encoder_head(x)
        pa = self.conv_fuse(pointwise_feature)
        x = torch.max(pa, 2)[0]
        patch_feature = x.view(batch_size, -1)
        return patch_feature.reshape(batch_size, 1, -1)


class Projector(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, hidden_size=1024):
        super(Projector, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(input_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.bn(self.l1(x.reshape(x.shape[0], -1)))
        x = self.l2(self.relu(x))
        return x.reshape(x.shape[0], -1)


class SimAttn_Seg(nn.Module):
    def __init__(self, knn_function, sub_function):
        super(SimAttn_Seg, self).__init__()

        self.knn_function = knn_function
        self.sub_function = sub_function
        self.project_function = Projector().to(device)
        self.online_encoder = Encoder_Head().to(device)
        self.patch_encoder = Encoder_Patch(self.online_encoder).to(device)
        self.online_x_attn = SegXAttn().to(device)

        self.target_encoder = None
        self.target_x_attn = None

    def forward(self, aug1, aug2):
        self.target_encoder = momentum_update(self.online_encoder, self.target_encoder)
        self.target_x_attn = momentum_update(self.online_x_attn, self.target_x_attn)

        # get subsample [B, 2048, 3]
        _, sub1 = self.sub_function(aug1, 2048)
        _, sub2 = self.sub_function(aug2, 2048)
        # print('sub1 shape: ', sub1.shape)
        # get patches [B, 8, n_f, 3], n_f = 256, 128, 512
        patches_1 = self.knn_function(sub1)
        patches_2 = self.knn_function(sub2)
        # print('patches_1 shape: ', patches_1.shape)
        # patch features
        patch_feature_1 = get_patches_feature(patches_1, self.patch_encoder)
        patch_feature_2 = get_patches_feature(patches_2, self.patch_encoder)
        # print("patch_feature_1 shape: ", patch_feature_1.shape)  # [4, 8, 1024]
        patch_features = torch.cat((patch_feature_1, patch_feature_2), dim=1)  # [4, 16, 1024]
        # print('concat patch feature shape: ', patch_features.shape)
        contrastive_loss = get_loss(
            sub1, sub2,
            patch_features,
            self.project_function,
            self.online_encoder,
            self.target_encoder,
            self.online_x_attn,
            self.target_x_attn
        )

        return contrastive_loss


if __name__ == "__main__":
    # basic test
    # rand_x_1 = torch.rand([2, 2679, 3])
    # rand_x_2 = torch.rand([2, 2867, 3])
    from utils.crops import b_FPS, new_k_patch_256

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    partseg = SimAttn_Seg(new_k_patch_256, b_FPS).to(device)
    # seg_loss = partseg(rand_x_1.cuda(), rand_x_2.cuda())
    # load data in
    class_choice = 'bag'
    train_dataset = ShapeNetPart(partition='trainval',
                                 num_points=2048,
                                 class_choice=class_choice)
    from torch.utils.data import DataLoader

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               num_workers=8,
                                               batch_size=2,
                                               shuffle=True,
                                               drop_last=True)

    for morph1, morph2, _, _ in train_loader:
        loss = partseg(morph1.to(device), morph2.to(device))
        print('loss is: ', loss)
