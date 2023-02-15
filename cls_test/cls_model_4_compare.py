import os.path

from network.basic_encoders import PointNet_CLS_Encoder, DGCNN_CLS_Encoder
from network.basic_encoders import DGCNN_CLS_Encoder_1, DG_Tail
from torch.cuda.amp import autocast
from utils.crops import b_FPS
import torch.nn as nn
import torch


def vis_local_net_para(net):
    net_dict = net.state_dict()
    count = 0
    for k in net_dict.keys():
        print(k)
        count += 1
    print('Total Parameters: ', count)


def vis_trained_net_para(path):
    loaded_paras = torch.load(path)
    for k in loaded_paras.keys():
        print(k)


def get_encoder(encoder, path):
    loaded_paras = torch.load(path)
    encoder = encoder.cuda()
    encoder_dict = encoder.state_dict()
    new_state_dict = {}

    for k in loaded_paras.keys():
        if k.startswith('online_encoder'):
            new_k = k[15:]
            new_state_dict[new_k] = loaded_paras[k]

    encoder_dict.update(new_state_dict)
    encoder.load_state_dict(encoder_dict)
    return encoder


class CLS_Model(nn.Module):
    def __init__(self, model_choice=0, use_pretrain=True, weight_path=None, freeze_encoder=False):
        super().__init__()
        # get model structure, 0 for PointNet, 1 for DGCNN
        if model_choice == 0:
            self.encoder_head = PointNet_CLS_Encoder().cuda()
            print("Encoder from PointNet.")
        else:
            self.encoder_head = DGCNN_CLS_Encoder().cuda()
            print("Encoder from DGCNN.")

        # get model weights
        if use_pretrain:
            print("Use the pretrained model weights!")
            self.encoder = get_encoder(self.encoder_head, weight_path)
        else:
            self.encoder = self.encoder_head

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            print('Freeze the pretrained encoder while training')
        else:
            print('Not freeze the pretrained encoder')

        self.relu = nn.ReLU()
        self.bn0 = nn.BatchNorm1d(1024)
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 40)

    def forward(self, x):
        # print('x shape before fps: ', x.shape)  # B, 10000, 3
        # downsample data from 10000 points to 1024 points
        _, x = b_FPS(x, 1024)
        # print('x shape after fps: ', x.shape)  # B, 1024, 3
        with autocast():
            if self.freeze_encoder:
                with torch.no_grad():
                    x = self.encoder(x)  # B, 1, 1024
            else:
                x = self.encoder(x)

            x = x.reshape(x.shape[0], -1)  # B, 1024
            x = self.bn0(x)
            x = self.relu(self.bn1(self.linear1(x)))
            x = self.dp1(x)
            x = self.relu(self.bn2(self.linear2(x)))
            x = self.dp2(x)
            x = self.linear3(x)
            return x


if __name__ == "__main__":
    def test_0():
        test_net_1 = PointNet_CLS_Encoder().cuda()
        test_net_2 = DGCNN_CLS_Encoder().cuda()
        # vis_local_net_para(test_net_1)  # 58
        # vis_local_net_para(test_net_2)  # 60
        path_pn = r'/home/haruki/下载/share/model_knn_1024_fps-100-0-v2.pth'
        # vis_trained_net_para(path_pn)
        # vis_trained_net_para(path_dg)
        # encoder = get_encoder(test_net_1, path_pn)
        test_model = CLS_Model(use_pretrain=False).cuda()
        rand_input = torch.rand([2, 10000, 3]).cuda()
        output = test_model(rand_input)
        print('output shape: ', output.shape)
        print('Done!')

    def pretrained_weight_test():
        path = r'/home/haruki/下载/SimAttention/scripts/weights/'
        name = 'model_knn_2048_proj_1_4_1-v2-1009-100-0.pth'
        print('model: ', name)
        model_path = os.path.join(path, name)
        vis_trained_net_para(model_path)

    # pretrained_weight_test()

    def test_3_full_dgcnn():
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_path = r'/home/haruki/下载/SimAttention/scripts/weights/'
        model_name = 'cls_dg_1205_cross_99.pth'
        model = DGCNN_CLS_Encoder_1().to(device)
        encoder = get_encoder(model, os.path.join(model_path, model_name))
        rand_input = torch.rand([2, 10000, 3]).to(device)
        # output = encoder(rand_input)
        # print('output shape: ', output.shape) # torch.Size([2, 1024, 10000])
        new_model = DG_Tail(encoder=encoder).to(device)
        output = new_model(rand_input)
        print('output shape: ', output.device)  # [2, 40]

    test_3_full_dgcnn()