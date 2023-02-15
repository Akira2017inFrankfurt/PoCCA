import math
import torch.nn as nn
from utils_data import *
from utils_model import *
from attention import CrossAttnBlock


class SimAttention_Multi_X_Attn(nn.Module):
    def __init__(self, online_encoder, patch_num_list):
        super(SimAttention_Multi_X_Attn, self).__init__()
        self.online_encoder = online_encoder
        self.patch_num_list = patch_num_list

        self.online_projector = ProjectMLP().cuda()
        self.online_attn = CrossAttnBlock().cuda()
        self.predictor = ProjectMLP().cuda()
        
        self.target_encoder = None
        self.target_projector = None
        self.target_attn = None

    def forward(self, aug1, aug2):
        self.target_encoder = momentum_update(self.online_encoder, self.target_encoder)
        self.target_projector = momentum_update(self.online_projector, self.target_projector)
        self.target_attn = momentum_update(self.online_attn, self.target_attn)

        # global downsample
        _, global_sample1 = b_FPS(aug1, 1024)
        _, global_sample2 = b_FPS(aug2, 1024)

        # global features
        global_feat1 = self.online_encoder(global_sample1)
        global_feat2 = self.target_encoder(global_sample2)
        global_feat3 = self.online_encoder(global_sample2)
        global_feat4 = self.target_encoder(global_sample1)
        
        # concat patch features
        patch_features = get_2_branch_patch_features(aug1, aug2, 
            self.patch_num_list, self.online_encoder)

        # cross attn and projector, online with predictor
        x_f_1 = self.predictor(self.online_projector(self.online_attn(global_feat1, patch_features)))
        x_f_2 = self.target_projector(self.target_attn(global_feat2, patch_features))
        x_f_3 = self.predictor(self.online_projector(self.online_attn(global_feat2, patch_features)))
        x_f_4 = self.target_projector(self.target_attn(global_feat1, patch_features))

        # get loss
        loss_1 = loss_fn(x_f_1, x_f_2)
        loss_2 = loss_fn(x_f_3, x_f_4)

        loss = loss_1 + loss_2
        return loss.mean()


if __name__ == "__main__":
    import torch
    from network.basic_encoders import PointNet_CLS_Encoder

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rand_x_0 = torch.rand([4, 2048, 3]).to(device)
    rand_x_1 = torch.rand([4, 2048, 3]).to(device)

    test_model = SimAttention_Multi_X_Attn(
        online_encoder=PointNet_CLS_Encoder().to(device),
        online_projector=ProjectMLP().to(device),
        predictor=ProjectMLP().to(device),
        x_attn=CrossedAttention().to(device),
        patch_num_list=[16, 4, 1]
    )

    rand_out = test_model(rand_x_0, rand_x_1)
    print('Done! output is: ', rand_out)