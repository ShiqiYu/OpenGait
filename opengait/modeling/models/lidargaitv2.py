import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .lidargaitv2_utils import PointNetSetAbstraction, PPPooling, PPPooling_UDP,NetVLAD

from ..base_model import BaseModel
from ..modules import SeparateFCs, SeparateBNNecks


class LidarGaitPlusPlus(BaseModel):
    def build_network(self, model_cfg):
        C = model_cfg['channel']
        C_out = model_cfg['SeparateFCs']['in_channels']
        scale_aware = model_cfg['scale_aware']
        normalize_dp = model_cfg['normalize_dp']
        sampling = model_cfg['sampling']

        npoints = model_cfg.get('npoints', [512, 256, 128])
        nsample = model_cfg.get('nsample', 32)
        in_channel = 4 if scale_aware else 3

        self.sa1 = PointNetSetAbstraction(npoint=npoints[0], radius=0.1, nsample=nsample, in_channel=in_channel, mlp=[2*C, 2*C, 4*C], group_all=False, sampling=sampling, scale_aware=scale_aware, normalize_dp=normalize_dp)
        self.sa2 = PointNetSetAbstraction(npoint=npoints[1], radius=0.2, nsample=nsample, in_channel=4*C + in_channel, mlp=[4*C, 4*C, 8*C], group_all=False, sampling=sampling, scale_aware=scale_aware, normalize_dp=normalize_dp)
        self.sa3 = PointNetSetAbstraction(npoint=npoints[2], radius=0.4, nsample=nsample, in_channel=8*C + in_channel, mlp=[8*C, 8*C, 16*C], group_all=False, sampling=sampling, scale_aware=scale_aware, normalize_dp=normalize_dp)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=16*C + in_channel, mlp=[16*C, 16*C, C_out], group_all=True, sampling=sampling, scale_aware=scale_aware, normalize_dp=normalize_dp)

        if model_cfg['pool'] == 'VLAD':
            self.pool = NetVLAD(num_clusters=16, dim=C_out, alpha=1.0)
        elif model_cfg['pool'] == 'GMaxP':
            self.pool = PPPooling_UDP([1])
        elif model_cfg['pool'] == 'PPP_UDP':
            self.pool = PPPooling_UDP(model_cfg['scale'])
        elif model_cfg['pool'] == 'PPP_UAP':
            self.pool = PPPooling(scale_aware=False, bin_num=model_cfg['scale'])
        elif model_cfg['pool'] == 'PPP_HAP':
            self.pool = PPPooling(scale_aware=True, bin_num=model_cfg['scale'])
        


        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs']) 


    def forward(self, inputs):
        ipts, labs, _, views, seqL = inputs

        xyz = ipts[0]
        B, T, N, C = xyz.shape
        xyz = rearrange(xyz, 'B T N C -> (B T) C N')


        l1_xyz, l1_points = self.sa1(xyz, None)
        l1_points = torch.max(l1_points, dim=-2)[0]

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = torch.max(l2_points, dim=-2)[0]

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = torch.max(l3_points, dim=-2)[0]

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        x = self.pool(l4_points, l3_xyz)


        x = rearrange(x, '(B T) feat p -> B T feat p', B=B)
        feat = x.max(1)[0]# x.mean(1) # x.max(1)[0]
        embed = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed)  # [n, c, p]
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
            },
            'inference_feat': {
                'embeddings': embed,
            }
        }
        return retval