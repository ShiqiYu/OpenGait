import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, SetBlockWrapper, conv3x3, conv1x1, BasicBlock2D, BasicBlockP3D

from einops import rearrange

import copy

class SkeletonGaitPP(BaseModel):

   def build_network(self, model_cfg):
       #B, C = [1, 4, 4, 1], 2
       in_C, B, C = model_cfg['Backbone']['in_channels'], model_cfg['Backbone']['blocks'], model_cfg['Backbone']['C']
       self.inference_use_emb = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

       self.inplanes = 32 * C
       self.sil_layer0 = SetBlockWrapper(nn.Sequential(
           conv3x3(1, self.inplanes, 1),
           nn.BatchNorm2d(self.inplanes),
           nn.ReLU(inplace=True)
       ))

       self.map_layer0 = SetBlockWrapper(nn.Sequential(
           conv3x3(2, self.inplanes, 1),
           nn.BatchNorm2d(self.inplanes),
           nn.ReLU(inplace=True)
       ))

       self.sil_layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, 32 * C, stride=[1, 1], blocks_num=B[0], mode='2d'))
       self.map_layer1 = copy.deepcopy(self.sil_layer1)
       self.fusion = AttentionFusion(32 * C)

       self.layer2 = self.make_layer(BasicBlockP3D, 64 * C, stride=[2, 2], blocks_num=B[1], mode='p3d')
       self.layer3 = self.make_layer(BasicBlockP3D, 128 * C, stride=[2, 2], blocks_num=B[2], mode='p3d')
       self.layer4 = self.make_layer(BasicBlockP3D, 256 * C, stride=[1, 1], blocks_num=B[3], mode='p3d')

       self.FCs = SeparateFCs(16, 256*C, 128*C)
       self.BNNecks = SeparateBNNecks(16, 128*C, class_num=model_cfg['SeparateBNNecks']['class_num'])

       self.TP = PackSequenceWrapper(torch.max)
       self.HPP = HorizontalPoolingPyramid(bin_num=[16])

   def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

       if max(stride) > 1 or self.inplanes != planes * block.expansion:
           if mode == '3d':
               downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
           elif mode == '2d':
               downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
           elif mode == 'p3d':
               downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
           else:
               raise TypeError('xxx')
       else:
           downsample = lambda x: x

       layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
       self.inplanes = planes * block.expansion
       s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
       for i in range(1, blocks_num):
           layers.append(
                   block(self.inplanes, planes, stride=s)
           )
       return nn.Sequential(*layers)

   def inputs_pretreament(self, inputs):
       ### Ensure the same data augmentation for heatmap and silhouette
       pose_sils = inputs[0]
       new_data_list = []
       for pose, sil in zip(pose_sils[0], pose_sils[1]):
           sil = sil[:, np.newaxis, ...] # [T, 1, H, W]
           pose_h, pose_w = pose.shape[-2], pose.shape[-1]
           sil_h, sil_w = sil.shape[-2], sil.shape[-1]
           if sil_h != sil_w and pose_h == pose_w:
               cutting = (sil_h - sil_w) // 2
               pose = pose[..., cutting:-cutting]
           cat_data = np.concatenate([pose, sil], axis=1) # [T, 3, H, W]
           new_data_list.append(cat_data)
       new_inputs = [[new_data_list], inputs[1], inputs[2], inputs[3], inputs[4]]
       return super().inputs_pretreament(new_inputs)

   def forward(self, inputs):
       ipts, labs, _, _, seqL = inputs

       pose = ipts[0]
       pose = pose.transpose(1, 2).contiguous()
       assert pose.size(-1) in [44, 48, 88, 96]
       maps = pose[:, :2, ...]
       sils = pose[:, -1, ...].unsqueeze(1)

       del ipts
       map0 = self.map_layer0(maps)
       map1 = self.map_layer1(map0)
       
       sil0 = self.sil_layer0(sils)
       sil1 = self.sil_layer1(sil0)

       out1 = self.fusion(sil1, map1)
       out2 = self.layer2(out1)
       out3 = self.layer3(out2)
       out4 = self.layer4(out3) # [n, c, s, h, w]

       # Temporal Pooling, TP
       outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]
       n, c, h, w = outs.size()

       # Horizontal Pooling Matching, HPM
       feat = self.HPP(outs)  # [n, c, p]

       embed_1 = self.FCs(feat)  # [n, c, p]
       embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
       
       if self.inference_use_emb:
            embed = embed_2
       else:
            embed = embed_1

       retval = {
           'training_feat': {
               'triplet': {'embeddings': embed_1, 'labels': labs},
               'softmax': {'logits': logits, 'labels': labs}
           },
           'visual_summary': {
               'image/sils': rearrange(pose * 255., 'n c s h w -> (n s) c h w'),
           },
           'inference_feat': {
               'embeddings': embed
           }
       }
       return retval

class AttentionFusion(nn.Module): 
    def __init__(self, in_channels=64, squeeze_ratio=16):
        super(AttentionFusion, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * 2, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv3x3(hidden_dim, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, in_channels * 2), 
            )
        )
    
    def forward(self, sil_feat, map_feat): 
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
        '''
        c = sil_feat.size(1)
        feats = torch.cat([sil_feat, map_feat], dim=1)
        score = self.conv(feats) # [n, 2 * c, s, h, w]
        score = rearrange(score, 'n (d c) s h w -> n d c s h w', d=2)
        score = F.softmax(score, dim=1)
        retun = sil_feat * score[:, 0] + map_feat * score[:, 1]
        return retun

class CatFusion(nn.Module): 
    def __init__(self, in_channels=64):
        super(CatFusion, self).__init__()
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * 2, in_channels), 
            )
        )

    def forward(self, sil_feat, map_feat): 
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
        '''
        feats = torch.cat([sil_feat, map_feat])
        retun = self.conv(feats)
        return retun

class PlusFusion(nn.Module): 
    def __init__(self):
        super(PlusFusion, self).__init__()

    def forward(self, sil_feat, map_feat): 
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
        '''
        return sil_feat + map_feat
