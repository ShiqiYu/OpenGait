import torch
import torch.nn as nn

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D
import torch.nn.functional as F

from einops import rearrange
import copy
import cv2
from kornia import morphology as morph

blocks_map = {
    '2d': BasicBlock2D, 
    'p3d': BasicBlockP3D, 
    '3d': BasicBlock3D
}

class MultiGaitpp(BaseModel):

    def build_network(self, model_cfg):
        in_C, B, C = model_cfg['Backbone']['in_channels'], model_cfg['Backbone']['blocks'], model_cfg['Backbone']['C']
        self.part1 = model_cfg['Backbone']['part1_channel']
        self.part2 = model_cfg['Backbone']['part2_channel']

 
        self.inplanes = 32 * C
        self.part1_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(self.part1, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))
 
        self.part2_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(self.part2, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))
 
        self.part1_layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, 32 * C, stride=[1, 1], blocks_num=B[0], mode='2d'))
        self.part2_layer1 = copy.deepcopy(self.part1_layer1)
 
        self.fusion = CatFusion(256)
 
        self.part1_layer2 = self.make_layer(BasicBlockP3D, 64 * C, stride=[2, 2], blocks_num=B[1], mode='p3d')
        self.part2_layer2 = copy.deepcopy(self.part1_layer2)
        self.layer2 = copy.deepcopy(self.part1_layer2)
        self.part1_layer3 = self.make_layer(BasicBlockP3D, 128 * C, stride=[2, 2], blocks_num=B[2], mode='p3d')
        self.part2_layer3 = copy.deepcopy(self.part1_layer3)
        self.layer3 = copy.deepcopy(self.part1_layer3)
        self.layer4 = self.make_layer(BasicBlockP3D, 256 * C, stride=[1, 1], blocks_num=B[3], mode='p3d')
        self.csquare = CSquare(64)


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

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        if len(ipts[0].size()) == 4:
            ipts = ipts[0].unsqueeze(1)
        else:
            ipts = ipts[0].transpose(1, 2).contiguous()

        part1 = ipts[:, :self.part1, ...]
        part2 = ipts[:, self.part1:, ...]
 
        del ipts
        part2 = self.part2_layer0(part2)
        part2 = self.part2_layer1(part2)
        part1 = self.part1_layer0(part1)
        part1 = self.part1_layer1(part1)
        out, attn1, attn2, attn_co = self.csquare(part2,part1)

        part2 = self.part2_layer2(part2*attn1)
        part1 = self.part1_layer2(part1*attn2)
        out = self.layer2(out)

        part2 = self.part2_layer3(part2)
        part1 = self.part1_layer3(part1)
        out = self.layer3(out)
        
        out = self.fusion([part1, out, part2])
        out = self.layer4(out)
        outs = self.TP(out, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        feat = self.HPP(outs)  # [n, c, p]
 
        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
 
        embed = embed_1
 
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval



class CatFusion(nn.Module): 
    def __init__(self, in_channels=64):
        super(CatFusion, self).__init__()
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * 3, in_channels), 
            )
        )

    def forward(self, feat_list): 
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
        '''
        # print(feat_list.shape)
        feats = torch.cat(feat_list, dim=1)
        retun = self.conv(feats)
        return retun


class CSquare(nn.Module): 
    def __init__(self, in_channels=64, squeeze_ratio=16, h=32, w=22):
        super(CSquare, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.TP_mean = PackSequenceWrapper(torch.mean)
        self.conv2 = SetBlockWrapper(nn.Sequential(
                conv1x1(in_channels, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, in_channels), 
            ))
        self.conv1 = SetBlockWrapper(nn.Sequential(
                conv1x1(in_channels, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, in_channels), 
            ))
        self.kernel = torch.ones((3,3))


    def channel_normalization(self, masked_attn):
        min_vals = masked_attn.min(dim=1, keepdim=True).values
        max_vals = masked_attn.max(dim=1, keepdim=True).values
        min_vals = min_vals.expand_as(masked_attn)
        max_vals = max_vals.expand_as(masked_attn)
        attn_norm = (masked_attn - min_vals) / (max_vals - min_vals + 1e-6)
        attn_norm = attn_norm.clamp(0, 1)
        return attn_norm

    def forward(self, x1, x2): 
        '''
        x1 [n, c, s, h, w]
        x2 [n, c, s, h, w] shape
        '''
        t = x2.size(2)
        attn_x2 = self.conv2(x2) 
        n, c, t, h, w = attn_x2.size()
        
        attn_x1 = self.conv1(x1) # [n, c, h, w]
        attn_x = torch.stack((attn_x1, attn_x2), dim=1)

        attn_x = F.softmax(attn_x, dim=1)
        attn_x1_softmax = attn_x[:, 0, ...]
        attn_x2_softmax = attn_x[:, 1, ...]
        attn_ = torch.min(attn_x1_softmax,attn_x2_softmax) #* mask

        attn = self.channel_normalization(attn_)
        attn_co = rearrange(attn, 'n c s h w -> (n s) c h w')

        return (x1+x2)/2 * attn, (1.-attn)*attn_x1_softmax, (1.-attn)*attn_x2_softmax, attn_co #87.2


