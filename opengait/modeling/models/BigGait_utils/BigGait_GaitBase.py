import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ...modules import SetBlockWrapper, SeparateFCs, SeparateBNNecks, PackSequenceWrapper, HorizontalPoolingPyramid
from torch.nn import functional as F

# ######################################## GaitBase ###########################################

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class AttentionFusion(nn.Module): 
    def __init__(self, in_channels, squeeze_ratio, feat_len):
        super(AttentionFusion, self).__init__()
        hidden_dim = int(in_channels / squeeze_ratio)
        self.feat_len = feat_len
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * feat_len, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv3x3(hidden_dim, hidden_dim), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU(inplace=True), 
                conv1x1(hidden_dim, in_channels * feat_len), 
            )
        )
    
    def forward(self, feat_list): 
        '''
            sil_feat: [n, c, s, h, w]
            map_feat: [n, c, s, h, w]
            ...
        '''
        feats = torch.cat(feat_list, dim=1)
        score = self.conv(feats) # [n, 2 * c, s, h, w]
        score = rearrange(score, 'n (c d) s h w -> n c d s h w', d=self.feat_len)
        score = F.softmax(score, dim=2)
        retun = feat_list[0]*score[:,:,0]
        for i in range(1, self.feat_len):
            retun += feat_list[i]*score[:,:,i]
        return retun


from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from ...modules import BasicConv2d
block_map = {'BasicBlock': BasicBlock,
             'Bottleneck': Bottleneck}

class Pre_ResNet9(ResNet):
    def __init__(self, type, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(Pre_ResNet9, self).__init__(block, layers)

        # Not used #
        self.fc = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        ############
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=strides[0], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)

        x = self.layer1(x)
        return x

class Post_ResNet9(ResNet):
    def __init__(self, type, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        super(Post_ResNet9, self).__init__(block, layers)
        # Not used #
        self.fc = None
        self.conv1 = None
        self.bn1 = None
        self.relu = None
        self.layer1 = None
        ############
        self.inplanes = channels[0]
        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=strides[1], dilate=False)
        self.layer3 = self._make_layer(
            block, channels[2], layers[2], stride=strides[2], dilate=False)
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=strides[3], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from ... import backbones
class Baseline(nn.Module):
    def __init__(self, model_cfg):
        super(Baseline, self).__init__()
        model_cfg['backbone_cfg']['in_channel'] = model_cfg['Denoising_Branch']['target_dim']
        self.pre_part = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))

        model_cfg['backbone_cfg']['in_channel'] = model_cfg['Appearance_Branch']['target_dim']
        self.pre_rgb = SetBlockWrapper(Pre_ResNet9(**model_cfg['backbone_cfg']))

        self.post_backbone = SetBlockWrapper(Post_ResNet9(**model_cfg['backbone_cfg']))
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        self.fusion = AttentionFusion(**model_cfg['AttentionFusion'])

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def vis_forward(self, denosing, appearance, seqL):
        denosing = self.pre_part(denosing)  # [n, c, s, h, w]
        appearance = self.pre_rgb(appearance)  # [n, c, s, h, w]
        outs = self.fusion([denosing, appearance])
        return denosing, appearance, outs

    def forward(self, denosing, appearance, seqL):
        denosing = self.pre_part(denosing)  # [n, c, s, h, w]
        appearance = self.pre_rgb(appearance)  # [n, c, s, h, w]
        outs = self.fusion([denosing, appearance])
        # heat_mapt = rearrange(outs, 'n c s h w -> n s h w c')
        del denosing, appearance
        outs = self.post_backbone(outs)

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        
        # Horizontal Pooling Matching, HPM
        outs = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(outs)  # [n, c, p]
        _, logits = self.BNNecks(embed_1)  # [n, c, p]
        # return embed_1, logits, heat_mapt
        return embed_1, logits
