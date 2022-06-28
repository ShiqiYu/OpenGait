import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv2d, SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper


class GLN(BaseModel):
    """
        http://home.ustc.edu.cn/~saihui/papers/eccv2020_gln.pdf
        Gait Lateral Network: Learning Discriminative and Compact Representations for Gait Recognition
    """

    def build_network(self, model_cfg):
        in_channels = model_cfg['in_channels']
        self.bin_num = model_cfg['bin_num']
        self.hidden_dim = model_cfg['hidden_dim']
        lateral_dim = model_cfg['lateral_dim']
        reduce_dim = self.hidden_dim
        self.pretrain = model_cfg['Lateral_pretraining']

        self.sil_stage_0 = nn.Sequential(BasicConv2d(in_channels[0], in_channels[1], 5, 1, 2),
                                         nn.LeakyReLU(inplace=True),
                                         BasicConv2d(
                                             in_channels[1], in_channels[1], 3, 1, 1),
                                         nn.LeakyReLU(inplace=True))

        self.sil_stage_1 = nn.Sequential(BasicConv2d(in_channels[1], in_channels[2], 3, 1, 1),
                                         nn.LeakyReLU(inplace=True),
                                         BasicConv2d(
                                             in_channels[2], in_channels[2], 3, 1, 1),
                                         nn.LeakyReLU(inplace=True))

        self.sil_stage_2 = nn.Sequential(BasicConv2d(in_channels[2], in_channels[3], 3, 1, 1),
                                         nn.LeakyReLU(inplace=True),
                                         BasicConv2d(
                                             in_channels[3], in_channels[3], 3, 1, 1),
                                         nn.LeakyReLU(inplace=True))

        self.set_stage_1 = copy.deepcopy(self.sil_stage_1)
        self.set_stage_2 = copy.deepcopy(self.sil_stage_2)

        self.set_pooling = PackSequenceWrapper(torch.max)

        self.MaxP_sil = SetBlockWrapper(nn.MaxPool2d(kernel_size=2, stride=2))
        self.MaxP_set = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sil_stage_0 = SetBlockWrapper(self.sil_stage_0)
        self.sil_stage_1 = SetBlockWrapper(self.sil_stage_1)
        self.sil_stage_2 = SetBlockWrapper(self.sil_stage_2)

        self.lateral_layer1 = nn.Conv2d(
            in_channels[1]*2, lateral_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.lateral_layer2 = nn.Conv2d(
            in_channels[2]*2, lateral_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.lateral_layer3 = nn.Conv2d(
            in_channels[3]*2, lateral_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.smooth_layer1 = nn.Conv2d(
            lateral_dim, lateral_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.smooth_layer2 = nn.Conv2d(
            lateral_dim, lateral_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.smooth_layer3 = nn.Conv2d(
            lateral_dim, lateral_dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.HPP = HorizontalPoolingPyramid()
        self.Head = SeparateFCs(**model_cfg['SeparateFCs'])

        if not self.pretrain:
            self.encoder_bn = nn.BatchNorm1d(sum(self.bin_num)*3*self.hidden_dim)
            self.encoder_bn.bias.requires_grad_(False)

            self.reduce_dp = nn.Dropout(p=model_cfg['dropout'])
            self.reduce_ac = nn.ReLU(inplace=True)
            self.reduce_fc = nn.Linear(sum(self.bin_num)*3*self.hidden_dim, reduce_dim, bias=False)

            self.reduce_bn = nn.BatchNorm1d(reduce_dim)
            self.reduce_bn.bias.requires_grad_(False)

            self.reduce_cls = nn.Linear(reduce_dim, model_cfg['class_num'], bias=False)

    def upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='nearest') + y

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]  # [n, s, h, w]
        del ipts
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        n, _, s, h, w = sils.size()

        ### stage 0 sil ###
        sil_0_outs = self.sil_stage_0(sils)
        stage_0_sil_set = self.set_pooling(sil_0_outs, seqL, options={"dim": 2})[0]

        ### stage 1 sil ###
        sil_1_ipts = self.MaxP_sil(sil_0_outs)
        sil_1_outs = self.sil_stage_1(sil_1_ipts)

        ### stage 2 sil ###
        sil_2_ipts = self.MaxP_sil(sil_1_outs)
        sil_2_outs = self.sil_stage_2(sil_2_ipts)

        ### stage 1 set ###
        set_1_ipts = self.set_pooling(sil_1_ipts, seqL, options={"dim": 2})[0]
        stage_1_sil_set = self.set_pooling(sil_1_outs, seqL, options={"dim": 2})[0]
        set_1_outs = self.set_stage_1(set_1_ipts) + stage_1_sil_set

        ### stage 2 set ###
        set_2_ipts = self.MaxP_set(set_1_outs)
        stage_2_sil_set = self.set_pooling(sil_2_outs, seqL, options={"dim": 2})[0]
        set_2_outs = self.set_stage_2(set_2_ipts) + stage_2_sil_set

        set1 = torch.cat((stage_0_sil_set, stage_0_sil_set), dim=1)
        set2 = torch.cat((stage_1_sil_set, set_1_outs), dim=1)
        set3 = torch.cat((stage_2_sil_set, set_2_outs), dim=1)

        # print(set1.shape,set2.shape,set3.shape,"***\n")

        # lateral 
        set3 = self.lateral_layer3(set3)
        set2 = self.upsample_add(set3, self.lateral_layer2(set2))
        set1 = self.upsample_add(set2, self.lateral_layer1(set1))

        set3 = self.smooth_layer3(set3)
        set2 = self.smooth_layer2(set2)
        set1 = self.smooth_layer1(set1)

        set1 = self.HPP(set1)
        set2 = self.HPP(set2)
        set3 = self.HPP(set3)

        feature = torch.cat([set1, set2, set3], -1)

        feature = self.Head(feature)

        # compact_bloack
        if not self.pretrain:
            bn_feature = self.encoder_bn(feature.view(n, -1))
            bn_feature = bn_feature.view(*feature.shape).contiguous()

            reduce_feature = self.reduce_dp(bn_feature)
            reduce_feature = self.reduce_ac(reduce_feature)
            reduce_feature = self.reduce_fc(reduce_feature.view(n, -1))

            bn_reduce_feature = self.reduce_bn(reduce_feature)
            logits = self.reduce_cls(bn_reduce_feature).unsqueeze(1)  # n c

            reduce_feature = reduce_feature.unsqueeze(1).contiguous()
            bn_reduce_feature = bn_reduce_feature.unsqueeze(1).contiguous()

        retval = {
            'training_feat': {},
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings':  feature  # reduce_feature # bn_reduce_feature
            }
        }
        if self.pretrain:
            retval['training_feat']['triplet'] = {'embeddings': feature, 'labels': labs}
        else:
            retval['training_feat']['triplet'] = {'embeddings': feature, 'labels': labs}
            retval['training_feat']['softmax'] = {'logits': logits, 'labels': labs}
        return retval
