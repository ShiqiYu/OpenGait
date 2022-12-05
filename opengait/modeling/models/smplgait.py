'''
Modifed from https://github.com/Gait3D/Gait3D-Benchmark/blob/72beab994c137b902d826f4b9f9e95b107bebd78/lib/modeling/models/smplgait.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks


class SMPLGait(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):
        # Baseline
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        # for SMPL
        self.fc1 = nn.Linear(85, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]    # [n, s, h, w]
        smpls = ipts[1]   # [n, s, d]

        # extract SMPL features
        n, s, d = smpls.size()
        sps = smpls.view(-1, d)
        del smpls

        sps = F.relu(self.bn1(self.fc1(sps)))
        sps = F.relu(self.bn2(self.dropout2(self.fc2(sps))))  # (B, 256)
        sps = F.relu(self.bn3(self.dropout3(self.fc3(sps))))  # (B, 256)
        sps = sps.reshape(n, 1, s, 16, 16)
        iden = Variable(torch.eye(16)).unsqueeze(
            0).repeat(n, 1, s, 1, 1)   # [n, 1, s, 16, 16]
        if sps.is_cuda:
            iden = iden.cuda()
        sps_trans = sps + iden   # [n, 1, s, 16, 16]

        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]
        outs_n, outs_c, outs_s, outs_h, outs_w = outs.size()

        zero_tensor = Variable(torch.zeros(
            (outs_n, outs_c, outs_s, outs_h, outs_h-outs_w)))
        if outs.is_cuda:
            zero_tensor = zero_tensor.cuda()
        # [n, s, c, h, h]  [n, s, c, 16, 16]
        outs = torch.cat([outs, zero_tensor], -1)
        outs = outs.reshape(outs_n*outs_c*outs_s, outs_h,
                            outs_h)   # [n*c*s, 16, 16]

        sps = sps_trans.repeat(1, outs_c, 1, 1, 1).reshape(
            outs_n * outs_c * outs_s, 16, 16)

        outs_trans = torch.bmm(outs, sps)
        outs_trans = outs_trans.reshape(outs_n, outs_c, outs_s, outs_h, outs_h)

        # Temporal Pooling, TP
        outs_trans = self.TP(outs_trans, seqL, options={"dim": 2})[
            0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs_trans)  # [n, c, p]
        embed_1 = self.FCs(feat)  # [n, c, p]

        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed_1
            }
        }
        return retval
