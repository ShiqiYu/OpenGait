import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import clones, is_list_or_tuple


class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, s, c, h, w]
            Out x: [n, s, ...]
        """
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1, c, h, w), *args, **kwargs)
        _ = x.size()
        _ = [n, s] + [*_[1:]]
        return x.view(*_)


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, seq_dim=1, **kwargs):
        """
            In  seqs: [n, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **kwargs)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(seq_dim, curr_start, curr_seqL)
            # save the memory
            # splited_narrowed_seq = torch.split(narrowed_seq, 256, dim=1)
            # ret = []
            # for seq_to_pooling in splited_narrowed_seq:
            #     ret.append(self.pooling_func(seq_to_pooling, keepdim=True, **kwargs)
            #                [0] if self.is_tuple_result else self.pooling_func(seq_to_pooling, **kwargs))
            rets.append(self.pooling_func(narrowed_seq, **kwargs))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out


class SeparateBNNecks(nn.Module):
    """
        GaitSet: Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.parallel_BN1d:
            p, n, c = x.size()
            x = x.transpose(0, 1).contiguous().view(n, -1)  # [n, p*c]
            x = self.bn1d(x)
            x = x.view(n, p, c).permute(1, 0, 2).contiguous()
        else:
            x = torch.cat([bn(_.squeeze(0)).unsqueeze(0)
                           for _, bn in zip(x.split(1, 0), self.bn1d)], 0)  # [p, n, c]
        if self.norm:
            feature = F.normalize(x, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            feature = x
            logits = feature.matmul(self.fc_bin)
        return feature, logits


class FocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def fix_BN(model):
    for module in model.modules():
        classname = module.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            module.eval()
