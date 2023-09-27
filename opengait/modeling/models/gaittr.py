import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..modules import Graph, SpatialAttention
import numpy as np
import math


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x * (torch.tanh(F.softplus(x)))

class STModule(nn.Module):
    def __init__(self,in_channels, out_channels, incidence, num_point):
        super(STModule, self).__init__()
        """
        This class implements augmented graph spatial convolution in case of Spatial Transformer
        Fucntion adapated from: https://github.com/Chiaraplizz/ST-TR/blob/master/code/st_gcn/net/gcn_attention.py
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.incidence = incidence
        self.num_point = num_point
        self.relu = Mish()
        self.bn = nn.BatchNorm2d(out_channels)
        self.data_bn = nn.BatchNorm1d(self.in_channels * self.num_point)
        self.attention_conv = SpatialAttention(in_channels=in_channels,out_channel=out_channels,A=self.incidence,num_point=self.num_point)
    def forward(self,x):
        N, C, T, V = x.size()
        # data normlization
        x = x.permute(0, 1, 3, 2).reshape(N, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(N, C, V, T).permute(0, 1, 3, 2)
        # adjacency matrix
        self.incidence = self.incidence.cuda(x.get_device())
        # N, T, C, V > NT, C, 1, V
        xa = x.permute(0, 2, 1, 3).reshape(-1, C, 1, V)
        # spatial attention
        attn_out = self.attention_conv(xa)
        # N, T, C, V > N, C, T, V
        attn_out = attn_out.reshape(N, T, -1, V).permute(0, 2, 1, 3)
        y = attn_out
        y = self.bn(self.relu(y))
        return y

class UnitConv2D(nn.Module):
    '''
    This class is used in GaitTR[TCN_ST] block.
    '''

    def __init__(self, D_in, D_out, kernel_size=9, stride=1, dropout=0.1, bias=True):
        super(UnitConv2D,self).__init__()
        pad = int((kernel_size-1)/2)
        self.conv = nn.Conv2d(D_in,D_out,kernel_size=(kernel_size,1)
                            ,padding=(pad,0),stride=(stride,1),bias=bias)
        self.bn = nn.BatchNorm2d(D_out)
        self.relu = Mish()
        self.dropout = nn.Dropout(dropout, inplace=False)
        #initalize
        self.conv_init(self.conv)

    def forward(self,x):
        x = self.dropout(x)
        x = self.bn(self.relu(self.conv(x)))
        return x

    def conv_init(self,module):
        n = module.out_channels
        for k in module.kernel_size:
            n = n*k
        module.weight.data.normal_(0, math.sqrt(2. / n))

class TCN_ST(nn.Module):
    """
    Block of GaitTR: https://arxiv.org/pdf/2204.03873.pdf
    TCN: Temporal Convolution Network
    ST: Sptail Temporal Graph Convolution Network
    """
    def __init__(self,in_channel,out_channel,A,num_point):
        super(TCN_ST, self).__init__()
        #params
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.A = A
        self.num_point = num_point
        #network
        self.tcn = UnitConv2D(D_in=self.in_channel,D_out=self.in_channel,kernel_size=9)
        self.st = STModule(in_channels=self.in_channel,out_channels=self.out_channel,incidence=self.A,num_point=self.num_point)
        self.residual = lambda x: x
        if (in_channel != out_channel):
            self.residual_s = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1),
                nn.BatchNorm2d(out_channel),
            )
            self.down = UnitConv2D(D_in=self.in_channel,D_out=out_channel,kernel_size=1,dropout=0)
        else:
            self.residual_s = lambda x: x
            self.down = None

    def forward(self,x):
        x0 = self.tcn(x) + self.residual(x)
        y = self.st(x0) + self.residual_s(x0)
        # skip residual
        y = y + (x if(self.down is None) else self.down(x))
        return y



class GaitTR(BaseModel):
    """
        GaitTR: Spatial Transformer Network on Skeleton-based Gait Recognition
        Arxiv : https://arxiv.org/abs/2204.03873.pdf
    """
    def build_network(self, model_cfg):

        in_c = model_cfg['in_channels']
        self.num_class = model_cfg['num_class']
        self.joint_format = model_cfg['joint_format']
        self.graph = Graph(joint_format=self.joint_format,max_hop=3)

        #### Network Define ####

        # ajaceny matrix
        self.A = torch.from_numpy(self.graph.A.astype(np.float32))

        #data normalization
        num_point = self.A.shape[-1]
        self.data_bn = nn.BatchNorm1d(in_c[0] * num_point)
        
        #backbone
        backbone = []
        for i in range(len(in_c)-1):
            backbone.append(TCN_ST(in_channel= in_c[i],out_channel= in_c[i+1],A=self.A,num_point=num_point))
        self.backbone = nn.ModuleList(backbone)

        self.fcn = nn.Conv1d(in_c[-1], self.num_class, kernel_size=1)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        x= ipts[0] 
        pose = x
        # x = N, T, C, V, M -> N, C, T, V, M
        x = x.permute(0, 2, 1, 3, 4)
        N, C, T, V, M = x.size()
        if len(x.size()) == 4:
            x = x.unsqueeze(1)
        del ipts

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)
        #backbone
        for _,m in enumerate(self.backbone):
            x = m(x)
        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1,V))
        # M pooling
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).mean(dim=1).view(N, c, t)#[n,c,t]
        # T pooling
        x = F.avg_pool1d(x, kernel_size=x.size()[2]) #[n,c]
        # C fcn
        x = self.fcn(x) #[n,c']
        x = F.avg_pool1d(x, x.size()[2:]) # [n,c']
        x = x.view(N, self.num_class) # n,c
        embed = x.unsqueeze(-1) # n,c,1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs}
            },
            'visual_summary': {
                'image/pose': pose.view(N*T, M, V, C)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
