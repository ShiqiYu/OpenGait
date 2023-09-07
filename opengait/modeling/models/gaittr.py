import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..modules import Graph
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
        self.attention_conv = spatial_attention(in_channels=in_channels,out_channel=out_channels,A=self.incidence,num_point=self.num_point)
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

class spatial_attention(nn.Module):
    """
    This class implements Spatial Transformer. 
    Function adapted from: https://github.com/leaderj1001/Attention-Augmented-Conv2d
    """
    def __init__(self, in_channels, out_channel, A, num_point, dk_factor=0.25, kernel_size=1, Nh=8, num=4, stride=1):
        super(spatial_attention, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dk = int(dk_factor * out_channel)
        self.dv = int(out_channel)
        self.num = num
        self.Nh = Nh
        self.num_point=num_point
        self.A = A[0] + A[1] + A[2]
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size,
                                    stride=stride,
                                    padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

    def forward(self, x):
        # Input x
        # (batch_size, channels, 1, joints)
        B, _, T, V = x.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, dvh or dkh, joints)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v obtained by doing 2D convolution on the input (q=XWq, k=XWk, v=XWv)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)

        # Calculate the scores, obtained by doing q*k
        # (batch_size, Nh, joints, dkh)*(batch_size, Nh, dkh, joints) =  (batch_size, Nh, joints,joints)
        # The multiplication can also be divided (multi_matmul) in case of space problems

        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, joints, dvh)
        # weights*V
        # (batch, Nh, joints, joints)*(batch, Nh, joints, dvh)=(batch, Nh, joints, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))

        attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.Nh))

        attn_out = attn_out.permute(0, 1, 4, 2, 3)

        # combine_heads_2d, combine heads only after having calculated each Z separately
        # (batch, Nh*dv, 1, joints)
        attn_out = self.combine_heads_2d(attn_out)

        # Multiply for W0 (batch, out_channels, 1, joints) with out_channels=dv
        attn_out = self.attn_out(attn_out)
        return attn_out

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        # T=1 in this case, because we are considering each frame separately
        N, _, T, V = qkv.size()

        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q*(dkh ** -0.5)
        flat_q = torch.reshape(q, (N, Nh, dkh, T * V))
        flat_k = torch.reshape(k, (N, Nh, dkh, T * V))
        flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, T * V))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        B, channels, T, V = x.size()
        ret_shape = (B, Nh, channels // Nh, T, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, T, V = x.size()
        ret_shape = (batch, Nh * dv, T, V)
        return torch.reshape(x, ret_shape)

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
