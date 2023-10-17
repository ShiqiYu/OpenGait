import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj_size=9, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_size = adj_size

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        #self.bn = nn.BatchNorm2d(self.out_features)
        self.bn = nn.BatchNorm1d(out_features * adj_size)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output_ = torch.bmm(adj, support)
        if self.bias is not None:
            output_ =  output_ + self.bias
        output = output_.view(output_.size(0), output_.size(1)*output_.size(2))
        output = self.bn(output)
        output = output.view(output_.size(0), output_.size(1), output_.size(2))

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, adj_size, nfeat, nhid, isMeanPooling = True):
        super(GCN, self).__init__()

        self.adj_size = adj_size
        self.nhid = nhid
        self.isMeanPooling = isMeanPooling
        self.gc1 = GraphConvolution(nfeat, nhid ,adj_size)
        self.gc2 = GraphConvolution(nhid, nhid, adj_size)

    def forward(self, x, adj):
        x_ = F.dropout(x, 0.5, training=self.training) 
        x_ = F.relu(self.gc1(x_, adj))
        x_ = F.dropout(x_, 0.5, training=self.training)
        x_ = F.relu(self.gc2(x_, adj))
        return x_

