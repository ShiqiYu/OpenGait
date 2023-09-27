import torch
import torch.nn as nn
from ..modules import TemporalBasicBlock, TemporalBottleneckBlock, SpatialBasicBlock, SpatialBottleneckBlock

class ResGCNModule(nn.Module):
    """
        ResGCNModule
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
                https://github.com/BNU-IVC/FastPoseGait
    """
    def __init__(self, in_channels, out_channels, block, A, stride=1, kernel_size=[9,2],reduction=4, get_res=False,is_main=False):
        super(ResGCNModule, self).__init__()

        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        if  block == 'initial':
            module_res, block_res = False, False
        elif block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            # stride =2
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )
        
        if block in ['Basic','initial']:
            spatial_block = SpatialBasicBlock
            temporal_block = TemporalBasicBlock
        if block == 'Bottleneck':
            spatial_block = SpatialBottleneckBlock
            temporal_block = TemporalBottleneckBlock
        self.scn = spatial_block(in_channels, out_channels, max_graph_distance, block_res,reduction)
        if in_channels == out_channels and is_main:
            tcn_stride =True
        else:
            tcn_stride = False
        self.tcn = temporal_block(out_channels, temporal_window_size, stride, block_res,reduction,get_res=get_res,tcn_stride=tcn_stride)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, A):
        A = A.cuda(x.get_device())
        return self.tcn(self.scn(x, A*self.edge), self.residual(x))

class ResGCNInputBranch(nn.Module):
    """
        ResGCNInputBranch_Module
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, input_branch, block, A, input_num , reduction = 4):
        super(ResGCNInputBranch, self).__init__()

        self.register_buffer('A', A)

        module_list = []
        for i in range(len(input_branch)-1):
            if i==0:
                module_list.append(ResGCNModule(input_branch[i],input_branch[i+1],'initial',A, reduction=reduction))
            else:
                module_list.append(ResGCNModule(input_branch[i],input_branch[i+1],block,A,reduction=reduction))
        

        self.bn = nn.BatchNorm2d(input_branch[0])
        self.layers = nn.ModuleList(module_list)

    def forward(self, x):

        x = self.bn(x)
        for layer in self.layers:
            x = layer(x, self.A)

        return x
    
    
class ResGCN(nn.Module):
    """
        ResGCN
        Arxiv: https://arxiv.org/abs/2010.09978
    """
    def __init__(self, input_num, input_branch, main_stream,num_class, reduction, block, graph):
        super(ResGCN, self).__init__()
        self.graph = graph
        self.head= nn.ModuleList(
            ResGCNInputBranch(input_branch, block, graph, input_num ,reduction)
            for _ in range(input_num)
        )
        
        main_stream_list = []
        for i in range(len(main_stream)-1):
            if main_stream[i]==main_stream[i+1]:
                stride = 1
            else:
                stride = 2
            if i ==0:
                main_stream_list.append(ResGCNModule(main_stream[i]*input_num,main_stream[i+1],block,graph,stride=1,reduction = reduction,get_res=True,is_main=True))
            else:
                main_stream_list.append(ResGCNModule(main_stream[i],main_stream[i+1],block,graph,stride = stride, reduction = reduction,is_main=True))
        self.backbone = nn.ModuleList(main_stream_list)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Linear(256, num_class)

    def forward(self, x):
        # input branch
        x_cat = []
        for i, branch in enumerate(self.head):
            x_cat.append(branch(x[:, i]))
        x = torch.cat(x_cat, dim=1)

        # main stream
        for layer in self.backbone:
            x = layer(x, self.graph)

        # output
        x = self.global_pooling(x)
        x = x.squeeze(-1)
        x = self.fcn(x.squeeze((-1)))
        
        return x