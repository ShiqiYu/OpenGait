import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..base_model import BaseModel

class MultiScaleGaitGraph(BaseModel):
    """
        Learning Rich Features for Gait Recognition by Integrating Skeletons and Silhouettes
        Github: https://github.com/YunjiePeng/BimodalFusion
    """

    def build_network(self, model_cfg):
        in_c = model_cfg['in_channels']
        out_c = model_cfg['out_channels']
        num_id = model_cfg['num_id']

        temporal_kernel_size = model_cfg['temporal_kernel_size']

        # load spatial graph
        self.graph = SpatialGraph(**model_cfg['graph_cfg'])
        A_lowSemantic = torch.tensor(self.graph.get_adjacency(semantic_level=0), dtype=torch.float32, requires_grad=False)
        A_mediumSemantic =  torch.tensor(self.graph.get_adjacency(semantic_level=1), dtype=torch.float32, requires_grad=False)
        A_highSemantic = torch.tensor(self.graph.get_adjacency(semantic_level=2), dtype=torch.float32, requires_grad=False)

        self.register_buffer('A_lowSemantic', A_lowSemantic)
        self.register_buffer('A_mediumSemantic', A_mediumSemantic)
        self.register_buffer('A_highSemantic', A_highSemantic)

        # build networks
        spatial_kernel_size = self.graph.num_A
        temporal_kernel_size = temporal_kernel_size
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.st_gcn_networks_lowSemantic = nn.ModuleList()
        self.st_gcn_networks_mediumSemantic = nn.ModuleList()
        self.st_gcn_networks_highSemantic = nn.ModuleList()
        for i in range(len(in_c)-1):
            if i == 0:
                self.st_gcn_networks_lowSemantic.append(st_gcn_block(in_c[i], in_c[i+1], kernel_size, 1, residual=False))
                self.st_gcn_networks_mediumSemantic.append(st_gcn_block(in_c[i], in_c[i+1], kernel_size, 1, residual=False))
                self.st_gcn_networks_highSemantic.append(st_gcn_block(in_c[i], in_c[i+1], kernel_size, 1, residual=False))
            else:
                self.st_gcn_networks_lowSemantic.append(st_gcn_block(in_c[i], in_c[i+1], kernel_size, 1))
                self.st_gcn_networks_mediumSemantic.append(st_gcn_block(in_c[i], in_c[i+1], kernel_size, 1))
                self.st_gcn_networks_highSemantic.append(st_gcn_block(in_c[i], in_c[i+1], kernel_size, 1))

            self.st_gcn_networks_lowSemantic.append(st_gcn_block(in_c[i+1], in_c[i+1], kernel_size, 1))
            self.st_gcn_networks_mediumSemantic.append(st_gcn_block(in_c[i+1], in_c[i+1], kernel_size, 1))
            self.st_gcn_networks_highSemantic.append(st_gcn_block(in_c[i+1], in_c[i+1], kernel_size, 1))

        self.edge_importance_lowSemantic = nn.ParameterList([
            nn.Parameter(torch.ones(self.A_lowSemantic.size()))
            for i in self.st_gcn_networks_lowSemantic])

        self.edge_importance_mediumSemantic = nn.ParameterList([
            nn.Parameter(torch.ones(self.A_mediumSemantic.size()))
            for i in self.st_gcn_networks_mediumSemantic])

        self.edge_importance_highSemantic = nn.ParameterList([
            nn.Parameter(torch.ones(self.A_highSemantic.size()))
            for i in self.st_gcn_networks_highSemantic])

        self.fc = nn.Linear(in_c[-1], out_c)
        self.bn_neck = nn.BatchNorm1d(out_c)
        self.encoder_cls = nn.Linear(out_c, num_id, bias=False)

    def semantic_pooling(self, x):
        cur_node_num = x.size()[-1]
        half_x_1, half_x_2 = torch.split(x, int(cur_node_num / 2), dim=-1)
        x_sp = torch.add(half_x_1, half_x_2) / 2
        return x_sp

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        
        x = ipts[0]  # [N, T, V, C]
        del ipts
        """
           N - the number of videos.
           T - the number of frames in one video.
           V - the number of keypoints.
           C - the number of features for one keypoint.
        """
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, C, T, V)

        y = self.semantic_pooling(x)
        z = self.semantic_pooling(y)
        for gcn_lowSemantic, importance_lowSemantic, gcn_mediumSemantic, importance_mediumSemantic, gcn_highSemantic, importance_highSemantic in zip(self.st_gcn_networks_lowSemantic, self.edge_importance_lowSemantic, self.st_gcn_networks_mediumSemantic, self.edge_importance_mediumSemantic, self.st_gcn_networks_highSemantic, self.edge_importance_highSemantic):
            x, _ = gcn_lowSemantic(x, self.A_lowSemantic * importance_lowSemantic)
            y, _ = gcn_mediumSemantic(y, self.A_mediumSemantic * importance_mediumSemantic)
            z, _ = gcn_highSemantic(z, self.A_highSemantic * importance_highSemantic)

            # Cross-scale Message Passing
            x_sp = self.semantic_pooling(x)
            y = torch.add(y, x_sp)
            y_sp = self.semantic_pooling(y)
            z = torch.add(z, y_sp)
        
        # global pooling for each layer
        x_sp = F.avg_pool2d(x, x.size()[2:])
        N, C, T, V = x_sp.size()
        x_sp = x_sp.view(N, C, T*V).contiguous()

        y_sp = F.avg_pool2d(y, y.size()[2:])
        N, C, T, V = y_sp.size()
        y_sp = y_sp.view(N, C, T*V).contiguous()

        z = F.avg_pool2d(z, z.size()[2:])
        N, C, T, V = z.size()
        z = z.permute(0, 2, 3, 1).contiguous()
        z = z.view(N, T*V, C)

        z_fc = self.fc(z.view(N, -1))
        bn_z_fc = self.bn_neck(z_fc)
        z_cls_score = self.encoder_cls(bn_z_fc)

        z_fc = z_fc.unsqueeze(-1).contiguous() # [n, c, p]
        z_cls_score = z_cls_score.unsqueeze(-1).contiguous() # [n, c, p]

        retval = {
            'training_feat': {
                'triplet_joints': {'embeddings': x_sp, 'labels': labs},
                'triplet_limbs': {'embeddings': y_sp, 'labels': labs},
                'triplet_bodyparts': {'embeddings': z_fc, 'labels': labs},
                'softmax': {'logits': z_cls_score, 'labels': labs}
            },
            'visual_summary': {},
            'inference_feat': {
                'embeddings': z_fc
            }
        }
        return retval

class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size, i.e. the number of videos.
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`.
            :math:`T_{in}/T_{out}` is a length of input/output sequence, i.e. the number of frames in a video.
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = SCN(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

class SCN(nn.Module):
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()
        # The defined module SCN are responsible only for the Spacial Graph (i.e. the graph in on frame),
        # and the parameter t_kernel_size in this situation is always set to 1.

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)
        """
        The 1x1 conv operation here stands for the weight metrix W.
        The kernel_size here stands for the number of different adjacency matrix, 
            which are defined according to the partitioning strategy.
        Because for neighbor nodes in the same subset (in one adjacency matrix), the weights are shared. 
        It is reasonable to apply 1x1 conv as the implementation of weight function.
        """


    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A

class SpatialGraph():
    """ Use skeleton sequences extracted by Openpose/HRNet to construct Spatial-Temporal Graph

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration Partitioning
        - gait_temporal: Gait Temporal Configuration Partitioning
            For more information, please refer to the section 'Partition Strategies' in PGG.
        layout (string): must be one of the follow candidates
        - body_12: Is consists of 12 joints.
            (right shoulder, right elbow, right knee, right hip, left elbow, left knee,
             left shoulder, right wrist, right ankle, left hip, left wrist, left ankle).
            For more information, please refer to the section 'Data Processing' in PGG.
        max_hop (int): the maximal distance between two connected nodes # 1-neighbor
        dilation (int): controls the spacing between the kernel points
    """
    def __init__(self,
                 layout='body_12', # Openpose here represents for body_12
                 strategy='spatial',
                 semantic_level=0,
                 max_hop=1,
                 dilation=1):
        self.layout = layout
        self.strategy = strategy
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node, self.neighbor_link_dic = self.get_layout_info(layout)
        self.num_A = self.get_A_num(strategy)

    def __str__(self):
        return self.A

    def get_A_num(self, strategy):
        if self.strategy == 'uniform':
            return 1
        elif self.strategy == 'distance':
            return 2
        elif (self.strategy == 'spatial') or (self.strategy == 'gait_temporal'):
            return 3
        else:
            raise ValueError("Do Not Exist This Strategy")

    def get_layout_info(self, layout):
        if layout == 'body_12':
            num_node = 12
            neighbor_link_dic = {
                0: [(7, 1), (1, 0), (10, 4), (4, 6),
                     (8, 2), (2, 3), (11, 5), (5, 9),
                     (9, 3), (3, 0), (9, 6), (6, 0)],
                1: [(1, 0), (4, 0), (0, 3), (2, 3), (5, 3)],
                2: [(1, 0), (2, 0)]
            }
            return num_node, neighbor_link_dic
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_edge(self, semantic_level):
        # edge is a list of [child, parent] pairs, regarding the center node as root node
        self_link = [(i, i) for i in range(int(self.num_node / (2 ** semantic_level)))]
        neighbor_link = self.neighbor_link_dic[semantic_level]
        edge = self_link + neighbor_link
        center = []
        if self.layout == 'body_12':
            if semantic_level == 0:
                center = [0, 3, 6, 9]
            elif semantic_level == 1:
                center = [0, 3]
            elif semantic_level == 2:
                center = [0]
        return edge, center

    def get_gait_temporal_partitioning(self, semantic_level):
        if semantic_level == 0:
            if self.layout == 'body_12':
                positive_node = {1, 2, 4, 5, 7, 8, 10, 11}
                negative_node = {0, 3, 6, 9}
        elif semantic_level == 1:
            if self.layout == 'body_12':
                positive_node = {1, 2, 4, 5}
                negative_node = {0, 3}
        elif semantic_level == 2:
            if self.layout == 'body_12':
                positive_node = {1, 2}
                negative_node = {0}
        return positive_node, negative_node
            
    def get_adjacency(self, semantic_level):
        edge, center = self.get_edge(semantic_level)
        num_node = int(self.num_node / (2 ** semantic_level))
        hop_dis = get_hop_distance(num_node, edge, max_hop=self.max_hop)
                
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((num_node, num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1

        normalize_adjacency = normalize_digraph(adjacency)
        # normalize_adjacency = adjacency # withoutNodeNorm

        # normalize_adjacency[a][b] = x
        # when x = 0, node b has no connection with node a within valid hop.
        # when x ≠ 0, the normalized adjacency from node b to node a is x.
        # the value of x is normalized by the number of adjacent neighbor nodes around the node b.

        if self.strategy == 'uniform':
            A = np.zeros((1, num_node, num_node))
            A[0] = normalize_adjacency
            return A
        elif self.strategy == 'distance':
            A = np.zeros((len(valid_hop), num_node, num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
            return A
        elif self.strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((num_node, num_node))
                a_close = np.zeros((num_node, num_node))
                a_further = np.zeros((num_node, num_node))
                for i in range(num_node):
                    for j in range(num_node):
                        if hop_dis[j, i] == hop:
                            j_hop_dis = min([hop_dis[j, _center] for _center in center])
                            i_hop_dis = min([hop_dis[i, _center] for _center in center])
                            if j_hop_dis == i_hop_dis:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif j_hop_dis > i_hop_dis:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
            return A
        elif self.strategy == 'gait_temporal':
            A = []
            positive_node, negative_node = self.get_gait_temporal_partitioning(semantic_level)
            for hop in valid_hop:
                a_root = np.zeros((num_node, num_node))
                a_positive = np.zeros((num_node, num_node))
                a_negative = np.zeros((num_node, num_node))
                for i in range(num_node):
                    for j in range(num_node):
                        if hop_dis[j, i] == hop:
                            if i == j:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif j in positive_node:
                                a_positive[j, i] = normalize_adjacency[j, i]
                            else:
                                a_negative[j, i] = normalize_adjacency[j, i]
                
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_negative)
                    A.append(a_positive)
            A = np.stack(A)
            return A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    # Calculate the shortest path between nodes
    # i.e. The minimum number of steps needed to walk from one node to another
    A = np.zeros((num_node, num_node)) # Ajacent Matrix
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
