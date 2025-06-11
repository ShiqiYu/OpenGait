import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable
import numpy as np


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def ball_query(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    xyz = xyz[:,:,:3]
    new_xyz = new_xyz[:,:,:3]
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_query(k, xyz, new_xyz):
    """
    Input:
        k: number of nearest neighbors to query
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: indices of k-nearest neighbors, [B, S, k]
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    xyz = xyz[:,:,:3]
    new_xyz = new_xyz[:,:,:3]
    dists = square_distance(new_xyz, xyz)
    #scaling_factor = torch.Tensor([1, 1, 0.6]).to(new_xyz.device)
    #dists = torch.sum(torch.square(new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) / scaling_factor, dim=-1)
    group_idx = dists.sort(dim=-1)[1][:, :, :k]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, sampling='ball',scale_aware=False, normalize_dp=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz[:,:,:3], npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)

    if sampling == 'ball':
        idx =  ball_query(radius, nsample, xyz, new_xyz)
    elif sampling == 'knn':
        idx =  knn_query(nsample, xyz, new_xyz)
    else:
        raise ValueError("Unsupported sampling type. Use 'ball' or 'knn'.")

    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if normalize_dp:  # and sampling!='knn':
        grouped_xyz_norm /= radius
    grouped_xyz_norm = grouped_xyz_norm if scale_aware else grouped_xyz_norm[:,:,:,:3]

    
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points, scale_aware=False):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    grouped_xyz = grouped_xyz if scale_aware else grouped_xyz[:,:,:,:3]
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, sampling='ball', scale_aware=False,normalize_dp=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.scale_aware = scale_aware
        self.normalize_dp = normalize_dp
        self.sampling = sampling
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points, scale_aware=self.scale_aware)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, sampling=self.sampling, scale_aware=self.scale_aware,normalize_dp=self.normalize_dp)
        # new_xyz: sampled points position data, [B, ], C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = new_points
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PPPooling_UDP():
    """
        Hierarchically Clustered Point Pooling
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x, xyz):
        """
            x  : [n, c, h, w]
            xyz: [n, 3, p]
            ret: [n, c, p] 
        """
        #print(xyz.shape)
        #x = rearrange(x, 'b n c -> b c n 1')
        n, c = x.size()[:2]
        _, idx = xyz[:, 2, :].sort()
        x = x.gather(2, idx.unsqueeze(1).unsqueeze(-1).expand_as(x))
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


class PPPooling():
    def __init__(self, scale_aware=False, bin_num=None):
        # 默认设置多个分辨率的分bin数量
        self.bin_num = bin_num if bin_num is not None else [16, 8, 4, 2, 1]
        self.scale_aware = scale_aware

    def __call__(self, point_clouds, points):
        # 调整维度：输入 point_clouds: B x C x N x 1 转换为 B x N x C，
        # points: B x C x N 转换为 B x N x C
        point_clouds = rearrange(point_clouds, 'B C N 1 -> B N C')
        points = rearrange(points, 'B C N -> B N C')
        B, N, C = point_clouds.shape

        if self.scale_aware: # PPPooling_HAP
            z = points[:, :, 3]  # shape: (B, N)
            # 固定的 z 范围（例如：0 到 2）
            z_min, z_max = 0.0, 2.0
        else:
            # PPPooling_UAP
            # 使用 points 的第 3 个通道作为 z 坐标，归一化到 [0, 1]
            z = points[:, :, 2]  # shape: (B, N)
            z_min = z.min(dim=1, keepdim=True)[0][0].item()
            z_max = z.max(dim=1, keepdim=True)[0][0].item()
            z_range = z_max - z_min + 1e-6
            z = (z - z_min) / z_range  # shape: (B, N)
            z_min, z_max = 0.0, 1.0

        all_pooled = []
        for M in self.bin_num:
            # 由于 z 已归一化，直接构造均匀分布的 bin 边界
            edges = torch.linspace(z_min, z_max, steps=M + 1, device=point_clouds.device)
            # 利用 bucketize 将每个点分配到 [0, M-1] 内的 bin（不需要额外处理首尾）
            # 注意：这里使用 edges[1:-1] 作为分界，保证边界值归到正确 bin
            bin_idx = torch.bucketize(z.contiguous(), edges[1:-1], right=False)  # shape: (B, N)

            # 为每个 bin计算 max 和 mean 池化值，利用 scatter_reduce 与 scatter_add 操作：
            # 构造初始 tensor，形状均为 (B, M, C)
            pooled_max = torch.full((B, M, C), float('-inf'), device=point_clouds.device, dtype=point_clouds.dtype)
            pooled_sum = torch.zeros((B, M, C), device=point_clouds.device, dtype=point_clouds.dtype)
            counts = torch.zeros((B, M, 1), device=point_clouds.device, dtype=point_clouds.dtype)

            # 将 bin_idx 扩展到与 point_clouds 对应的维度 (B, N, C)
            bin_idx_exp = bin_idx.unsqueeze(-1).expand(-1, -1, C)
            # max 池化：scatter_reduce 计算每个 bin 内的最大值
            pooled_max = pooled_max.scatter_reduce(1, bin_idx_exp, point_clouds, reduce='amax', include_self=True)
            # sum 池化：scatter_add 计算每个 bin 内的和
            pooled_sum = pooled_sum.scatter_add(1, bin_idx_exp, point_clouds)
            # 计算每个 bin 的计数
            counts = counts.scatter_add(1, bin_idx.unsqueeze(-1), torch.ones((B, N, 1), device=point_clouds.device))
            # 计算 mean 池化
            pooled_mean = pooled_sum / counts.clamp(min=1)
            # 这里采用 max 与 mean 的和作为最终池化结果（也可以用 concat）
            pooled = pooled_max + pooled_mean
            # 将没有点（max为 -inf）的 bin 置 0
            pooled[pooled == float('-inf')] = 0

            all_pooled.append(pooled)

        # 将各分辨率下的池化结果在 bin 维度上拼接，并调整为 B x C x M_total
        output = torch.cat(all_pooled, dim=1)
        output = rearrange(output, 'B M C -> B C M')
        return output


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x, xyz):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten  b num c -> b num c
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad.unsqueeze(-1)
