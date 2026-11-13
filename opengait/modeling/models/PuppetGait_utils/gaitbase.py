"""Layer-group-conditioned GaitBase used by the PuppetGait recognition branches.

The first half of the feature groups share one GaitBase and the second half
share another. Original and canonical branches have independent parameters.
"""

import math

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet, conv3x3

from ...modules import (
    BasicConv2d,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateBNNecks,
    SeparateFCs,
    SetBlockWrapper,
)
from ..BigGait_utils.BigGait_GaitBase import Baseline_Single as GaitBase


class BasicBlock_Time(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1,
        base_width=64, dilation=1, norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64 or dilation > 1:
            raise ValueError("PuppetGait uses standard, non-dilated basic blocks.")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.temb_proj = nn.Sequential(nn.ReLU(), nn.Linear(256, planes))

    def forward(self, x, temb=None, use_Time=True):
        identity = x
        out = self.conv1(x)
        if temb is not None and use_Time:
            out = out + self.temb_proj(temb)[:, :, None, None]
        out = self.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ConditionedSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self:
            x = module(x, *args, **kwargs)
        return x


class Pre_ResNet9(ResNet):
    def __init__(
        self, type, block, channels, in_channel, layers, strides,
        maxpool=False,
    ):
        if block != "BasicBlock_Time":
            raise ValueError("PuppetGait requires BasicBlock_Time.")
        self.maxpool_flag = maxpool
        super().__init__(BasicBlock, layers)
        self.fc = self.layer2 = self.layer3 = self.layer4 = None
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)
        self.layer1 = self._make_layer(
            BasicBlock_Time, channels[0], layers[0], stride=strides[0],
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        layer = super()._make_layer(block, planes, blocks, stride, dilate)
        return ConditionedSequential(*layer.children())

    def forward(self, x, *args, **kwargs):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.maxpool_flag:
            x = self.maxpool(x)
        return self.layer1(x, *args, **kwargs)


class Post_ResNet9(ResNet):
    def __init__(
        self, type, block, channels, in_channel, layers, strides,
        maxpool=False,
    ):
        if block != "BasicBlock_Time":
            raise ValueError("PuppetGait requires BasicBlock_Time.")
        super().__init__(BasicBlock, layers)
        self.fc = self.conv1 = self.bn1 = self.relu = self.layer1 = None
        self.inplanes = channels[0]
        self.layer2 = self._make_layer(
            BasicBlock_Time, channels[1], layers[1], stride=strides[1],
        )
        self.layer3 = self._make_layer(
            BasicBlock_Time, channels[2], layers[2], stride=strides[2],
        )
        self.layer4 = self._make_layer(
            BasicBlock_Time, channels[3], layers[3], stride=strides[3],
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        layer = super()._make_layer(block, planes, blocks, stride, dilate)
        return ConditionedSequential(*layer.children())

    def forward(self, x, *args, **kwargs):
        x = self.layer2(x, *args, **kwargs)
        x = self.layer3(x, *args, **kwargs)
        return self.layer4(x, *args, **kwargs)


class Baseline_Single(GaitBase):
    def __init__(self, model_cfg):
        nn.Module.__init__(self)
        self.pre_rgb = SetBlockWrapper(Pre_ResNet9(**model_cfg["backbone_cfg"]))
        self.post_backbone = SetBlockWrapper(Post_ResNet9(**model_cfg["backbone_cfg"]))
        self.FCs = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks = SeparateBNNecks(**model_cfg["SeparateBNNecks"])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg["bin_num"])


def get_timestep_embedding(
    timesteps, embedding_dim, max_timesteps=40, frequency_scaling=10,
):
    half_dim = embedding_dim // 2
    frequency = math.log(max_timesteps * frequency_scaling) / (half_dim - 1)
    frequency = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -frequency)
    frequency = frequency.to(device=timesteps.device)
    embedding = timesteps.float()[:, None] * frequency[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
    if embedding_dim % 2:
        embedding = nn.functional.pad(embedding, (0, 1, 0, 0))
    return embedding


class Baseline_ShareTime_2B(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.num_FPN = model_cfg["num_FPN"]
        self.Gait_Net_1 = Baseline_Single(model_cfg)
        self.Gait_Net_2 = Baseline_Single(model_cfg)
        self.Gait_List = nn.ModuleList(
            [self.Gait_Net_1] * (self.num_FPN - self.num_FPN // 2)
            + [self.Gait_Net_2] * (self.num_FPN // 2)
        )
        self.t_channel = 256
        self.temb_proj = nn.Sequential(
            nn.Linear(self.t_channel, self.t_channel),
            nn.ReLU(),
            nn.Linear(self.t_channel, self.t_channel),
        )

    def forward(self, x, seqL):
        return self.test_2(self.test_1(x), seqL)

    def test_1(self, x):
        n, _, s, _, _ = x.shape
        features = list(torch.chunk(x, self.num_FPN, dim=1))
        groups = torch.tensor(list(range(self.num_FPN))).to(x)
        groups = groups.view(1, -1).repeat(n * s, 1)
        for index in range(self.num_FPN):
            embedding = get_timestep_embedding(
                groups[:, index], self.t_channel, max_timesteps=self.num_FPN,
            ).to(x)
            embedding = self.temb_proj(embedding)
            features[index] = self.Gait_List[index].test_1(
                features[index], temb=embedding,
            )
        return torch.cat(features, dim=1)

    def test_2(self, x, seqL):
        embeddings, logits = [], []
        for network, feature in zip(self.Gait_List, x.chunk(self.num_FPN, dim=1)):
            embedding, logit = network.test_2(feature, seqL)
            embeddings.append(embedding)
            logits.append(logit)
        return embeddings, logits
