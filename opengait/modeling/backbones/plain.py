"""The plain backbone.

    The plain backbone only contains the BasicConv2d, FocalConv2d and MaxPool2d and LeakyReLU layers.
"""

import torch.nn as nn
from ..modules import BasicConv2d, FocalConv2d


class Plain(nn.Module):
    """
    The Plain backbone class.

    An implicit LeakyRelu appended to each layer except maxPooling. 
    The kernel size, stride and padding of the first convolution layer are 5, 1, 2, the ones of other layers are 3, 1, 1.

    Typical usage: 
    - BC-64: Basic conv2d with output channel 64. The input channel is the output channel of previous layer.

    - M: nn.MaxPool2d(kernel_size=2, stride=2)].

    - FC-128-1: Focal conv2d with output channel 64 and halving 1(divided to 2^1=2 parts).

    Use it in your configuration file.
    """

    def __init__(self, layers_cfg, in_channels=1):
        super(Plain, self).__init__()
        self.layers_cfg = layers_cfg
        self.in_channels = in_channels

        self.feature = self.make_layers()

    def forward(self, seqs):
        out = self.feature(seqs)
        return out

    def make_layers(self):
        """
        Reference: torchvision/models/vgg.py
        """
        def get_layer(cfg, in_c, kernel_size, stride, padding):
            cfg = cfg.split('-')
            typ = cfg[0]
            if typ not in ['BC', 'FC']:
                raise ValueError('Only support BC or FC, but got {}'.format(typ))
            out_c = int(cfg[1])

            if typ == 'BC':
                return BasicConv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
            return FocalConv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, halving=int(cfg[2]))

        Layers = [get_layer(self.layers_cfg[0], self.in_channels,
                            5, 1, 2), nn.LeakyReLU(inplace=True)]
        in_c = int(self.layers_cfg[0].split('-')[1])
        for cfg in self.layers_cfg[1:]:
            if cfg == 'M':
                Layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = get_layer(cfg, in_c, 3, 1, 1)
                Layers += [conv2d, nn.LeakyReLU(inplace=True)]
                in_c = int(cfg.split('-')[1])
        return nn.Sequential(*Layers)
