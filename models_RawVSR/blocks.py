"""
paper: Exploit Camera Raw Data for Video Super-Resolution via Hidden Markov Model Inference
file: blocks.py
author: Xiaohong Liu
date: 18/09/19
"""

import torch
import torch.nn as nn


class MakeDenseConv(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDenseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDBBlock layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDenseConv(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv_mid = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, padding=0)
        self.conv_out = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, padding=0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.lrelu(self.conv_in(out))
        out = self.lrelu(self.conv_mid(out))
        out = torch.sigmoid(self.conv_out(out))
        return out


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, reduction_ratio=2, merged_channels=1):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, merged_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(merged_channels, reduction_ratio ** 2 * merged_channels, kernel_size,
                               stride=reduction_ratio, padding=(kernel_size - 1) // 2)
        self.conv3 = nn.ConvTranspose2d(reduction_ratio ** 2 * merged_channels, merged_channels, kernel_size,
                                        stride=reduction_ratio, padding=(kernel_size - 1) // 2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out1 = self.lrelu(self.conv1(x))
        out2 = self.lrelu(self.conv2(out1))
        out3 = torch.sigmoid(self.conv3(out2, out1.size()))
        return out3


class SARDB(nn.Module):
    def __init__(self, in_channels, out_channels, num_rdb, num_dense_layer, growth_rate, kernel_size=3):
        super(SARDB, self).__init__()
        self.rdb_blocks = nn.ModuleDict()
        self.num_rdb = num_rdb
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        for i in range(num_rdb):
            self.rdb_blocks.update({'{}'.format(i): RDB(out_channels, num_dense_layer, growth_rate)})

        # add attention to RDBBlock
        self.CA = ChannelAttention((num_rdb + 1) * out_channels)
        self.conv_1x1 = nn.Conv2d((num_rdb + 1) * out_channels, out_channels, kernel_size=1, padding=0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x_list = [None for _ in range(self.num_rdb + 1)]
        x_list[0] = self.lrelu(self.in_conv(x))

        for i in range(self.num_rdb):
            x_list[i + 1] = self.rdb_blocks['{}'.format(i)](x_list[i])

        x_cat = torch.cat(x_list, 1)

        out_ca = self.CA(x_cat)
        out = x_cat * out_ca
        out = self.conv_1x1(out)

        return out
