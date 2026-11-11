# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch


def conv1x1(
    in_planes: int, out_planes: int, stride: int = 1, bias: bool = False
) -> torch.nn.Conv2d:
    return torch.nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=bias
    )


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    bias: bool = False,
) -> torch.nn.Conv2d:
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 32,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride, bias=bias)
        self.bn1 = torch.nn.GroupNorm(groups, out_channels)
        self.relu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, bias=bias)
        self.bn2 = torch.nn.GroupNorm(groups, out_channels)

        if in_channels == out_channels and stride == 1:
            self.downsample = None
        else:
            skip_conv = conv1x1 if stride == 1 else conv3x3
            self.downsample = torch.nn.Sequential(
                skip_conv(in_channels, out_channels, stride=stride, bias=bias),
                torch.nn.GroupNorm(groups, out_channels),
            )
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
