"""Temporal-spatial convolutional encoder for TF EEG maps."""

from __future__ import annotations

import torch
import torch.nn as nn


class _TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act2(x + y)


class TemporalEncoder(nn.Module):
    """EEGNet-style temporal-spatial frontend over TF maps.

    Expected input shape: (B, 1, H, T)
    Output shape: (B, 64, T')
    """

    def __init__(self, in_height: int) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(in_height, 1), groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
        )
        self.pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.tcn = nn.Sequential(
            _TemporalResidualBlock(64, dilation=1),
            _TemporalResidualBlock(64, dilation=2),
            _TemporalResidualBlock(64, dilation=4),
            _TemporalResidualBlock(64, dilation=8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.refine(x)
        x = self.pool(x)
        x = x.squeeze(2)
        x = self.tcn(x)
        return x
