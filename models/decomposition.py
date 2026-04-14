import math

import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(
            1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1
        )
        end = x[:, -1:, :].repeat(
            1, math.floor((self.kernel_size - 1) // 2), 1
        )
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Decompose time series into seasonal and trend components.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend
