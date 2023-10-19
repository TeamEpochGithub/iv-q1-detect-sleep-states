from typing import Callable
import torch
from torch import nn


class ConvTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3, emb_dim: int = 256,
        conv_kernel: int = 3, conv_stride: int = 1, conv_pad: int = 1,
        pool_kernel: int = 2, pool_stride: int = 1, pool_pad: int = 0,
        activation: Callable = nn.ReLU
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=emb_dim,
                              kernel_size=conv_kernel, stride=conv_stride, padding=conv_pad)
        self.act = activation(inplace=True)
        self.max_pool = nn.MaxPool1d(
            kernel_size=pool_kernel, stride=pool_stride, padding=pool_pad)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.act(x)
        x = self.max_pool(x)
        return x
