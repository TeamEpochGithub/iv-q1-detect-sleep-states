from typing import Callable
import torch
from torch import nn


class SimpleTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3, emb_dim: int = 256, hidden_layers: int = 64, kernel_size: int = 7, depth: int = 2
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_layers,
                               kernel_size=kernel_size, stride=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool1d(
            kernel_size=kernel_size, stride=3, padding=0)

        self.conv2 = nn.Conv1d(in_channels=hidden_layers, out_channels=emb_dim,
                               kernel_size=kernel_size, stride=3, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.max_pool2 = nn.MaxPool1d(
            kernel_size=kernel_size, stride=3, padding=0)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.max_pool2(x)

        return x
