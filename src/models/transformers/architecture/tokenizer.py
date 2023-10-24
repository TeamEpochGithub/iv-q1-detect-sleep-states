from typing import Callable
import torch
from torch import nn

class PatchTokenizer(nn.Module):
    def __init__(self, channels: int = 2, emb_dim: int = 192, patch_size: int = 36):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.linear_projection = nn.Linear(
            self.patch_size*channels, emb_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for patch tokenizer.

        Args:
            x: (batch_size, seq_len, channels)
        """
        x = x.permute(0, 2, 1)

        bs, c, l = x.shape  # (bs, c, l)
        torch._assert(
            l % self.patch_size == 0, 
            "Sequence not disivible by patch_size"
        )

        # below 2 lines creates patches using view/reshape and permutations
        # (bs, c, no_of_patches, patch_l) -> (bs, no_of_patches, c, patch_l)
        x = x.view(bs, c, l // self.patch_size, self.patch_size).permute(0, 2, 1, 3)
        # (bs, no_of_patches, c*seq_len)
        x = x.reshape(bs, l // self.patch_size, c*self.patch_size)

        # linear projection to embedding dimension
        x = self.linear_projection(x)
        return x


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
    

class ConvTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3, emb_dim: int = 256, hidden_layers: int = 64, kernel_size: int = 3, depth: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = emb_dim
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.depth = depth

        self.AvgPool1D1 = nn.AvgPool1d(kernel_size=5, stride=4, padding=1)
        self.AvgPool1D2 = nn.AvgPool1d(kernel_size=5, stride=16, padding=1)

        self.layer1 = self.down_layer(
            self.in_channels, self.hidden_layers, self.kernel_size, 1, 2)
        self.layer2 = self.down_layer(self.hidden_layers, int(
            self.hidden_layers * 2), self.kernel_size, 4, 2)
        self.layer3 = self.down_layer(int(self.hidden_layers * 2) + int(
            self.in_channels), int(self.hidden_layers * 3), self.kernel_size, 4, 2)
        self.layer4 = self.down_layer(int(self.hidden_layers * 3) + int(
            self.in_channels), emb_dim, self.kernel_size, 4, 2)

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(ConBrBlock(input_layer, out_layer, kernel, stride, 1))
        for _ in range(depth):
            block.append(ReBlock(out_layer, out_layer, kernel, 1))
        return nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)

        # Encoder
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)

        x = torch.cat([out_1, pool_x1], 1)
        out_2 = self.layer3(x)

        x = torch.cat([out_2, pool_x2], 1)
        out = self.layer4(x)

        return out


class ConBrBlock(nn.Module):
    """
    Convolution + Batch Normalization + ReLU Block.

    Args:
        in_layer: number of input channels
        out_layer: number of output channels
        kernel_size: kernel size for convolution
        stride: stride for convolution
        dilation: dilation for convolution
    """

    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(ConBrBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size,
                               stride=stride, dilation=dilation, padding=3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)

        return out


class SeBlock(nn.Module):
    """
    Squeeze and Excitation Block.
    """

    def __init__(self, in_layer, out_layer):
        super(SeBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer // 8,
                               kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer // 8, in_layer,
                               kernel_size=1, padding=0)
        self.fc = nn.Linear(1, out_layer // 8)
        self.fc2 = nn.Linear(out_layer // 8, out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_se = nn.functional.adaptive_avg_pool1d(x, 1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)

        x_out = torch.add(x, x_se)
        return x_out


class ReBlock(nn.Module):
    """
    Residual Block.
    """

    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(ReBlock, self).__init__()

        self.cbr1 = ConBrBlock(in_layer, out_layer, kernel_size, 1, dilation)
        self.cbr2 = ConBrBlock(out_layer, out_layer, kernel_size, 1, dilation)
        self.seblock = SeBlock(out_layer, out_layer)

    def forward(self, x):
        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out
