from typing import Callable
import torch
from torch import nn
from ...architectures.seg_unet_1d_cnn import conbr_block, re_block


class PatchTokenizer(nn.Module):
    """
    Patch tokenizer for transformer encoder.
    :param channels: Number of channels in input.
    :param emb_dim: Embedding dimension.
    :param patch_size: Patch size.
    """

    def __init__(self, channels: int = 2, emb_dim: int = 192, patch_size: int = 36) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.linear_projection = nn.Linear(
            self.patch_size*channels, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for patch tokenizer.
        :param x: Input tensor (bs, l, c).
        :return: Output tensor (bs, l_c, e).
        """
        x = x.permute(0, 2, 1)
        bs, c, len = x.shape  # (bs, c, l)
        torch._assert(
            len % self.patch_size == 0,
            "Sequence not disivible by patch_size"
        )

        # Below 2 lines creates patches using view/reshape and permutations
        # (bs, c, no_of_patches, patch_l) -> (bs, no_of_patches, c, patch_l)
        x = x.view(bs, c, len // self.patch_size,
                   self.patch_size).permute(0, 2, 1, 3)
        # (bs, no_of_patches, c*seq_len)
        x = x.reshape(bs, len // self.patch_size, c*self.patch_size)

        # linear projection to embedding dimension
        x = self.linear_projection(x)
        return x


class SimpleTokenizer(nn.Module):
    """
    Simple convolutional tokenizer for transformer encoder.
    :param channels: Number of channels in input.
    :param emb_dim: Embedding dimension.
    :param hidden_layers: Hidden layers for convolutional tokenizer.
    :param kernel_size: Kernel size for convolutional tokenizer.
    :param depth: Depth of convolutional tokenizer.
    """

    def __init__(self, channels: int = 3, emb_dim: int = 256, hidden_layers: int = 64, kernel_size: int = 7, depth: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=hidden_layers,
                               kernel_size=kernel_size, stride=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool1d(
            kernel_size=kernel_size, stride=3, padding=0)

        self.conv2 = nn.Conv1d(in_channels=hidden_layers, out_channels=emb_dim,
                               kernel_size=kernel_size, stride=3, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.max_pool2 = nn.MaxPool1d(
            kernel_size=kernel_size, stride=3, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for simple tokenizer.
        :param x: Input tensor (bs, l, c).
        :return: Output tensor (bs, l_c, e).
        """
        x = x.permute(0, 2, 1)  # (bs, c, l)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.max_pool2(x)
        x = x.permute(0, 2, 1)  # (bs, l_c, e)
        return x


class ConvTokenizer(nn.Module):
    """
    Convolutional tokenizer for transformer encoder based on unet encoder.
    :param channels: Number of channels in input.
    :param emb_dim: Embedding dimension.
    :param hidden_layers: Hidden layers for convolutional tokenizer.
    :param kernel_size: Kernel size for convolutional tokenizer.
    :param depth: Depth of convolutional tokenizer.
    """

    def __init__(self, channels: int = 3, emb_dim: int = 256, hidden_layers: int = 64, kernel_size: int = 3, depth: int = 2) -> None:
        super().__init__()
        self.in_channels = channels
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

    def down_layer(self, input_layer: int = 3, out_layer: int = 256, kernel: int = 3, stride: int = 1, depth: int = 2) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Down layer for unet encoder.
        :param input_layer: Number of input channels.
        :param out_layer: Number of output channels.
        :param kernel: Kernel size for convolutional tokenizer.
        :param stride: Stride for convolutional tokenizer.
        :param depth: Depth of convolutional tokenizer.
        """
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for _ in range(depth):
            block.append(re_block(out_layer, out_layer, kernel, 1))
        return nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for convolutional tokenizer.
        :param x: Input tensor (bs, l, c).
        :return: Output tensor (bs, l_c, e).
        """
        x = x.permute(0, 2, 1)  # (bs, c, l)
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)

        # Encoder
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)

        x = torch.cat([out_1, pool_x1], 1)
        out_2 = self.layer3(x)

        x = torch.cat([out_2, pool_x2], 1)
        out = self.layer4(x)
        out = out.permute(0, 2, 1)  # (bs, l_c, e)
        return out
