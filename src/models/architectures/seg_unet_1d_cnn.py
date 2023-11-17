import math

import torch
from torch import nn

from src.logger.logger import logger
from src.models.model import ModelException


class ConBrBlock(nn.Module):
    """
    Convolution + Batch Normalization + ReLU Block.
    """

    def __init__(self, in_layer: int, out_layer: int, kernel_size: int | tuple[int], stride: int | tuple[int],
                 dilation: int | tuple[int], padding: str | int | tuple[int] = 3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_layer, out_channels=out_layer, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, bias=True)
        self.bn = nn.BatchNorm1d(num_features=out_layer)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)

        return out


class SeBlock(nn.Module):
    """
    Squeeze and Excitation Block.
    """

    def __init__(self, in_layer: int, out_layer: int):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_layer, out_channels=out_layer // 8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=out_layer // 8, out_channels=in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(in_features=1, out_features=out_layer // 8)
        self.fc2 = nn.Linear(in_features=out_layer // 8, out_features=out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, in_layer: int, out_layer: int, kernel_size: int | tuple[int], dilation: int | tuple[int],
                 dropout: float):
        super().__init__()

        self.cbr1 = ConBrBlock(in_layer=in_layer, out_layer=out_layer, kernel_size=kernel_size, stride=1,
                               dilation=dilation)
        self.cbr2 = ConBrBlock(in_layer=out_layer, out_layer=out_layer, kernel_size=kernel_size, stride=1,
                               dilation=dilation)
        self.seblock = SeBlock(in_layer=out_layer, out_layer=out_layer)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_re = self.dropout(x_re)
        x_out = torch.add(x, x_re)
        return x_out


class SegUnet1D(nn.Module):
    """
    SegUnetId model. Contains the architecture of the SegUnetId model used for state and event segmentation.
    """

    def __init__(self, in_channels: int, window_size: int, out_channels: int, model_type: str,
                 activation: str | None = 'relu', hidden_layers: int = 8, kernel_size: int = 7, depth: int = 2,
                 dropout: float = 0, stride: int = 4, padding: int = 1, n_layers: int = 4) -> None:
        super().__init__()

        # Set model dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size
        self.model_type = model_type

        # Set model params
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size  # FIXME Only works with 7
        self.depth = depth
        if activation is None:
            self.activation = nn.Identity
        else:
            self.activation = nn.functional.__dict__[activation]
        self.dropout = dropout
        self.stride = stride
        self.padding = padding
        # self.dilation = dilation

        if n_layers < 2:
            logger.critical('Unet must have at least 2 layers')
            raise ModelException('Unet must have at least 2 layers')
        self.n_layers = n_layers

        self.avg_pools: nn.ModuleList[nn.AvgPool1d] = nn.ModuleList(
            [nn.AvgPool1d(kernel_size=5, stride=self.stride ** i, padding=self.padding) for i in
             range(1, self.n_layers - 1)])

        self.layers: nn.ModuleList[nn.Sequential] = nn.ModuleList([
            self.down_layer(input_layer=self.in_channels, out_layer=self.hidden_layers, kernel=self.kernel_size,
                            stride=1, depth=self.depth, dropout=self.dropout),  # Hardcoded layer 0
            self.down_layer(input_layer=self.hidden_layers, out_layer=int(self.hidden_layers * 2),
                            kernel=self.kernel_size, stride=self.stride, depth=self.depth,
                            dropout=self.dropout),  # Hardcoded layer 1
            *[self.down_layer(input_layer=int(self.hidden_layers * i) + int(self.in_channels),
                              out_layer=int(self.hidden_layers * (i + 1)),
                              kernel=self.kernel_size, stride=self.stride, depth=self.depth, dropout=self.dropout) for i
              in range(2, self.n_layers)]
        ])

        self.cbrs: nn.ModuleList[ConBrBlock] = nn.ModuleList(
            [ConBrBlock(in_layer=int(self.hidden_layers * ((i * 2) + 1)), out_layer=int(self.hidden_layers * i),
                        kernel_size=self.kernel_size, stride=1, dilation=1)
             for i in range(1, self.n_layers)])

        self.upsample = nn.Upsample(scale_factor=self.stride, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.outcov = nn.Conv1d(in_channels=self.hidden_layers, out_channels=self.out_channels,
                                kernel_size=self.kernel_size, stride=1, padding=3)

        self.verify_network_params()

    def verify_network_params(self) -> None:
        """Verify that the given parameters are valid.

        TODO Expand this function to verify all parameters
        TODO Run this in the config checker instead #190
        """

        # Verify that each down layer's output has the same size as the next layer's input
        layer_sizes: list = [self.window_size,
                             *[self.window_size / (self.stride ** i) for i in range(1, self.n_layers)]]

        final_pool_layer_size = math.floor((self.window_size + 2 * self.padding - 5) / (self.stride ** (self.n_layers - 1)) + 1)
        if layer_sizes[-1] != final_pool_layer_size:
            logger.warning(f"The final pooling layer ({final_pool_layer_size}) must have the same size as the last down layer ({layer_sizes[-1]}). Model may crash!")

        for i in range(1, self.n_layers):
            out_layer_size = math.floor(
                ((layer_sizes[i - 1] + 2 * 3 - 1 * (self.kernel_size - 1) - 1) / self.stride) + 1)
            if layer_sizes[i] != out_layer_size:
                logger.warning(f"The down input layer {i} ({layer_sizes[i]}) and it's previous layer ({out_layer_size}) must have the same size. Model may crash!")


    @staticmethod
    def down_layer(input_layer: int, out_layer: int, kernel: int | tuple[int], stride: int | tuple[int], depth: int,
                   dropout: float, dilation: int = 1) -> nn.Sequential:
        block: list[nn.Module] = [
            ConBrBlock(in_layer=input_layer, out_layer=out_layer, kernel_size=kernel, stride=stride,
                       dilation=dilation, padding=3)]
        for _ in range(depth):
            block.append(
                ReBlock(in_layer=out_layer, out_layer=out_layer, kernel_size=kernel, dilation=1, dropout=dropout))

        return nn.Sequential(*block)

    def forward(self, x: torch.Tensor, use_activation: bool = True) -> torch.Tensor:
        pools = [pool(x) for pool in self.avg_pools]

        # Encoder
        encoder_out_prev: torch.Tensor = self.layers[0](x)
        encoder_outputs: list[torch.Tensor] = [encoder_out_prev, self.layers[1](encoder_out_prev)]

        for layer, pool in zip(self.layers[2:], pools):
            x = torch.cat([encoder_outputs[-1], pool], 1)
            encoder_outputs.append(layer(x))

        # Decoder
        decoder_outputs: list = [encoder_outputs[-1]]

        for i, cbr in reversed(list(enumerate(self.cbrs))):
            up = torch.cat([self.upsample(decoder_outputs[-1]), encoder_outputs[i]], 1)
            decoder_outputs.append(cbr(up))

        out = self.outcov(decoder_outputs[-1])

        if self.model_type == "state-segmentation":
            out = self.softmax(out)
        elif self.model_type == "event-segmentation" and use_activation:
            out = self.activation(out)
        return out
