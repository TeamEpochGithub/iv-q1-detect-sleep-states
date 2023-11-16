import math
from functools import reduce

import torch
from torch import nn

from src.logger.logger import logger
from src.models.model import ModelException


class ConBrBlock(nn.Module):
    """
    Convolution + Batch Normalization + ReLU Block.
    """

    def __init__(self, in_layer: int, out_layer: int, kernel_size: int | tuple[int], stride: int | tuple[int], dilation: int | tuple[int], padding: str | int | tuple[int] = 3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
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

        self.conv1 = nn.Conv1d(in_layer, out_layer // 8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer // 8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1, out_layer // 8)
        self.fc2 = nn.Linear(out_layer // 8, out_layer)
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

    def __init__(self, in_layer: int, out_layer: int, kernel_size: int | tuple[int], dilation: int | tuple[int], dropout: float):
        super().__init__()

        self.cbr1 = ConBrBlock(in_layer, out_layer, kernel_size, 1, dilation)
        self.cbr2 = ConBrBlock(out_layer, out_layer, kernel_size, 1, dilation)
        self.seblock = SeBlock(out_layer, out_layer)
        self.dropout = nn.Dropout(dropout)

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

    def __init__(self, in_channels: int, window_size: int, out_channels: int, model_type: str, downsampling_factor: int,
                 activation: str | None = 'relu', hidden_layers: int = 8, kernel_size: int = 7,
                 depth: int = 2, dropout: float = 0, stride: int = 4, padding: int = 1, n_layers: int = 4) -> None:
        super().__init__()

        # Set model dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size
        self.model_type = model_type
        self.downsampling_factor = downsampling_factor  # TODO Keep?

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
        if n_layers < 2:
            logger.critical('Unet must have at least 2 layers')
            raise ModelException('Unet must have at least 2 layers')
        self.n_layers = n_layers

        # If we downsample by 12 (17280/12), we need to have a stride of 2.
        # if self.downsampling_factor == 12:
        #     self.stride = 2
        #     self.padding = 2

        # if self.window_size < 17280:
        #     self.stride = 2
        #     self.padding = 2

        self.verify_network_params()

        self.avg_pools: nn.ModuleList[nn.AvgPool1d] = nn.ModuleList([nn.AvgPool1d(kernel_size=5, stride=self.stride ** i, padding=self.padding) for i in range(1, self.n_layers - 1)])
        self.layers: nn.ModuleList[nn.Sequential] = nn.ModuleList([self.down_layer(int(self.hidden_layers * i) + int(self.in_channels), int(self.hidden_layers * (i + 1)), self.kernel_size, self.stride, self.depth, self.dropout) for i in range(self.n_layers)])
        self.layers[1] = self.down_layer(self.hidden_layers, int(self.hidden_layers * 2), self.kernel_size, self.stride, self.depth, self.dropout)  # Hardcode layer 1 here
        self.cbrs: nn.ModuleList[ConBrBlock] = nn.ModuleList([ConBrBlock(int(self.hidden_layers * ((i * 2) + 1)), int(self.hidden_layers * (i + 1)), self.kernel_size, 1, 1) for i in range(1, self.n_layers)])

        # self.AvgPool1D1 = nn.AvgPool1d(kernel_size=5, stride=self.stride ** 1, padding=self.padding)
        # self.AvgPool1D2 = nn.AvgPool1d(kernel_size=5, stride=self.stride ** 2, padding=self.padding)
        # # self.AvgPool1D3 = nn.AvgPool1d(kernel_size=5, stride=self.stride ** 3, padding=self.padding)
        #
        # self.layer1 = self.down_layer(self.in_channels, self.hidden_layers, self.kernel_size, 1, self.depth, self.dropout)
        # self.layer2 = self.down_layer(self.hidden_layers, int(self.hidden_layers * 2), self.kernel_size, self.stride, self.depth, self.dropout)
        # self.layer3 = self.down_layer(int(self.hidden_layers * 2) + int(self.in_channels), int(self.hidden_layers * 3), self.kernel_size, self.stride, self.depth, self.dropout)
        # self.layer4 = self.down_layer(int(self.hidden_layers * 3) + int(self.in_channels), int(self.hidden_layers * 4), self.kernel_size, self.stride, self.depth, self.dropout)
        # # self.layer5 = self.down_layer(int(self.hidden_layers * 4) + int(self.in_channels), int(self.hidden_layers * 5), self.kernel_size, self.stride, self.depth, self.dropout)  # FIXME: This is not used?
        #
        # self.cbr_up1 = ConBrBlock(int(self.hidden_layers * 7), int(self.hidden_layers * 3), self.kernel_size, 1, 1)
        # self.cbr_up2 = ConBrBlock(int(self.hidden_layers * 5), int(self.hidden_layers * 2), self.kernel_size, 1, 1)
        # self.cbr_up3 = ConBrBlock(int(self.hidden_layers * 3), self.hidden_layers, self.kernel_size, 1, 1)
        # self.upsample = nn.Upsample(scale_factor=self.stride, mode='nearest')
        # self.upsample1 = nn.Upsample(scale_factor=self.stride, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        # self.relu = nn.ReLU()  # FIXME: This is not used
        self.outcov = nn.Conv1d(self.hidden_layers, self.out_channels, kernel_size=self.kernel_size, stride=1, padding=3)

    def verify_network_params(self) -> None:
        """Verify that the given parameters are valid.

        TODO Doesn't work
        """
        # Verify that the window can be divides by the stride multiple times
        if (self.window_size / (self.stride ** self.n_layers)).is_integer():
            logger.warning("The window size must be divisible by the stride for the number of layers. Model may crash!")

        if (self.window_size / (self.stride ** 1) == math.floor(self.window_size + 2 * self.padding - self.kernel_size) / self.stride + 1):
            logger.warning("The first down and pooling layers must have the same size. Model may crash!")

    @staticmethod
    def down_layer(input_layer: int, out_layer: int, kernel: int | tuple[int], stride: int | tuple[int], depth: int, dropout: float) -> nn.Sequential:
        block: list[nn.Module] = [ConBrBlock(input_layer, out_layer, kernel, stride, 1)]
        for _ in range(depth):
            block.append(ReBlock(out_layer, out_layer, kernel, 1, dropout))

        return nn.Sequential(*block)

    def forward(self, x: torch.Tensor, use_activation: bool = True) -> torch.Tensor:
        prev_x = x
        # pool_x1 = self.AvgPool1D1(x)
        # pool_x2 = self.AvgPool1D2(x)
        pools = [pool(x) for pool in self.avg_pools]

        # Encoder
        encoder_out_prev: torch.Tensor = self.layers[0](x)
        encoder_outputs: list[torch.Tensor] = [encoder_out_prev, self.layers[1](encoder_out_prev)]

        for layer, pool in zip(self.layers[2:], pools):
            x = torch.cat([encoder_outputs[-1], pool], 1)
            encoder_outputs.append(layer(x))

        # encoder_outputs: list = reduce(lambda out, layer_pool: layer_pool[0](torch.cat([out, layer_pool[1]], 1)), zip(self.layers[1:-1], pools), out_0)
        # encoder_outputs.append(self.layers[-1](encoder_out_prev))


        # x = torch.cat([out_1, pool_x1], 1)
        # out_2 = self.layer3(x)
        #
        # x = torch.cat([out_2, pool_x2], 1)
        # x = self.layer4(x)

        # Decoder
        ups: list = []
        for i, cbr in reversed(list(enumerate(self.cbrs))):
            ups.append(cbr(torch.cat([self.upsample(encoder_outputs[i + 1]), encoder_outputs[i]], 1)))


        # up = self.upsample(x)
        # up = torch.cat([up, out_2], 1)
        # up = self.cbr_up1(up)
        #
        # up = self.upsample(up)
        # up = torch.cat([up, out_1], 1)
        # up = self.cbr_up2(up)
        #
        # up = self.upsample(up)
        # up = torch.cat([up, out_0], 1)
        # up = self.cbr_up3(up)

        out = self.outcov(ups[-1])

        if self.model_type == "state-segmentation":
            out = self.softmax(out)
        elif self.model_type == "event-segmentation":
            if use_activation:
                out = self.activation(out)
        return out
