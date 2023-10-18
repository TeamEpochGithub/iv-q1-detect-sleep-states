import torch
from torch import nn


class ConBrBlock(nn.Module):
    """
    Convolution + Batch Normalization + ReLU Block.
    """

    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(ConBrBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=3, bias=True)
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

        self.conv1 = nn.Conv1d(in_layer, out_layer // 8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer // 8, in_layer, kernel_size=1, padding=0)
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


class SegUnet1D(nn.Module):
    """
    SegUnetId model. Contains the architecture of the SegUnetId model.
    """

    def __init__(self, in_channels: int, out_channels: int, config: dict):
        super(SegUnet1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_layers = config["hidden_layers"]
        self.kernel_size = config["kernel_size"]
        self.depth = config["depth"]

        self.AvgPool1D1 = nn.AvgPool1d(kernel_size=5, stride=4, padding=1)
        self.AvgPool1D2 = nn.AvgPool1d(kernel_size=5, stride=16, padding=1)
        self.AvgPool1D3 = nn.AvgPool1d(kernel_size=5, stride=64, padding=1)

        self.layer1 = self.down_layer(self.in_channels, self.hidden_layers, self.kernel_size, 1, 2)
        self.layer2 = self.down_layer(self.hidden_layers, int(self.hidden_layers * 2), self.kernel_size, 4, 2)
        self.layer3 = self.down_layer(int(self.hidden_layers * 2) + int(self.in_channels), int(self.hidden_layers * 3), self.kernel_size, 4, 2)
        self.layer4 = self.down_layer(int(self.hidden_layers * 3) + int(self.in_channels), int(self.hidden_layers * 4), self.kernel_size, 4, 2)
        self.layer5 = self.down_layer(int(self.hidden_layers * 4) + int(self.in_channels), int(self.hidden_layers * 5), self.kernel_size, 4, 2)

        self.cbr_up1 = ConBrBlock(int(self.hidden_layers * 7), int(self.hidden_layers * 3), self.kernel_size, 1, 1)
        self.cbr_up2 = ConBrBlock(int(self.hidden_layers * 5), int(self.hidden_layers * 2), self.kernel_size, 1, 1)
        self.cbr_up3 = ConBrBlock(int(self.hidden_layers * 3), self.hidden_layers, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.outcov = nn.Conv1d(self.hidden_layers, self.out_channels, kernel_size=self.kernel_size, stride=1, padding=3)

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(ConBrBlock(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(ReBlock(out_layer, out_layer, kernel, 1))
        return nn.Sequential(*block)

    def forward(self, x):
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)

        # Encoder
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)

        x = torch.cat([out_1, pool_x1], 1)
        out_2 = self.layer3(x)

        x = torch.cat([out_2, pool_x2], 1)
        x = self.layer4(x)

        # Decoder
        up = self.upsample1(x)
        up = torch.cat([up, out_2], 1)
        up = self.cbr_up1(up)

        up = self.upsample(up)
        up = torch.cat([up, out_1], 1)
        up = self.cbr_up2(up)

        up = self.upsample(up)
        up = torch.cat([up, out_0], 1)
        up = self.cbr_up3(up)

        out = self.outcov(up)
        out = self.softmax(out)

        # out = nn.functional.softmax(out,dim=2)

        return out
