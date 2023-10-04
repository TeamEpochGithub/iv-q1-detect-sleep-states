from torch import nn


class SegSimple1DCNN(nn.Module):
    """
    This contains a baseline 1D CNN architecture for segmenting the data. It takes currently a day as input, outputs 0 / 1 / 2 for every timestep.
    """

    def __init__(self, window_length, in_channels, config):
        super(SegSimple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=4)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, padding=1)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool1d(kernel_size=4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4256, window_length)
        self.relu6 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu6(x)
        return x
