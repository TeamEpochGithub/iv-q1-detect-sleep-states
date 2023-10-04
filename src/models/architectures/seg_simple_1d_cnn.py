from torch import nn


class SegSimple1DCNN(nn.Module):
    """
    This contains a baseline 1D CNN architecture for segmenting the data. It takes currently a day as input, outputs 0 / 1 / 2 for every timestep.
    """
    def __init__(self, window_length, in_channels, config):
        super(SegSimple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=5, kernel_size=5, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=5, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(43180, window_length)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x