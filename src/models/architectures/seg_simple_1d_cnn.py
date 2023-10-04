from torch import nn


class SegSimple1DCNN(nn.Module):
    """
    This contains a baseline 1D CNN architecture for segmenting the data. It takes currently a day as input, outputs 0 / 1 / 2 for every timestep.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=5)
        self.fc1 = nn.Linear(40 * 96, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2)
        x = x.view(-1, 40 * 96)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x