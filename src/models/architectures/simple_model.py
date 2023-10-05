from torch import nn


class SimpleModel(nn.Module):
    """
    Pytorch implementation of a really simple baseline model.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, config):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
