from torch import nn
import torch


class SimpleGRU(nn.Module):
    def __init__(self, config, input_size):
        # will read the model config for the hyperparameters
        # except for the number of features
        super().__init__()
        self.config = config
        self.input_size = input_size
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.bidirectional = config['bidirectional']
        self.batch_first = config['batch_first']
        self.GRU = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          dropout=self.dropout, bidirectional=self.bidirectional, batch_first=self.batch_first)
        if self.bidirectional:
            self.pooling = nn.Conv1d(in_channels=2*self.hidden_size, out_channels=3, kernel_size=1, stride=1)
        else:
            self.pooling = nn.Conv1d(in_channels=self.hidden_size, out_channels=3, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.bidirectional:
            h0 = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).to(x._device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x._device)
        x1, _ = self.GRU(x, h0)
        # apply pooling to return 1 label per timestamp
        x2 = self.pooling(x1.permute(0, 2, 1))
        y = self.softmax(x2)
        return y
