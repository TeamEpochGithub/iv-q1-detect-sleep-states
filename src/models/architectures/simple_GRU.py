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
            self.fc = nn.Linear(self.hidden_size * 2 * self.num_layers, 1)
        else:
            self.fc = nn.Linear(self.hidden_size * self.num_layers, 1)

    def forward(self, x):
        if self.bidirectional:
            h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        _, x = self.GRU(x, h0)
        # put the batch first
        x = x.permute(1, 0, 2)
        x = torch.flatten(x, start_dim=1)
        y = self.fc(x)
        return y
