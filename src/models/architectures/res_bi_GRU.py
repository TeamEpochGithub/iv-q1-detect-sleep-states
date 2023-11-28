from torch import nn


class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, internal_layers=1, bidir=True, dropout=0):
        super(ResidualBiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.internal_layers = internal_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            internal_layers,
            batch_first=True,
            bidirectional=bidir
        )

        # These are added for testing, LSTM performs a bit worse and RNN is complete garbage.
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            internal_layers,
            batch_first=True,
            bidirectional=bidir
        )

        self.rnn = nn.RNN(
            hidden_size,
            hidden_size,
            internal_layers,
            batch_first=True,
            bidirectional=bidir
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(
            hidden_size * dir_factor, hidden_size * dir_factor * 2
        )
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.gelu(res)
        res = self.dropout(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.gelu(res)
        res = self.dropout(res)

        # skip connection
        res = res + x

        return res, new_h
