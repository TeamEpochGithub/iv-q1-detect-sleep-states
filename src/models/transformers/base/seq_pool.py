import torch
from einops import rearrange
from torch import nn


class SeqPool(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.dense = nn.Linear(emb_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, seq_len, emb_dim = x.shape
        identity = x
        x = self.dense(x)
        x = rearrange(
            x, 'bs seq_len 1 -> bs 1 seq_len', seq_len=seq_len
        )
        x = self.softmax(x)
        x = x @ identity
        x = rearrange(
            x, 'bs 1 e_d -> bs e_d', e_d=emb_dim
        )
        return x
    
class LSTMPooling(nn.Module):
    def __init__(self, emb_dim=256, hidden_size=64):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = 2
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, all_hidden_states):
        ## forward
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out
