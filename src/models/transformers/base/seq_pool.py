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
    def __init__(self, emb_dim=256):
        super(LSTMPooling, self).__init__()
        self.emb_dim = emb_dim
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim // 2, proj_size=1, bidirectional=False, batch_first=True, dropout=0.1)
    
    def forward(self, all_hidden_states):
        # Forward propagate LSTM
        out, _ = self.lstm(all_hidden_states, None)
        out = out.reshape(-1, out.shape[1])
        return out
    
class NoPooling(nn.Module):
    def __init__(self, emb_dim=256):
        super(NoPooling, self).__init__()
        self.emb_dim = emb_dim
    
    def forward(self, all_hidden_states):
        return all_hidden_states.reshape(all_hidden_states.shape[0], -1)
    
    
