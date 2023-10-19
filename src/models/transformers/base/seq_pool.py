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
