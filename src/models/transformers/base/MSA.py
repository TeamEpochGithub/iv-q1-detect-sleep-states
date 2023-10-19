import math
import torch
from torch import nn
import torch.nn.functional as F


class MSA(nn.Module):
    """Multi-head Self Attention Block"""

    def __init__(
        self, heads: int, emb_dim: int,
        dropout: float = 0., attention_dropout: float = 0.
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_h = heads
        self.head_dim = self.emb_dim // self.n_h
        self.q = nn.Linear(self.emb_dim, self.emb_dim)
        self.k = nn.Linear(self.emb_dim, self.emb_dim)
        self.v = nn.Linear(self.emb_dim, self.emb_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.linear_projection = nn.Linear(self.emb_dim, self.emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # (bs,     s_l,      e_d)
        batch_s, seq_len, emb_dim = x.shape
        # (bs, s_l, e_d) -> (bs, s_l, n_h, h_d) -> (bs, n_h, s_l, h_d)
        x_q = self.q(x).view(
            batch_s, seq_len, self.n_h, self.head_dim).transpose(1, 2)
        x_k = self.k(x).view(
            batch_s, seq_len, self.n_h, self.head_dim).transpose(1, 2)
        x_v = self.v(x).view(
            batch_s, seq_len, self.n_h, self.head_dim).transpose(1, 2)
        # @ operator is the convention for matrix multiplication, throughout python
        # q @ k.T -> (bs, n_h, s_l, h_d) @ (bs, n_h, h_d, s_l) -> (bs, n_h, s_l, s_l)
        # Softmax((q @ k.T)/root(h_d)) @ v
        #   -> (bs, n_h, s_l, s_l) @ (bs, n_h, s_l, h_d) -> (bs, n_h, s_l, h_d)
        attention = (x_q @ x_k.transpose(-2, -1)) / math.sqrt(x_q.size(-1))
        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        # (bs, n_h, s_l, h_d) -> (bs, s_l, n_h, h_d) -> (bs, s_l, e_d)
        x = (attention @ x_v).transpose(1, 2).reshape(batch_s, seq_len, emb_dim)
        x = self.linear_projection(x)
        x = self.dropout(x)
        return x
