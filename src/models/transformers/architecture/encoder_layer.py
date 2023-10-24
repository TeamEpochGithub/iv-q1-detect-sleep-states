from torch.nn.modules import (
    MultiheadAttention,
    Dropout,
    BatchNorm1d
)
from torch import nn
from .feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, emb_dim: int = 92, heads: int = 6, forward_dim: int = 2048, dropout = 0.1):
        super().__init__()
        self.norm_1 = BatchNorm1d(emb_dim)
        self.norm_2 = BatchNorm1d(emb_dim)
        self.attn = MultiheadAttention(heads, emb_dim)
        self.ff = FeedForward(emb_dim, forward_dim)
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x