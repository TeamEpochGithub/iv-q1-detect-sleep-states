import torch

from torch import nn
from .MSA import MSA
from .MLP import MLP


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block"""

    def __init__(
        self, n_h: int, emb_dim: int, feat_dim: int,
        dropout: float = 0, attention_dropout: float = 0
    ):
        super().__init__()
        self.msa = MSA(heads=n_h, emb_dim=emb_dim, dropout=dropout,
                       attention_dropout=attention_dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = MLP(emb_dim, feat_dim, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor):
        identity = x
        x = self.msa(x)
        x += identity
        x = self.norm1(x)
        identity = x
        x = self.ffn(x)
        x += identity
        x = self.norm2(x)
        return x
