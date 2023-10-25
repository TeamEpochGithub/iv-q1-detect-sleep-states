from torch.nn.modules import (
    MultiheadAttention,
    Dropout,
    BatchNorm1d
)
from torch import nn
from .feed_forward import FeedForward
import torch


class DecoderLayer(nn.Module):
    """
    Decoder layer of the transformer architecture.
    :param emb_dim: Embedding dimension.
    :param heads: Number of heads in the multihead attention.
    :param forward_dim: Dimension of the feed forward layer.
    :param dropout: Dropout rate.
    """

    def __init__(self, emb_dim: int = 92, heads: int = 6, forward_dim: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_1 = BatchNorm1d(emb_dim)
        self.norm_2 = BatchNorm1d(emb_dim)
        self.norm_3 = BatchNorm1d(emb_dim)

        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)
        self.dropout_3 = Dropout(dropout)

        self.attn_1 = MultiheadAttention(
            num_heads=heads, embed_dim=emb_dim, batch_first=True)
        self.attn_2 = MultiheadAttention(
            num_heads=heads, embed_dim=emb_dim, batch_first=True)
        self.ff = FeedForward(emb_dim=emb_dim, forward_dim=forward_dim)

    def forward(self, x: torch.Tensor, e_outputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder layer.
        :param x: Input tensor.
        :param e_outputs: Output of the encoder layer.
        :return: Output of layer tensor.
        """
        x = x.permute(0, 2, 1)
        x2 = self.norm_1(x)
        x = x.permute(0, 2, 1)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2)[0])
        x = x.permute(0, 2, 1)
        x2 = self.norm_2(x)
        x = x.permute(0, 2, 1)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs)[0])
        x = x.permute(0, 2, 1)
        x2 = self.norm_3(x)
        x2 = x2.permute(0, 2, 1)
        x = x.permute(0, 2, 1) + self.dropout_3(self.ff(x2))
        return x
