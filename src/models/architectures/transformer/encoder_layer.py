import torch
from torch import nn
from torch.nn.modules import (
    MultiheadAttention,
    Dropout,
    LayerNorm,
)

from src.models.architectures.transformer.attention.bahdanau_attention import BahdanauAttention
from src.models.architectures.transformer.attention.sparse_attention import SparseAttention
from .feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """
    Encoder layer of the transformer architecture.
    :param emb_dim: Embedding dimension.
    :param heads: Number of heads in the multihead attention.
    :param forward_dim: Dimension of the feed forward layer.
    :param dropout: Dropout rate.
    """

    def __init__(self, heads: int = 6, emb_dim: int = 92, expansion: int = 4, dropout: float = 0.1,
                 attention: dict = {"type": "normal"}) -> None:
        super().__init__()
        self.norm_1 = LayerNorm(emb_dim)
        self.norm_2 = LayerNorm(emb_dim)
        self.att_type = attention.get("type", "normal")
        if self.att_type == "normal":
            self.attn = MultiheadAttention(
                num_heads=heads, embed_dim=emb_dim, batch_first=True)
        elif self.att_type == "sparse":
            self.attn = SparseAttention(heads=heads, block_size=attention.get(
                "block_size", 64), attn_mode=attention.get("attn_mode", "e"),
                                        local_attn_ctx=attention.get("local_attn_ctx", 10))
        elif self.att_type == "bahdanau":
            self.attn = BahdanauAttention(
                hidden_size=emb_dim, bidirectional=False)
        self.ff = FeedForward(emb_dim=emb_dim, expansion=expansion)
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None, is_causal: bool = False,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the encoder layer.
        :param x: Input tensor.
        :return: Output of layer tensor.
        """
        x1 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x1, x1, x1)[0])
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
