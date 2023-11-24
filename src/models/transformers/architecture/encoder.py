from .encoder_layer import EncoderLayer
from torch import nn
from torch.nn.modules import BatchNorm1d
import torch


class Encoder(nn.Module):
    """
    Encoder of the transformer architecture.
    :param tokenizer: Tokenizer.
    :param pe: Positional encoding.
    :param emb_dim: Embedding dimension.
    :param forward_dim: Dimension of the feed forward layer.
    :param n_layers: Number of encoder layers.
    :param heads: Number of heads in the multihead attention.
    """

    def __init__(self, tokenizer: nn.Module, pe: nn.Module, emb_dim: int = 92, forward_dim: int = 2048,
                 n_layers: int = 6, heads: int = 8, dropout: float = 0.0, attention: dict = {"type": "normal"}) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.tokenizer = tokenizer
        self.pe = pe
        encoders = []
        for _ in range(0, n_layers):
            encoders.append(
                EncoderLayer(emb_dim, heads, forward_dim,
                             dropout=dropout, attention=attention)
            )
        self.encoder_stack = nn.Sequential(*encoders)
        self.norm = BatchNorm1d(emb_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.
        :param src: Input tensor. (bs, l, c)
        :return: Output of encoder tensor. (bs, l_e, e)
        """
        x = self.tokenizer(src)
        x = self.pe(x)
        x = self.encoder_stack(x)
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
