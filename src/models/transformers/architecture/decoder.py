from .decoder_layer import DecoderLayer
from torch import nn
from torch.nn.modules import BatchNorm1d
import torch


class Decoder(nn.Module):
    """
    Decoder of the transformer architecture.
    :param tokenizer: Tokenizer.
    :param pe: Positional encoding.
    :param emb_dim: Embedding dimension.
    :param forward_dim: Dimension of the feed forward layer.
    :param n_layers: Number of decoder layers.
    :param heads: Number of heads in the multihead attention.
    """

    def __init__(self, tokenizer: nn.Module, pe: nn.Module, emb_dim: int = 92, forward_dim: int = 2048, n_layers: int = 6, heads: int = 8) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.tokenizer = tokenizer
        self.pe = pe
        decoders = []
        for _ in range(0, n_layers):
            decoders.append(
                DecoderLayer(emb_dim, heads, forward_dim)
            )
        self.decoder_stack = nn.Sequential(*decoders)
        self.norm = BatchNorm1d(emb_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        :param src: Input tensor. (bs, l, c)
        :return: Output of decoder tensor. (bs, l_e, e)
        """
        x = self.tokenizer(src)  # (bs, l, c) -> (bs, l_e, e)
        x = self.pe(x)  # (bs, l_e, e) -> (bs, l_e, e)
        x = self.decoder_stack(x)  # (bs, l_e, e) -> (bs, l_e, e)
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
