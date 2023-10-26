import torch
from torch import nn
from typing import Type
import math


class FixedPositionalEncoding(nn.Module):
    """
    Positional encoding of input.
    :param d_model: the embed dim.
    :param dropout: the dropout value.
    :param max_len: the max. length of the incoming sequence.
    :param scale_factor: scale factor for the positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 1024,
        scale_factor: float = 1.0,
    ) -> None:
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fixed positional encoding layer.
        :param x: Input tensor.
        :return: Output tensor.
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Positional encoding of input. This is learnable.
    :param d_model: the embed dim.
    :param dropout: the dropout value.
    :param max_len: the max. length of the incoming sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024) -> None:
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(max_len, 1, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs of forward function
        :param x: Input tensor.
        :return: Output tensor.
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class OtherPositionalEncoding(nn.Module):
    """
    Other positional encoding of input. This is learnable.
    :param d_model: the embed dim.
    :param dropout: the dropout value.
    """

    def __init__(self, max_len: int = 480, emb_dim: int = 92) -> None:
        self.pos_emb = nn.Parameter(torch.randn(
            [1, max_len, emb_dim]).normal_(std=0.02))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of other encoding
        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.pos_emb(x) + x


def get_pos_encoder(pos_encoding: str) -> Type[nn.Module]:
    """
    Get positional encoding from config.
    :param pos_encoding: Positional encoding.
    :return: Positional encoding.
    """
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding
    elif pos_encoding == "other":
        return OtherPositionalEncoding

    raise NotImplementedError(
        "pos_encoding should be 'learnable'/'fixed', not '{}'".format(
            pos_encoding)
    )
