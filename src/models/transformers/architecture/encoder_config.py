import torch.nn as nn
from .encoder import Encoder
from .tokenizer import PatchTokenizer, ConvTokenizer, SimpleTokenizer
from .positional_encoding import FixedPositionalEncoding, OtherPositionalEncoding
import torch
from src.logger.logger import logger


class EncoderConfig(nn.Module):
    """
    Class to create transformer encoder from config.
    :param tokenizer: Tokenizer.
    :param tokenizer_args: Tokenizer arguments.
    :param pe: Positional encoding.
    :param emb_dim: Embedding dimension.
    :param forward_dim: Dimension of the feed forward layer.
    :param n_layers: Number of encoder layers.
    :param heads: Number of heads in the multihead attention.
    """

    def __init__(self, tokenizer: str = "patch", tokenizer_args: dict = {}, pe: str = "fixed",
                 emb_dim: int = 192, forward_dim: int = 2048, n_layers: int = 6, heads: int = 8, 
                 seq_len: int = 17280, dropout: float = 0.0, attention: dict = {"type": "normal"}) -> None:
        super().__init__()

        if tokenizer == "patch":
            assert seq_len % tokenizer_args["patch_size"] == 0, "Sequence length must be divisible by patch size"
        self.tokenizer = get_tokenizer(tokenizer, emb_dim, tokenizer_args)
        with torch.no_grad():
            x = torch.randn([1, seq_len, tokenizer_args["channels"]])
            out = self.tokenizer(x)
            _, l_c, _ = out.shape
            logger.debug("Transformer attention size" + str(l_c))

        self.pe = get_positional_encoding(pe, emb_dim=emb_dim, max_len=l_c)
        self.output_size = emb_dim * l_c
        self.model = Encoder(self.tokenizer, self.pe,
                             emb_dim, forward_dim, n_layers, heads, dropout=dropout, attention=attention)

    def get_output_size(self) -> int:
        """
        Get output size of encoder.
        :return: Output size of encoder.
        """
        return self.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Call forward pass of model.
        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.model(x)


def get_tokenizer(tokenizer: str = "patch", emb_dim: int = 92, tokenizer_args: dict = {}) -> nn.Module:
    """
    Get tokenizer from config.
    :param tokenizer: Tokenizer.
    :param emb_dim: Embedding dimension.
    :param tokenizer_args: Tokenizer arguments.
    :return: Tokenizer.
    """
    if tokenizer == "patch":
        return PatchTokenizer(emb_dim=emb_dim, **tokenizer_args)
    elif tokenizer == "unet_encoder":
        return ConvTokenizer(emb_dim=emb_dim, **tokenizer_args)
    elif tokenizer == "simple_conv":
        return SimpleTokenizer(emb_dim=emb_dim, **tokenizer_args)


def get_positional_encoding(pe: str = "fixed", emb_dim: int = 92, max_len: int = 480) -> nn.Module:
    """
    Get positional encoding from config.
    :param pe: Positional encoding.
    :param emb_dim: Embedding dimension.
    :return: Positional encoding.
    """
    if pe == "fixed":
        return FixedPositionalEncoding(d_model=emb_dim, max_len=max_len)
    elif pe == "other":
        return OtherPositionalEncoding(emb_dim=emb_dim, max_len=max_len)
