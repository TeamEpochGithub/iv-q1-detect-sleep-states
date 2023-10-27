from torch import nn
import torch

# Base imports
from .pooling import SeqPool, LSTMPooling, NoPooling
from .encoder_config import EncoderConfig


class TransformerPool(nn.Module):
    """
    Transformer encoder with patching.
    :param heads: Number of heads in transformer
    :param emb_dim: Embedding dimension
    :param forward_dim: Forward dimension in transformer
    :param n_layers: Number of layers in transformer
    :param patch_size: Size of patch in transformer
    :param seq_len: Length of sequence in transformer
    :param num_class: Number of classes in transformer
    :param pooling: Type of pooling to use
    :param tokenizer: Type of tokenizer to use
    :param tokenizer_args: Arguments for tokenizer
    :param pe: Type of positional encoding to use
    :param dropout: Dropout rate to use
    """

    def __init__(
        self, heads: int = 8, emb_dim: int = 92, forward_dim: int = 2048,
        n_layers: int = 6,
        seq_len: int = 17280, num_class: int = 2, pooling: str = "none", tokenizer: str = "patch", tokenizer_args: dict = {},
        pe: str = "fixed", dropout: float = 0.1, no_head: bool = False
    ) -> None:
        super(TransformerPool, self).__init__()
        self.encoder = EncoderConfig(
            tokenizer=tokenizer, tokenizer_args=tokenizer_args, pe=pe, emb_dim=emb_dim, forward_dim=forward_dim, n_layers=n_layers, heads=heads, seq_len=seq_len)
        self.no_head = no_head
        if pooling == "none":
            self.seq_pool = NoPooling(emb_dim=emb_dim)
            self.mlp_head = nn.Linear(
                self.encoder.get_output_size(), num_class)
        if pooling == "lstm":
            self.seq_pool = LSTMPooling(emb_dim=emb_dim)
            self.mlp_head = nn.Linear(self.encoder.get_output_size() / emb_dim, num_class)
        elif pooling == "softmax":
            self.seq_pool = SeqPool(emb_dim=emb_dim)
            self.mlp_head = nn.Linear(emb_dim, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for transformer encoder.
        :param x: Input tensor (bs, l, c).
        :return: Output tensor (bs, num_class).
        """
        # Pass x through encoder (bs, l, c) -> (bs, l_e, e)
        x = self.encoder(x)

        # Perform sequential pooling (bs, l_e, e) -> (bs, e_pool)
        x = self.seq_pool(x)

        # MLP head used to get logits (bs, e_pool) -> (bs, num_class)
        if self.no_head:
            return x
        x = self.mlp_head(x)

        return x
