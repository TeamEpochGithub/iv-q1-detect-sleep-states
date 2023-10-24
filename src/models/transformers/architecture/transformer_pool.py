from torch import nn, Tensor

# Base imports
from .pooling import SeqPool, LSTMPooling, NoPooling
from .encoder_config import EncoderConfig


class TransformerPool(nn.Module):
    """
    Transformer encoder with patching.

    Args:
        heads: number of heads for transformer encoder
        emb_dim: embedding dimension
        feat_dim: forward dimension for transformer encoder
        dropout: dropout for transformer encoder
        layers: number of transformer encoder layers
        patch_size: patch size for patching
        channels: number of channels in input
        seq_len: sequence length of input
        num_class: number of classes for regression
    """

    def __init__(
        self, heads: int = 8, emb_dim: int = 92, forward_dim: int = 2048,
        n_layers: int = 6, patch_size: int = 36,
        seq_len: int = 17280, num_class: int = 2, pooling: str = "none", tokenizer: str = "patch", tokenizer_args: dict = {},
        pe: str = "fixed", dropout: float = 0.1
    ) -> None:
        super(TransformerPool, self).__init__()
        self.encoder = EncoderConfig(
            tokenizer=tokenizer, tokenizer_args=tokenizer_args, pe=pe, emb_dim=emb_dim, forward_dim=forward_dim, n_layers=n_layers, heads=heads)
        if pooling == "none":
            self.seq_pool = NoPooling(emb_dim=emb_dim)
            self.mlp_head = nn.Linear(
                (seq_len // patch_size) * emb_dim, num_class)
        if pooling == "lstm":
            self.seq_pool = LSTMPooling(emb_dim=emb_dim)
            self.mlp_head = nn.Linear((seq_len // patch_size), num_class)
        elif pooling == "softmax":
            self.seq_pool = SeqPool(emb_dim=emb_dim)
            self.mlp_head = nn.Linear(emb_dim, num_class)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, seq_length, feat_dim)

        Returns:
            (batch_size, num_classes)
        """
        # Pass x through encoder
        x = self.encoder(x)

        # Perform sequential pooling
        x = self.seq_pool(x)

        # MLP head used to get logits
        x = self.mlp_head(x)

        return x
