from torch import nn
import torch

# Base imports
from .pooling import SeqPool, LSTMPooling, NoPooling
from .encoder_config import EncoderConfig
from ...architectures.seg_unet_1d_cnn import ConBrBlock


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
        pe: str = "fixed", dropout: float = 0.1, t_type: str = "regression"
    ) -> None:
        super(TransformerPool, self).__init__()
        self.encoder = EncoderConfig(
            tokenizer=tokenizer, tokenizer_args=tokenizer_args, pe=pe, emb_dim=emb_dim, forward_dim=forward_dim, n_layers=n_layers, heads=heads, seq_len=seq_len)
        with torch.no_grad():
            x = torch.randn([1, seq_len, tokenizer_args["channels"]])
            out = self.encoder(x)
            _, l_e, e = out.shape
            self.l_e = l_e
            self.e = e
        if pooling == "none":
            self.seq_pool = NoPooling(emb_dim=emb_dim)
            self.mlp_head = nn.Linear(
                self.encoder.get_output_size(), num_class)
        if pooling == "lstm":
            self.seq_pool = LSTMPooling(emb_dim=emb_dim)
            self.mlp_head = nn.Linear(
                self.encoder.get_output_size() / emb_dim, num_class)
        elif pooling == "softmax":
            self.seq_pool = SeqPool(emb_dim=emb_dim)
            self.mlp_head = nn.Linear(emb_dim, num_class)
        self.t_type = t_type
        # No head is used for segmentation
        if t_type == "state":
            self.untokenize = nn.Linear(self.l_e, seq_len)
            self.mlp_head = nn.Linear(self.e, num_class)
            self.last_layer = nn.Sigmoid()
        elif t_type == "event":
            self.conbr_1 = ConBrBlock(emb_dim, emb_dim // 2, 3, 1, 1, padding=1)
            self.conbr_2 = ConBrBlock(emb_dim // 2, emb_dim // 4, 3, 1, 1, padding=1)
            self.upsample = nn.Upsample(scale_factor=(seq_len // self.l_e), mode='nearest')
            self.outcov = nn.Conv1d(emb_dim // 4, num_class, kernel_size=3, stride=1, padding=1)
            self.last_layer = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for transformer encoder.
        :param x: Input tensor (bs, l, c).
        :return: Output tensor (bs, num_class).
        """
        # Pass x through encoder (bs, l, c) -> (bs, l_e, e)
        x = self.encoder(x)

        if self.t_type == "state":
            # Perform sequential pooling (bs, l_e, e) -> (bs, len, num_class)
            # Untokenize (bs, l_e, e) -> (bs, l, e)
            x = self.untokenize(x.permute(0, 2, 1)).permute(0, 2, 1)
            # MLP head used to get logits (bs, l, e) -> (bs, l, num_class)
            x = self.mlp_head(x)
            # Softmax (bs, l, num_class) -> (bs, l, num_class)
            x = self.last_layer(x)
        elif self.t_type == "event":
            # Perform conbr_1 (bs, l_e, e) -> (bs, e // 2, l_e)
            x = self.conbr_1(x.permute(0, 2, 1))
            # Upsample (bs, e_pool, l_e) -> (bs, e_pool, l)
            x = self.upsample(x)
            # Perform conbr_2 (bs, e_pool, l) -> (bs, e_pool // 2, l)
            x = self.conbr_2(x)
            # Perform outcov (bs, e_pool // 2, l) -> (bs, num_class, l)
            x = self.outcov(x)
            # Last layer (bs, num_class, l) -> (bs, l, num_class)
            x = self.last_layer(x.permute(0, 2, 1))
        else:
            # Perform sequential pooling (bs, l_e, e) -> (bs, e_pool)
            x = self.seq_pool(x)
            # MLP head used to get logits (bs, e_pool) -> (bs, num_class)
            x = self.mlp_head(x)

        return x
