from torch import nn
import torch
from src import data_info

from src.models.architectures.multi_res_bi_GRU import MultiResidualBiGRU

# Base imports
from .pooling import SeqPool, LSTMPooling, NoPooling
from .encoder_config import EncoderConfig
from ...architectures.seg_unet_1d_cnn import ConBrBlock


class UTransformerPool(nn.Module):
    """
    Transformer encoder with patching and a pooling layer.
    :param heads: Number of heads in transformer
    :param emb_dim: Embedding dimension
    :param expansion: Expansion factor in transformer encoder
    :param n_layers: Number of layers in transformer encoder
    :param patch_size: Size of patch in transformer encoder
    :param seq_len: Length of sequence before patching
    :param num_class: Number of classes per step
    :param pooling: Type of pooling to use
    :param pooling_args: Arguments for pooling
    :param tokenizer: Type of tokenizer to use
    :param tokenizer_args: Arguments for tokenizer
    :param pe: Type of positional encoding to use
    :param dropout: Dropout rate to use
    :param attention: Attention type to use
    """

    def __init__(
        self, heads: int = 8, emb_dim: int = 92, expansion: int = 4,
        n_layers: int = 6, seq_len: int = 17280, num_class: int = 2, pooling: str = "none",
        pooling_args: dict = None, tokenizer: str = "patch", tokenizer_args: dict = None,
        pe: str = "fixed", dropout: float = 0.1, attention: dict = {"type": "normal"}
    ) -> None:
        super(UTransformerPool, self).__init__()

        # Selfs
        self.pooling = pooling
        self.seq_len = seq_len
        self.patch_size = tokenizer_args.get("patch_size", 1)
        self.no_features = len(data_info.X_columns)
        self.pooling_args = pooling_args
        self.pooling_args["input_size"] = self.no_features
        self.pooling_args["out_size"] = num_class

        # Ensure emb_dim is divisible by heads
        emb_dim = emb_dim // heads * heads
        if emb_dim < heads:
            emb_dim = heads
        assert emb_dim % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.emb_dim = emb_dim
        self.forward_dim = emb_dim * expansion
        self.encoder = EncoderConfig(tokenizer=tokenizer, tokenizer_args=tokenizer_args, pe=pe, emb_dim=emb_dim,
                                     forward_dim=emb_dim * expansion, n_layers=n_layers, heads=heads, seq_len=seq_len, dropout=dropout, attention=attention)
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
        elif pooling == "lstm":
            self.seq_pool = LSTMPooling(emb_dim=emb_dim)
            self.mlp_head = nn.Linear(
                self.encoder.get_output_size() / emb_dim, num_class)
        elif pooling == "softmax":
            self.seq_pool = SeqPool(emb_dim=emb_dim)
            self.mlp_head = nn.Linear(emb_dim, num_class)

        self.conbr_1 = ConBrBlock(
            emb_dim, emb_dim // 2, 3, 1, 1, padding=1)
        self.conbr_2 = ConBrBlock(
            emb_dim // 2, emb_dim // 4, 3, 1, 1, padding=1)
        self.upsample = nn.Upsample(scale_factor=(
            seq_len // self.l_e), mode='nearest')
        self.outcov = nn.Conv1d(
            emb_dim // 4, num_class, kernel_size=3, stride=1, padding=1)
        self.last_layer = nn.ReLU()
        if pooling == "gru":
            self.gru = MultiResidualBiGRU(**self.pooling_args)
            self.before_res = nn.Linear(
                in_features=emb_dim, out_features=self.no_features)

    def forward(self, x: torch.Tensor, use_activation: bool = True) -> torch.Tensor:
        """
        Forward function for transformer encoder.
        :param x: Input tensor (bs, l, c).
        :return: Output tensor (bs, num_class).
        """
        # Pass x through encoder (bs, l, c) -> (bs, l_e, e)
        curr_x = self.encoder(x)
        assert curr_x.shape[2] == self.e

        # Perform gru (bs, l_e, e) -> (bs, l, num_class)
        if self.pooling == "gru":
            curr_x = self.upsample(curr_x.permute(0, 2, 1))
            curr_x = self.before_res(curr_x.permute(0, 2, 1))
            x += curr_x
            x, _ = self.gru(x, use_activation=use_activation)
            return x

        # Perform conbr_1 (bs, l_e, e) -> (bs, e // 2, l_e)
        x = self.conbr_1(curr_x.permute(0, 2, 1))
        assert x.shape[2] == self.l_e

        # Upsample (bs, e_pool, l_e) -> (bs, e_pool, l)
        x = self.upsample(x)
        assert x.shape[2] == self.seq_len

        # Perform conbr_2 (bs, e_pool, l) -> (bs, e_pool // 2, l)
        x = self.conbr_2(x)
        # Perform outcov (bs, e_pool // 2, l) -> (bs, num_class, l)
        x = self.outcov(x)
        if not use_activation:
            return x.permute(0, 2, 1)
        # Last layer (bs, num_class, l) -> (bs, l, num_class)
        x = self.last_layer(x.permute(0, 2, 1))

        return x
