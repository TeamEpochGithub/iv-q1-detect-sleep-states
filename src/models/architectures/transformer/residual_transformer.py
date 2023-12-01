from torch import nn
import torch

from src.models.architectures.multi_res_bi_GRU import MultiResidualBiGRU
from src.models.architectures.transformer.encoder_layer import EncoderLayer
from src.models.architectures.transformer.patch_tokenizer import PatchTokenizer
from src.models.architectures.transformer.positional_encoding import FixedPositionalEncoding, LearnablePositionalEncoding


class ResidualTransformer(nn.Module):
    """
    Transformer encoder with patching and a possible residual model connection.
    :param heads: Number of heads in transformer
    :param emb_dim: Embedding dimension
    :param expansion: Expansion factor in transformer encoder
    :param n_layers: Number of layers in transformer encoder
    :param seq_len: Length of sequence before patching
    :param input_size: Number of features in input
    :param num_class: Number of classes per step
    :param residual_model: Model to use with residual connection
    :param patch_size: Size of patch to use
    :param pe: Type of positional encoding to use
    :param dropout: Dropout rate to use
    :param attention: Attention type to use
    """

    def __init__(
        self, heads: int = 8, emb_dim: int = 92, expansion: int = 4,
        n_layers: int = 6, seq_len: int = 17280, input_size: int = -1, num_class: int = 2, residual_model: dict = None, patch_size: int = 1,
        pe: str = "fixed", dropout: float = 0.1, attention: dict = {"type": "normal"}
    ) -> None:
        super(ResidualTransformer, self).__init__()

        # Init args
        self.patch_size = patch_size
        self.num_class = num_class
        self.no_features = input_size
        self.residual_model = residual_model
        l_e = seq_len // patch_size

        # Ensure emb_dim is divisible by heads
        emb_dim = emb_dim // heads * heads
        if emb_dim < heads:
            emb_dim = heads
        assert emb_dim % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.emb_dim = emb_dim

        # Patching
        self.patching = PatchTokenizer(
            channels=input_size, emb_dim=emb_dim, patch_size=patch_size)

        # Positional encoding
        if pe == "fixed":
            self.pos_encoding = FixedPositionalEncoding(
                emb_dim=emb_dim, dropout=dropout, max_len=l_e, scale_factor=1.0)
        elif pe == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(
                emb_dim=emb_dim, dropout=dropout, max_len=l_e)

        # Create transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=EncoderLayer(
                heads=heads, emb_dim=emb_dim, expansion=expansion, dropout=dropout, attention=attention),
            num_layers=n_layers,
            norm=nn.LayerNorm(emb_dim)
        )

        # Upsample to get back to original sequence length
        self.upsample = nn.Upsample(scale_factor=(
            seq_len // l_e), mode='nearest')

        # Layer to get number of classes
        self.linear_to_classes = nn.Linear(
            in_features=emb_dim, out_features=num_class)
        self.activation = nn.ReLU()

        # Residual model connection
        if residual_model:
            self.residual_model = MultiResidualBiGRU(**self.residual_model)
            self.linear_to_features = nn.Linear(
                in_features=emb_dim, out_features=input_size)

    def forward(self, x: torch.Tensor, use_activation: bool = True) -> torch.Tensor:
        """
        Forward function for transformer encoder.
        :param x: Input tensor (bs, l, c).
        :param use_activation: Whether to use activation function.
        :return: Output tensor (bs, num_class).
        """

        # Patching (bs, l, c) -> (bs, l_e, e)
        x1 = self.patching(x)

        # Positional encoding (bs, l_e, e) -> (bs, l_e, e)
        x1 = self.pos_encoding(x1)

        # Pass x through transformer encoder (bs, l_e, e) -> (bs, l_e, e)
        x1 = self.transformer_encoder(x1)

        # Upsample (bs, l_e, e) -> (bs, l, e)
        x1 = self.upsample(x1.permute(0, 2, 1)).permute(0, 2, 1)

        if self.residual_model:
            # Secondary model to use with residual connection
            x1 = self.linear_to_features(
                x1)  # (bs, l, e) -> (bs, l, c)
            x += x1
            # (bs, l, c) -> (bs, l, num_class)
            x, _ = self.residual_model(x, use_activation=use_activation)
            assert x.shape[2] == self.num_class
        else:
            # Linear layer to get number of classes
            x = self.linear_to_classes(x1)
            if use_activation:
                x = self.activation(x)
        return x
