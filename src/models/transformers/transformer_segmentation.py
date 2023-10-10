import math
from torch import nn, Tensor
from .architecture.positional_encoding import get_pos_encoder
from torch.nn.modules import TransformerEncoderLayer
from .architecture.encoder import TransformerBatchNormEncoderLayer
from .utils import get_activation_fn


class SegmentTransformer(nn.module):
    """
    Transformer for time series segmentation.

    """

    def __init__(
        self,
        feat_dim: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
        dropout: float = 0.1,
        pos_encoding: str = "fixed",
        activation: str = "gelu",
        norm: str = "BatchNorm",
        freeze: bool = False,
    ):
        super(SegmentTransformer, self).__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads

        # Linear model to project the input to the d_model dimension
        self.project_inp = nn.Linear(feat_dim, d_model)

        # Get positional encoding
        self.pos_enc = get_pos_encoder(pos_encoding)(
            d_model, dropout=dropout * (1.0 - freeze), max_len=seq_len
        )

        # Build the transformer encoder
        if norm == "LayerNorm":
            encoder_layer = TransformerEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation,
            )
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation,
            )

        # Set the transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)

        # Set the activation function
        self.act = get_activation_fn(activation)

        # Set the dropout layer
        self.dropout1 = nn.Dropout(dropout)

        # Set the feature dimension
        self.feat_dim = feat_dim

        # Set the number of classes
        self.num_classes = num_classes

        # Build the output module
        self.output_layer = self.build_output_module(
            d_model, seq_len, num_classes)

    def build_output_module(
            self, d_model: int, seq_len: int, num_classes: int) -> nn.Module:
        """
        Build linear layer that maps from d_model*seq_len to seq_len*num_classes.
        """
        output_layer = nn.Linear(d_model*seq_len, seq_len*num_classes)

        return output_layer

    def forward(self, x: Tensor, padding_masks: Tensor) -> Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, feat_dim)
            padding_masks: padding masks of shape (batch_size, seq_len)
        Returns:
            output: (batch_size, seq_len, num_classes)
        """

        # Permute beause the transformer expects the input to be of shape (seq_len, batch_size, feat_dim)
        inp = x.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)

        # Add positional encoding
        inp = self.pos_enc(inp)

        # Get output
        output = self.transformer_encoder(
            inp, src_key_padding_mask=~padding_masks)
        output = self.act(output)
        output = output.permute(1, 0, 2)
        output = self.dropout1(output)

        # Final output
        output = output * padding_masks.unsqueeze(-1)
        # (batch_size, seq_len * d_model)
        output = output.reshape(output.shape[0], -1)
        # (batch_size, seq_len * num_classes)
        output = self.output_layer(output)

        return output
