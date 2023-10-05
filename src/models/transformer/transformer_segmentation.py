from torch import nn, Tensor
from .positional_encoding import get_pos_encoder
from torch.nn.modules import TransformerEncoderLayer
from .encoder import TransformerBatchNormEncoderLayer
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
