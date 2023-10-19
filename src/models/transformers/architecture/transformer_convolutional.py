from torch import nn, Tensor
import torch

# Base imports
from ..base.conv_tokenizer import ConvTokenizer
from ..base.encoder_block import TransformerEncoderBlock
from ..base.seq_pool import SeqPool


class ConvolutionalTransformerEncoder(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.

    Args:
        feat_dim: feature dimension
        max_len: maximum length of the input sequence
        d_model: the embed dim
        n_heads: the number of heads in the multihead attention models
        n_layers: the number of sub-encoder-layers in the encoder
        dim_feedforward: the dimension of the feedforward network model
        num_classes: the number of classes in the classification task
        dropout: the dropout value
        pos_encoding: positional encoding method
        act_int: the activation function of intermediate layer, relu or gelu
        act_out: the activation function of output layer, relu, gelu, sigmoid or none
        norm: the normalization layer
        freeze: whether to freeze the positional encoding layer
    """

    def __init__(
        self, conv_kernel: int = 3, conv_stride: int = 2, conv_pad: int = 3,
        pool_kernel: int = 3, pool_stride: int = 2, pool_pad: int = 1,
        heads: int = 4, emb_dim: int = 256, forward_dim: int = 2*256,
        dropout: float = 0.1, attention_dropout: float = 0.1, layers: int = 7,
        channels: int = 2, sequence_size: int = 17280, num_class: int = 2
    ):
        super(ConvolutionalTransformerEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.sequence_size = sequence_size

        self.tokenizer = ConvTokenizer(
            channels, emb_dim, conv_kernel, conv_stride, conv_pad, pool_kernel, pool_stride, pool_pad, nn.ReLU)

        with torch.no_grad():
            x = torch.randn([1, channels, sequence_size])
            out = self.tokenizer(x)
            _, _, l_c = out.shape

        self.linear_projection = nn.Linear(l_c, emb_dim)

        self.pos_emb = nn.Parameter(torch.randn(
            [1, l_c, emb_dim]).normal_(std=0.02))

        self.dropout = nn.Dropout(dropout)
        encoders = []
        for _ in range(layers):
            encoders.append(TransformerEncoderBlock(
                heads, emb_dim, forward_dim, dropout, attention_dropout))
        self.encoder_stack = nn.Sequential(*encoders)
        self.seq_pool = SeqPool(emb_dim=emb_dim)
        self.mlp_head = nn.Linear(emb_dim, num_class)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, seq_length, feat_dim)

        Returns:
            (batch_size, num_classes)
        """
        # Input is (batch_size, seq_length, feat_dim) but should be (bs, feat_dim, seq_length) for conv1d
        x = x.permute(0, 2, 1)
        bs, _, _ = x.shape

        # Apply convolutional tokenizer, output is (batch_size, emb_dim, seq_length)
        x = self.tokenizer(x)

        # Permute as positional embeddings are (batch_size, seq_length, emb_dim)
        x = x.permute(0, 2, 1)

        # Add positional embeddings
        x = self.pos_emb.expand(bs, -1, -1) + x
        x = self.dropout(x)

        # Pass through transformer encoder
        x = self.encoder_stack(x)

        # Perform sequential pooling
        x = self.seq_pool(x)

        # MLP head used to get logits
        x = self.mlp_head(x)

        x = self.act(x)

        return x
