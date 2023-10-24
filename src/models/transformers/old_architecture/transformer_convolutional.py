from torch import nn, Tensor
import torch

# Base imports
from ..architecture.tokenizer import ConvTokenizer, SimpleTokenizer
from ..base.encoder_block import TransformerEncoderBlock
from ..architecture.pooling import SeqPool
from src.logger.logger import logger


class ConvolutionalTransformerEncoder(nn.Module):
    """
    Transformer encoder with convolutional tokenizer.

    Args:
        kernel: kernel size for convolutional tokenizer
        depth: depth of convolutional tokenizer
        hidden_layers: hidden layers for convolutional tokenizer
        heads: number of heads for transformer encoder
        emb_dim: embedding dimension
        forward_dim: forward dimension for transformer encoder
        dropout: dropout for transformer encoder
        attention_dropout: dropout for attention in transformer encoder
        layers: number of transformer encoder layers
        channels: number of channels in input
        sequence_size: sequence size of input
        num_class: number of classes for regression
    """

    def __init__(
        self, kernel: int = 3, depth: int = 2, hidden_layers: int = 64,
        heads: int = 4, emb_dim: int = 256, forward_dim: int = 2*256,
        dropout: float = 0.1, attention_dropout: float = 0.1, layers: int = 7,
        channels: int = 2, sequence_size: int = 17280, num_class: int = 2
    ) -> None:
        super(ConvolutionalTransformerEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.sequence_size = sequence_size

        self.tokenizer = SimpleTokenizer(
            channels, emb_dim, hidden_layers, kernel, depth)

        with torch.no_grad():
            x = torch.randn([1, channels, sequence_size])
            out = self.tokenizer(x)
            _, _, l_c = out.shape
            logger.debug("Convolutional transformer attention size" + str(l_c))

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
