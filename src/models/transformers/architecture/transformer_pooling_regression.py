from torch import nn, Tensor
import torch

# Base imports
from ..base.encoder_block import TransformerEncoderBlock
from ..base.seq_pool import SeqPool
from src.logger.logger import logger


class PatchPoolTransformerEncoder(nn.Module):
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
        self, heads: int = 12, emb_dim: int = 768, feat_dim: int = 3072, 
        dropout: float = 0.1, layers: int = 12, patch_size: int = 16, 
        channels: int = 3, seq_len: int = 17280, num_class: int = 2
    ) -> None:
        super(PatchPoolTransformerEncoder, self).__init__()
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.linear_projection = nn.Linear(
            self.patch_size*channels, emb_dim)
        self.pos_emb = nn.Parameter(
            torch.randn(
                [1, self.seq_len // patch_size, emb_dim]
            ).normal_(std=0.02) # init from torchvision, which is inspired by BERT
        )
        self.dropout = nn.Dropout(dropout)
        encoders = []
        for _ in range(0, layers):
            encoders.append(
                TransformerEncoderBlock(
                    n_h=heads, emb_dim=emb_dim, feat_dim=feat_dim,
                    dropout=dropout
                )
            )
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
        bs, c, l = x.shape  # (bs, c, l)
        torch._assert(
            self.seq_len % self.patch_size == 0, 
            "Sequence not disivible by patch_size"
        )

        # below 2 lines creates patches using view/reshape and permutations
        # (bs, c, no_of_patches, patch_l) -> (bs, no_of_patches, c, patch_l)
        x = x.view(bs, c, self.seq_len // self.patch_size, self.patch_size).permute(0, 2, 1, 3)
        # (bs, no_of_patches, c*seq_len)
        x = x.reshape(bs, self.seq_len // self.patch_size, c*self.patch_size)

        # linear projection to embedding dimension
        x = self.linear_projection(x)

        # expands position embedding batch wise, add embedding dropout
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
