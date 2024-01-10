import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    """
    FeedForward block
    :param emb_dim: Embedding dimension.
    :param forward_dim: Dimension of the feed forward layer.
    :param dropout: Dropout rate.
    """

    def __init__(self, emb_dim: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer1 = nn.Linear(emb_dim, emb_dim * expansion)
        self.activation = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(emb_dim * expansion, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param x: Input tensor.
        :return: Output of feed forward layer tensor.
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
