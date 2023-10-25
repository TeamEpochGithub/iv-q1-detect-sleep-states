from torch import nn
import torch
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    FeedForward block
    :param emb_dim: Embedding dimension.
    :param forward_dim: Dimension of the feed forward layer.
    :param dropout: Dropout rate.
    """

    def __init__(self, emb_dim: int, forward_dim: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer1 = nn.Linear(emb_dim, forward_dim)
        self.activation = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(forward_dim, emb_dim)

        # Below is an init from torchvision
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.normal_(self.layer1.bias, std=1e-6)
        nn.init.normal_(self.layer2.bias, std=1e-6)

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