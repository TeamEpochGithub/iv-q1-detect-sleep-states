from torch import nn
import torch


class MLP(nn.Module):
    """MLP block"""

    def __init__(self, emb_dim: int, feat_dim: int, dropout: float = 0):
        super().__init__()
        self.layer1 = nn.Linear(emb_dim, feat_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(feat_dim, emb_dim)

        # below init from torchvision
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.normal_(self.layer1.bias, std=1e-6)
        nn.init.normal_(self.layer2.bias, std=1e-6)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x
