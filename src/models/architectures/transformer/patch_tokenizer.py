import torch
from torch import nn


class PatchTokenizer(nn.Module):
    """
    Patch tokenizer for transformer encoder, patches input to decrease sequence length.
    :param channels: Number of channels in input.
    :param emb_dim: Embedding dimension.
    :param patch_size: Patch size.
    """

    def __init__(self, channels: int = 2, emb_dim: int = 192, patch_size: int = 36) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.linear_projection = nn.Linear(self.patch_size*channels, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for patch tokenizer.
        :param x: Input tensor (bs, l, c).
        :return: Output tensor (bs, l_c, e).
        """
        x = x.permute(0, 2, 1)
        bs, c, length = x.shape  # (bs, c, l)
        assert length % self.patch_size == 0, "Sequence not divisible by patch_size"

        # (bs, c, no_of_patches, patch_l) -> (bs, no_of_patches, c, patch_l)
        x = x.view(bs, c, length // self.patch_size,
                   self.patch_size).permute(0, 2, 1, 3)
        # (bs, no_of_patches, c, patch_l) -> (bs, len // patch_l, c*patch_l)
        x = x.reshape(bs, length // self.patch_size, c*self.patch_size)

        # linear projection to embedding dimension
        x = self.linear_projection(x)
        return x
