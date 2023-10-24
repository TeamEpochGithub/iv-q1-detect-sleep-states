import torch.nn as nn
from .encoder import Encoder
from .tokenizer import PatchTokenizer, ConvTokenizer, SimpleTokenizer
from .positional_encoding import FixedPositionalEncoding

class EncoderConfig(nn.module):
    def __init__(self, tokenizer: str = "patch", tokenizer_args: dict = {}, pe: str = "fixed", emb_dim: int = 192, forward_dim: int = 2048, n_layers: int = 6, heads: int = 8, patch_size: int = 36):
        super().__init__()
        self.tokenizer = get_tokenizer(tokenizer, emb_dim, tokenizer_args)
        self.pe = get_positional_encoding(pe, emb_dim)
        self.model = Encoder(self.tokenizer, self.pe, emb_dim, forward_dim, n_layers, heads)
    def forward(self, src, mask):
        x = self.tokenizer(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

def get_tokenizer(tokenizer, emb_dim, tokenizer_args={}):
    if tokenizer == "patch":
        return PatchTokenizer(emb_dim, **tokenizer_args)
    elif tokenizer == "unet_encoder":
        return ConvTokenizer(emb_dim=emb_dim, **tokenizer_args)
    elif tokenizer == "simple_conv":
        return SimpleTokenizer(emb_dim, **tokenizer_args)


def get_positional_encoding(pe, emb_dim):
    if pe == "fixed":
        return FixedPositionalEncoding(emb_dim)
