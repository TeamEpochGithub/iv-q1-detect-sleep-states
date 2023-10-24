from .positional_encoding import FixedPositionalEncoding
from .encoder_layer import EncoderLayer
from src.logger.logger import logger

from torch import nn
from torch.nn.modules import BatchNorm1d
import copy
import torch

class Encoder(nn.Module):
    def __init__(self, tokenizer, pe, emb_dim, forward_dim, n_layers, heads):
        super().__init__()
        self.n_layers = n_layers
        self.tokenizer = tokenizer
        self.pe = pe
        self.layers = get_clones(EncoderLayer(emb_dim, heads, forward_dim), n_layers)
        self.norm = BatchNorm1d(emb_dim)
    def forward(self, src, mask):
        x = self.tokenizer(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
def get_clones(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])