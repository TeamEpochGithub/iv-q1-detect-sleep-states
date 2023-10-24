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
        encoders = []
        for _ in range(0, n_layers):
            encoders.append(
                EncoderLayer(emb_dim, heads, forward_dim)
            )
        self.encoder_stack = nn.Sequential(*encoders)
        self.norm = BatchNorm1d(emb_dim)
    def forward(self, src):
        x = self.tokenizer(src)
        x = self.pe(x)
        x = self.encoder_stack(x)
        return self.norm(x)
    
def get_clones(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])