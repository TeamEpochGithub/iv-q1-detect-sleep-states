from .positional_encoding import FixedPositionalEncoding
from .encoder_layer import EncoderLayer
from src.logger.logger import logger

from torch import nn
from torch.nn.modules import BatchNorm1d
import copy
import torch

class Encoder(nn.Module):
    def __init__(self, tokenizer, pe, seq_len, d_model, N, heads):
        super().__init__()
        self.N = N
        self.tokenizer = tokenizer
        self.pe = pe

        with torch.no_grad():
            x = torch.randn([1, channels, seq_len])
            out = self.tokenizer(x)
            _, _, l_c = out.shape
            logger.debug("Transformer attention size" + str(l_c))
        
        self.pe = FixedPositionalEncoding(d_model, max_len=l_c)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = BatchNorm1d(d_model)
    def forward(self, src, mask):
        x = self.tokenizer(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])