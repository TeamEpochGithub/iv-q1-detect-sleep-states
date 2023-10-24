from .encoder import Encoder


class EncoderConfig(nn.module):
    def __init__(self, tokenizer, pe, emb_dim, forward_dim, n_layers, heads):
        super().__init__()
        self.n_layers = n_layers
        self.tokenizer = tokenizer
        self.pe = pe
        self.model = Encoder(self.tokenizer, self.pe, self.emb_dim, self.forward_dim, self.n_layers, self.heads)
    def forward(self, src, mask):
        x = self.tokenizer(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

def get_tokenizer(tokenizer, emb_dim, patch_size: int = ):
    if tokenizer == "patch":
        return PatchTokenizer(emb_dim, patch_size)
    elif tokenizer == "unet_encoder":
        return 