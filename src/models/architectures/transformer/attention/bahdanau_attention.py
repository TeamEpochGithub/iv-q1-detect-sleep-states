from torch import nn
import torch
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    '''
    Bahdanau Attention Module
    :param hidden_size: Hidden size of the attention module.
    :param bidirectional: Whether the attention module is bidirectional.
    '''
    def __init__(
        self,
        hidden_size: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.D = 2 if bidirectional else 1
        # Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate.
        self.Ws = nn.Linear(self.D * hidden_size, self.D * hidden_size)
        self.Va = nn.Linear(self.D * hidden_size, 1)
        self.Wa = nn.Linear(self.D * hidden_size, self.D * hidden_size)
        self.Ua = nn.Linear(self.D * hidden_size, self.D * hidden_size)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor]:
        '''
        Forward pass of the bahdanau attention module.
        :param q: Query tensor.
        :param k: Key tensor.
        :param v: Value tensor.
        :return: Output of layer tensor.
        '''
        # q,k,v=dec,enc_all,enc_all
        # project decoder initial states
        q = q.unsqueeze(1) if len(q.shape) == 2 else q

        attn_scores = F.tanh(self.Wa(q) + self.Ua(k))  # (B,L,D*H)
        attn_scores = self.Va(attn_scores).squeeze(-1)  # BL
        attn_scores = F.softmax(attn_scores, dim=1).unsqueeze(2)  # BL1
        attn_v = torch.mul(attn_scores, v).sum(dim=1, keepdim=True)  # B1H
        return attn_v, attn_scores  # (B,D*H), (B,L)
