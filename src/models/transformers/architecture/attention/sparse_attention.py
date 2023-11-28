import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SparseAttention(nn.Module):
    '''
    Sparse Attention Module
    :param heads: Number of heads in the multihead attention.
    :param attn_mode: Type of sparse attention.
    :param local_attn_ctx: Context size for local attention.
    :param block_size: Block size for sparse attention.
    '''

    def __init__(self, heads: int, attn_mode: str = 'e', local_attn_ctx: int = 10, block_size: int = 32) -> None:
        super(SparseAttention, self).__init__()
        self.heads = heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.block_size = block_size

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the sparse attention module.
        :param q: Query tensor.
        :param k: Key tensor.
        :param v: Value tensor.
        :return: Output of layer tensor.
        '''
        return self.blocksparse_attention_impl(q, k, v, self.heads, self.attn_mode, self.local_attn_ctx, self.block_size)

    def blocksparse_attention_impl(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                   heads: int, attn_mode: str = "e", local_attn_ctx: int = 10, block_size: int = 32) -> torch.Tensor:
        '''
        Block sparse attention implementation.
        :param q: Query tensor.
        :param k: Key tensor.
        :param v: Value tensor.
        :param heads: Number of heads in the multihead attention.
        :param attn_mode: Type of sparse attention.
        :param local_attn_ctx: Context size for local attention.
        :param block_size: Block size for sparse attention.
        :return: Output of layer tensor.
        '''

        n_ctx = q.size()[1]
        if attn_mode == 'strided':
            q = self.strided_transpose(q, n_ctx, local_attn_ctx, block_size)
            k = self.strided_transpose(k, n_ctx, local_attn_ctx, block_size)
            v = self.strided_transpose(v, n_ctx, local_attn_ctx, block_size)
        n_state = q.size()[-1] // heads
        scale_amount = 1.0 / np.sqrt(n_state)
        w = torch.matmul(q, k.transpose(-2, -1))
        w = F.softmax(w * scale_amount, dim=-1)
        a = torch.matmul(w, v)
        if attn_mode == 'strided':
            n, t, embd = a.size()
            block_ctx = n_ctx // local_attn_ctx
            a = torch.reshape(a, [n, local_attn_ctx, block_ctx, embd])
            a = a.permute(0, 2, 1, 3)
            a = torch.reshape(a, [n, t, embd])
        return a

    def attention_impl(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, attn_mode: str = "all", local_attn_ctx: int = 10) -> torch.Tensor:
        '''
        Attention implementation.
        :param q: Query tensor.
        :param k: Key tensor.
        :param v: Value tensor.
        :param heads: Number of heads in the multihead attention.
        :param attn_mode: Type of sparse attention.
        :param local_attn_ctx: Context size for local attention.
        :return: Output of layer tensor.
        '''
        q = self.split_heads(q, heads)
        k = self.split_heads(k, heads)
        v = self.split_heads(v, heads)
        n_timesteps = k.size()[2]
        mask = self.get_attn_mask(
            n_timesteps, attn_mode, local_attn_ctx).float()
        w = torch.matmul(q, k.transpose(-2, -1))
        scale_amount = 1.0 / np.sqrt(q.size()[-1])
        w = w * scale_amount
        w = w * mask + -1e9 * (1 - mask)
        w = F.softmax(w, dim=-1)
        a = torch.matmul(w, v)
        a = self.merge_heads(a)
        return a

    def get_attn_mask(self, n: int, attn_mode: str = "all", local_attn_ctx: int = 10) -> torch.Tensor:
        '''
        Get attention mask.
        :param n: Sequence length.
        :param attn_mode: Type of sparse attention.
        :param local_attn_ctx: Context size for local attention.
        :return: Attention mask.
        '''
        if attn_mode == 'all':
            b = torch.tril(torch.ones([n, n]))
        elif attn_mode == 'local':
            bandwidth = local_attn_ctx
            ctx = min(n - 1, bandwidth - 1)
            b = torch.tril(torch.ones([n, n]), ctx)
        elif attn_mode == 'strided':
            stride = local_attn_ctx
            x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
            y = torch.transpose(x, 0, 1)
            z = torch.zeros([n, n], dtype=torch.int32)
            q = z + x
            k = z + y
            c1 = q >= k
            c2 = torch.eq(torch.fmod(q - k, stride), 0)
            c3 = torch.logical_and(c1, c2)
            b = c3.float()
        else:
            raise ValueError('Not yet implemented')
        b = torch.reshape(b, [1, 1, n, n])
        return b

    def split_heads(self, x: torch.Tensor, n: int) -> torch.Tensor:
        '''
        Split heads.
        :param x: Input tensor.
        :param n: Number of heads.
        :return: Output tensor.
        '''
        return torch.transpose(self.split_states(x, n), 0, 2, 1, 3)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Merge heads.
        :param x: Input tensor.
        :return: Output tensor.
        '''
        return self.merge_states(torch.transpose(x, 0, 2, 1, 3))

    @staticmethod
    def strided_transpose(x: torch.Tensor, n_ctx: int, local_attn_ctx: int = 10, block_size: int = 32) -> torch.Tensor:
        '''
        Compute strided transpose.
        :param x: Input tensor.
        :param n_ctx: Sequence length.
        :param local_attn_ctx: Context size for local attention.
        :param block_size: Block size for sparse attention.
        :return: Output tensor.
        '''

        # Calculate the number of blocks
        block_ctx = n_ctx // local_attn_ctx
        assert block_ctx % block_size == 0, f'{block_ctx}, {block_size}'
        n, t, embd = x.size()
        x = torch.reshape(x, [n, block_ctx, local_attn_ctx, embd])
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, [n, t, embd])
        return x

    @staticmethod
    def split_states(x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Split states into multiple heads. (batch, pixel, state) -> (batch, pixel, head, head_state)
        :param x: Input tensor.
        :param n: Number of heads.
        :return: Output tensor.
        """
        x_shape = x.size()
        m = x_shape[-1]
        new_x_shape = x_shape[:-1] + [n, m // n]
        return torch.reshape(x, new_x_shape)

    @staticmethod
    def merge_states(x: torch.Tensor) -> torch.Tensor:
        """
        Merge states from multiple heads. (batch, pixel, head, head_state) -> (batch, pixel, state)
        :param x: Input tensor.
        :return: Output tensor.
        """
        x_shape = x.size()
        new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
        return torch.reshape(x, new_x_shape)
