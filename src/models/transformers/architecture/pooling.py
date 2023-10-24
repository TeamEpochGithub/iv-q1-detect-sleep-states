from torch import nn
import torch


class SeqPool(nn.Module):
    """
    Sequence pooling layer with softmax.
    :param emb_dim: Embedding dimension.
    """

    def __init__(self, emb_dim: int = 256) -> None:
        super().__init__()
        self.dense = nn.Linear(emb_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the sequence pooling layer.
        :param x: Input tensor.
        :return: Output tensor.
        """
        bs, _, emb_dim = x.shape
        identity = x
        x = self.dense(x)
        x = x.permute(0, 2, 1)
        x = self.softmax(x)
        x = x @ identity
        x = x.reshape(bs, emb_dim)
        return x


class LSTMPooling(nn.Module):
    """
    LSTM pooling layer.
    :param emb_dim: Embedding dimension.
    """

    def __init__(self, emb_dim: int = 256) -> None:
        super(LSTMPooling, self).__init__()
        self.emb_dim = emb_dim
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim // 2,
                            proj_size=1, bidirectional=False, batch_first=True, dropout=0.1)

    def forward(self, all_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM pooling layer.
        :param all_hidden_states: Input tensor.
        :return: Output tensor.
        """
        out, _ = self.lstm(all_hidden_states, None)
        out = out.reshape(-1, out.shape[1])
        return out


class NoPooling(nn.Module):
    """
    No pooling layer.
    :param emb_dim: Embedding dimension.
    """

    def __init__(self, emb_dim: int = 256) -> None:
        super(NoPooling, self).__init__()
        self.emb_dim = emb_dim

    def forward(self, all_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the no pooling layer.
        :param all_hidden_states: Input tensor.
        :return: Output tensor.
        """
        return all_hidden_states.reshape(all_hidden_states.shape[0], -1)
