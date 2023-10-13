import torch.nn.functional as F


def get_activation_fn(activation: str):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    elif activation == 'sigmoid':
        return F.sigmoid
    raise ValueError(f"Activation should be relu/gelu, not {activation}.")
