import numpy as np
import torch


def patch_x_data(x: torch.tensor, patch_size: int) -> torch.tensor:
    """
    Patch the x data which has a shape of (window, seq_len, x) to (window, seq_len // patch_size, patch_size * x)
    :param x: x data
    :param patch_size: patch size
    :return: patched x data
    """
    # Patch the data for the features
    x = torch.reshape(
        x, (x.shape[0], x.shape[1] // patch_size, patch_size, x.shape[2]))
    x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
    return x


def patch_y_data(y: torch.tensor, patch_size: int) -> torch.tensor:
    """
    Patch the y data which has a shape of (window, seq_len, y) to (window, seq_len // patch_size, patch_size * y)
    :param y: y data
    :param patch_size: patch size
    :param comb_method: combination method
    :return: patched y data
    """
    y = torch.reshape(
        y, (y.shape[0], y.shape[1] // patch_size, patch_size, y.shape[2]))
    y = torch.transpose(y, 1, 2)
    y = torch.mode(y, 1).values
    return y


def unpatch_data(d: np.array, patch_size: int) -> np.array:
    """
    Unpatch the data which has a shape of (seq_len // patch_size, pred)
    :param d: data
    :param patch_size: patch size
    :return: unpatched data
    """
    # TODO: Complete this function #144
    pass
