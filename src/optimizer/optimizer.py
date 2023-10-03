import torch.optim as optim
from torch import nn


class OptimizerException(Exception):
    pass


class Optimizer:
    """
    This is a static class for optimizer functions.
    """

    def __init__(self) -> None:
        # Init function
        pass

    @staticmethod
    def get_optimizer(optimizer_name: str, learning_rate: float, model: nn.Module):
        """
        This function returns the correct optimizer function.
        :param optimizer_name: name of the optimizer
        :param learning_rate: learning rate for the optimizer
        :param model: model to optimize
        :return: optimizer function
        """
        match optimizer_name:
            case "adam-torch":
                return optim.Adam(model.parameters(), lr=learning_rate)
            case "sgd-torch":
                return optim.SGD(model.parameters(), lr=learning_rate)
            case "adagrad-torch":
                return optim.Adagrad(model.parameters(), lr=learning_rate)
            case "adadelta-torch":
                return optim.Adadelta(model.parameters(), lr=learning_rate)
            case "rmsprop-torch":
                return optim.RMSprop(model.parameters(), lr=learning_rate)
            case _:
                raise OptimizerException("Optimizer function not found: " + optimizer_name)
