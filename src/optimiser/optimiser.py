import torch.optim as optim
from torch import nn


class OptimiserException(Exception):
    pass


class Optimiser:
    """
    This is a static class for optimiser functions.
    """

    def __init__(self) -> None:
        # Init function
        pass

    @staticmethod
    def get_optimiser(optimiser_name: str, learning_rate: float, weight_decay: float = 0.0, model: nn.Module = None):
        """
        This function returns the correct optimiser function.
        :param optimiser_name: name of the optimiser
        :param learning_rate: learning rate for the optimiser
        :param model: model to optimize
        :return: optimiser function
        """
        match optimiser_name:
            case "adam-torch":
                return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            case "sgd-torch":
                return optim.SGD(model.parameters(), lr=learning_rate)
            case "adagrad-torch":
                return optim.Adagrad(model.parameters(), lr=learning_rate)
            case "adadelta-torch":
                return optim.Adadelta(model.parameters(), lr=learning_rate)
            case "rmsprop-torch":
                return optim.RMSprop(model.parameters(), lr=learning_rate)
            case _:
                raise OptimiserException("Optimiser function not found: " + optimiser_name)
