# This is the base class for loss
import torch
import torch.optim as optim


class OptimizerException(Exception):
    pass


class Optimizer():
    """
    This is a static class for loss functions.
    """

    def __init__(self):
        # Init function
        pass

    @staticmethod
    def get_optimizer(optimizer_name, learning_rate, model):
        """
        This function looks up the correct loss function and returns it.
        :param loss_name: name of the loss function
        :return: the loss function
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
