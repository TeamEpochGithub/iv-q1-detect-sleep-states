# This is the base class for loss
from torch import nn


class LossException(Exception):
    pass


class Loss():
    """
    This is a static class for loss functions.
    """

    def __init__(self):
        # Init function
        pass

    @staticmethod
    def get_loss(loss_name):
        """
        This function looks up the correct loss function and returns it.
        :param loss_name: name of the loss function
        :return: the loss function
        """
        match loss_name:
            case "mse-torch":
                return nn.MSELoss()
            case "mae-torch":
                return nn.L1Loss()
            case "crossentropy-torch":
                return nn.CrossEntropyLoss()
            case "binarycrossentropy-torch":
                return nn.BCELoss()
            case _:
                raise LossException("Loss function not found: " + loss_name)
