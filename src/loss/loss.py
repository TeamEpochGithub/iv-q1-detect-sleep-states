# This is the base class for loss
from torch import nn
from .regression_loss import RegressionLoss
from .event_regression_loss import EventRegressionLoss
from .nan_regression_loss import NanRegressionLoss
from .event_regression_loss_mae import EventRegressionLossMae


class LossException(Exception):
    pass


class Loss:
    """
    This is a static class for loss functions.
    """

    def __init__(self) -> None:
        # Init function
        pass

    @staticmethod
    def get_loss(loss_name: str) -> nn.Module:
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
            case "regression":
                return RegressionLoss()
            case "event-regression":
                return EventRegressionLoss()
            case "event-regression-mae":
                return EventRegressionLossMae()
            case "nan-regression":
                return NanRegressionLoss()
            case _:
                raise LossException("Loss function not found: " + loss_name)
