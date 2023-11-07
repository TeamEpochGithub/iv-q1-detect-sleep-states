# This is the base class for loss
from torch import nn
import torch

from src.loss.event_regression_loss_rmse import EventRegressionLossRMSE
from .regression_loss import RegressionLoss
from .event_regression_loss import EventRegressionLoss
from .nan_regression_loss import NanRegressionLoss
from .event_regression_loss_mae import EventRegressionLossMAE


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
    def get_loss(loss_name: str, **kwargs) -> nn.Module:
        """
        This function looks up the correct loss function and returns it.
        :param loss_name: name of the loss function
        :return: the loss function
        """
        match loss_name:
            case "mse-torch":
                return nn.MSELoss(**kwargs)
            case "mae-torch":
                return nn.L1Loss(**kwargs)
            case "ce-torch":
                return nn.CrossEntropyLoss(**kwargs)
            case "bce-torch":
                return nn.BCELoss(**kwargs)
            case "focal-loss":
                return FocalLoss(**kwargs)
            case "shrinkage-loss":
                return ShrinkageLoss(**kwargs)
            case "bce-logits-torch":
                return nn.BCEWithLogitsLoss(**kwargs)
            case "kldiv-torch":
                return nn.KLDivLoss(**kwargs, reduction="batchmean")
            case "regression":
                return RegressionLoss()
            case "event-regression":
                return EventRegressionLoss()
            case "event-regression-mae":
                return EventRegressionLossMAE()
            case "event-regression-rmse":
                return EventRegressionLossRMSE()
            case "nan-regression":
                return NanRegressionLoss()
            case _:
                raise LossException("Loss function not found: " + loss_name)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha: int = 0.8, gamma: int = 2, c: int = 0.2, smooth=1):
        l1_loss = torch.abs(inputs - targets)
        focal_loss = l1_loss ** (2 + gamma)

        return focal_loss.mean()


class ShrinkageLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, reduction: str = "mean"):
        super(ShrinkageLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, alpha: int = 2, c: int = 0.2):

        l1_loss = torch.abs(inputs - targets)
        shrinkage_loss = ((l1_loss) ** 2) * torch.exp(targets) / \
            (1 + torch.exp(alpha * (c - l1_loss)))
        if self.reduction == "mean":
            return shrinkage_loss.mean()
        return shrinkage_loss
