import torch.nn as nn


class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()

    def forward(self, y_pred, y_true, mask):
        """
        :param y_pred: predicted values
        :param y_true: true values
        :return: loss
        """
        # If y_true is 1, it is a Nan value so make loss 0 for y_pred
        # If y_true is 0, it is a valid value so calculate loss for y_pred
        # Always calculate loss for y_pred nan predictions

        # Calculate loss as mean squared error
        loss = (y_true - y_pred) ** 2

        # Apply mask
        loss = loss * mask

        return loss.mean()
