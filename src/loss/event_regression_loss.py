import torch.nn as nn


class EventRegressionLoss(nn.Module):
    def __init__(self):
        super(EventRegressionLoss, self).__init__()

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
        loss = (y_true[:, :2] - y_pred) ** 2

        # Use mask to turn into nans
        mask = mask[:, :2]
        mask[mask == 0] = float('nan')

        # Apply mask
        loss = loss * mask

        return loss.nanmean()
