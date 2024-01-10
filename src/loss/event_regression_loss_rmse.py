import torch
import torch.nn as nn


class EventRegressionLossRMSE(nn.Module):
    def __init__(self):
        super(EventRegressionLossRMSE, self).__init__()

    def forward(self, y_pred, y_true, mask):
        """
        :param y_pred: predicted values
        :param y_true: true values
        :return: loss
        """
        # If y_true is 1, it is a Nan value so make loss 0 for y_pred
        # If y_true is 0, it is a valid value so calculate loss for y_pred
        # Always calculate loss for y_pred nan predictions

        # Calculate loss as mean absolute error
        loss = torch.sqrt((y_true[:, :2] - y_pred) ** 2)

        # Use mask to get proper loss
        mask = mask[:, :2]

        # Apply mask
        loss = loss * mask
        loss = torch.sum(loss) / torch.sum(mask)
        return loss
