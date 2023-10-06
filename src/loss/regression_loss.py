import torch.nn as nn
import torch


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


# Main
if __name__ == "__main__":
    test_pred = [[123, 444, 0.3, 0.2], [4123, 2235, 0.5, 0.2]]
    test_true = [[100, 400, 0, 0], [-1, -1, 1, 1]]

    test_pred = torch.tensor(test_pred)
    test_true = torch.tensor(test_true)
    loss = RegressionLoss()

    # Create mask from true values
    # If index 2 is 1 then index 0 is 0 else 1
    # If index 3 is 1 then index 1 is 0 else 1
    # Index 2 and 3 always have 1
    mask = torch.ones_like(test_true)
    mask[:, 0] = 1 - test_true[:, 2]
    mask[:, 1] = 1 - test_true[:, 3]

    print(loss(test_pred, test_true, mask))
