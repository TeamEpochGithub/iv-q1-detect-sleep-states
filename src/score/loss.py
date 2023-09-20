# This is the base class for loss


class Loss():
    def __init__(self):
        # Init function
        pass

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError
