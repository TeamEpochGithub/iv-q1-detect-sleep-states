# Model class with basic methods for training and evaluation


class Model:
    def __init__(self, config):
        # Init function
        if config is None:
            self.config = None
        else:
            self.config = config

    def train(self, data):
        # Train function
        pass

    def pred(self, data):
        # Predict function
        return [1,2]

    def evaluate(self, data):
        # Evaluate function
        pass
