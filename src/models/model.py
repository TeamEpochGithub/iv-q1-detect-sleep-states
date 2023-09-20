# Model class with basic methods for training and evaluation


class Model:
    def __init__(self, model=None):
        # Init function
        if model is None:
            self.stored_model = None
        else:
            self.stored_model = model

    def train(self, data):
        # Train function
        pass

    def pred(self, data):
        # Predict function
        pass

    def evaluate(self, data):
        # Evaluate function
        pass
