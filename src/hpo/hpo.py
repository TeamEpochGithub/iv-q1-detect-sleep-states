# This is the hyperparameter optimization class file


class HPO:
    def __init__(self, model, data, params):
        self.model = model
        self.data = data
        self.params = params

    def optimize(self):
        print("Optimizing hyperparameters")
        return 1
