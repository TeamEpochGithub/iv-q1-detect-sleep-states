# Create a class for ensemble learning

# Imports
import numpy as np


class Ensemble:

    # Init function
    def __init__(self, models=None, weight_matrix=None, combination_method="addition"):
        if models is None:
            self.models = []
        else:
            self.models = models

        if weight_matrix is None:
            self.weight_matrix = np.ones(len(self.models))
        else:
            self.weight_matrix = weight_matrix

    def pred(self, data):
        # Run each model
        predictions = []
        for model in self.models:
            predictions += model.pred(data)

        # Weight the predictions
        predictions = np.array(predictions)
        predictions = np.average(
            predictions, axis=0, weights=self.weight_matrix)

        return predictions
