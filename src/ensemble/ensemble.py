# Create a class for ensemble learning

# Imports
import numpy as np
import pandas as pd

from ..logger.logger import logger


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

    def pred(self, data: np.ndarray) -> np.ndarray:
        """
        Prediction function for the ensemble.
        Feeds the models data window-by-window, averages their predictions
        and converts the window-relative steps to absolute steps since the start of the series

        :param data: 3D tensor with shape (window, n_timesteps, n_features)
        :return: 3D array with shape (window, 2), with onset and wakeup steps (nan if no detection)
        """
        logger.info("Predicting with ensemble")
        # Run each model
        predictions = []
        for model in self.models:
            # group data by series_id, apply model.pred to each group, and get the output pairs
            # get the step at the index of the prediction
            model_pred = np.array([model.pred(window) for window in data])

            # split the series of tuples into two column
            predictions.append(model_pred)

        # TODO: consider how to combine non-Nan and NaNs in the predictions #146

        # Weight the predictions
        predictions = np.array(predictions)
        aggregate = np.average(
            predictions, axis=0, weights=self.weight_matrix)

        return aggregate
