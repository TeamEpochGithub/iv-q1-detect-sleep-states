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

    def pred(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: complete dataset with engineered features
        :return: numpy array with per every day one tuple of onset and awake
        """

        logger.info("Predicting with ensemble")
        # Run each model
        predictions = []
        for model in self.models:
            # group data by series_id, apply model.pred to each group, and get the output pairs
            model_pred = (data
                          .groupby(['series_id', 'window'])
                          .apply(model.pred).reset_index(0, drop=True))

            # split the series of tuples into two columns
            model_pred = pd.DataFrame(model_pred.to_list(), columns=['onset', 'awake'])
            predictions.append(model_pred)

        # Weight the predictions
        predictions = np.array(predictions)
        predictions = np.average(
            predictions, axis=0, weights=self.weight_matrix)

        return predictions
