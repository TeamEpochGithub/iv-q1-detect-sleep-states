# Create a class for ensemble learning

# Imports
import numpy as np

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

    def pred(self, data: np.ndarray, pred_cpu: bool) -> np.ndarray:
        """
        Prediction function for the ensemble.
        Feeds the models data window-by-window, averages their predictions
        and converts the window-relative steps to absolute steps since the start of the series

        :param data: 3D tensor with shape (window, n_timesteps, n_features)
        :param pred_cpu: whether to predict on cpu
        :return: 3D array with shape (window, 2), with onset and wakeup steps (nan if no detection)
        """
        logger.info("Predicting with ensemble")
        logger.info("Data shape: " + str(data.shape))
        # Run each model
        predictions = []
        # model_pred is (onset, wakeup) tuples for each window
        for model in self.models:
            # If the model has the device attribute, it is a pytorch model and we want to pass the pred_cpu argument.
            if hasattr(model, 'device'):
                model_pred = model.pred(data, pred_cpu)
            else:
                model_pred = model.pred(data)

            # Model_pred is (onset, wakeup) tuples for each window
            # Split the series of tuples into two column
            predictions.append(model_pred)

        # TODO: consider how to combine non-Nan and NaNs in the predictions #146

        # Weight the predictions
        predictions = np.array(predictions)
        aggregate = np.average(
            predictions, axis=0, weights=self.weight_matrix)

        return aggregate
