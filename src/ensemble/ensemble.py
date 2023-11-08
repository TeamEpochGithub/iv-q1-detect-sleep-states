# Create a class for ensemble learning

# Imports
import numpy as np

from ..logger.logger import logger
from typing import List
from ..configs.load_model_config import ModelConfigLoader


class Ensemble:

    # Init function
    def __init__(self, model_configs: List[ModelConfigLoader] = None, weight_matrix: List[int] = None, combination_method: str = "addition"):
        if model_configs is None:
            self.model_configs = []
        else:
            self.model_configs = model_configs

        if weight_matrix is None:
            self.weight_matrix = np.ones(len(self.model_configs))
        elif len(weight_matrix) != len(self.model_configs):
            logger.critical("Weight matrix length does not match number of models")
            raise ValueError("Weight matrix length does not match number of models")
        elif np.sum(weight_matrix) != 1:
            logger.critical("Weight matrix must sum to 1")
            raise ValueError("Weight matrix must sum to 1")
        elif np.any(weight_matrix) <= 0:
            logger.critical("Weight matrix must be positive")
            raise ValueError("Weight matrix must be positive")
        else:
            self.weight_matrix = weight_matrix

    def get_models(self):
        """
        Get the models from the ensemble
        :return: the models
        """
        return self.model_configs

    def pred(self, data: np.ndarray, pred_with_cpu: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Prediction function for the ensemble.
        Feeds the models data window-by-window, averages their predictions
        and converts the window-relative steps to absolute steps since the start of the series

        :param data: 3D tensor with shape (window, n_timesteps, n_features)
        :param pred_with_cpu: whether to use the cpu for prediction
        :return: 3D array with shape (window, 2), with onset and wakeup steps (nan if no detection)
        """
        logger.info("Predicting with ensemble")
        logger.info("Data shape: " + str(data.shape))
        # Run each model
        predictions = []
        confidences = []
        # model_pred is (onset, wakeup) tuples for each window
        for model_config in self.model_configs:
            model = model_config.get_model()
            # If the model has the device attribute, it is a pytorch model and we want to pass the pred_cpu argument.
            if hasattr(model, 'device'):
                model_pred = model.pred(data, pred_with_cpu=pred_with_cpu)
            else:
                model_pred = model.pred(data)

            # Model_pred is tuple of np.array(onset, awake), np.array(confidences) for each window
            # Split the series of tuples into two column
            predictions.append(model_pred[0])
            confidences.append(model_pred[1])

        # TODO: consider how to combine non-Nan and NaNs in the predictions #146

        # Weight the predictions
        predictions = np.array(predictions)
        aggregate = np.average(
            predictions, axis=0, weights=self.weight_matrix)

        # Weight the confidences
        confidences = np.array(confidences)
        aggregate_confidences = np.average(
            confidences, axis=0, weights=self.weight_matrix)

        return aggregate, aggregate_confidences
