import numpy as np
import pandas as pd
import torch

from ..logger.logger import logger


class Model:
    """
    Model class with basic methods for training and evaluation. This class should be overwritten by the user.
    """

    def __init__(self, config: dict) -> None:
        self.model_type = "base-model"
        # Init function
        if config is None:
            self.config = None
        else:
            self.config = config
        # Check if gpu is available, else return an exception
        if not torch.cuda.is_available():
            logger.critical("GPU not available")
            raise ModelException("GPU not available")

        logger.info(f"--- Device set to model {type(self).__name__}: " + torch.cuda.get_device_name(0))
        self.device = torch.device("cuda")

    def get_type(self) -> str:
        """
        Get type function for the model.
        :return: type of the model
        """
        return self.model_type

    def load_config(self, config: dict) -> None:
        """
        Load config function for the model. This function should be overwritten by the user.
        :param config: configuration to set up the model
        """
        logger.info("--- Loading configuration of model not necessary or not implemented")

    def get_default_config(self) -> dict:
        """
        Get default config function for the model. This function should be overwritten by the user.
        :return: default config
        """
        logger.info("--- No default configuration of model or not implemented")
        return {}

    def train(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        """
        Train function for the model. This function should be overwritten by the user.
        :param X_train: the training data
        :param X_test: the test data
        :param y_train: the training labels
        :param y_test: the test labels
        """
        # Get hyperparameters from config (epochs, lr, optimizer)
        logger.info("--- Training of model not necessary or not implemented")

    def train_full(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model on the full dataset. This function should be overwritten by the user.
        :param X_train: the training data
        :param y_train: the training labels
        """
        # Get hyperparameters from config (epochs, lr, optimizer)
        logger.info("--- Training of model not necessary or not implemented")

    def pred(self, X_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Prediction function for the model. This function should be overwritten by the user.
        :param X_pred: unlabeled data
        :return: the predictions
        """
        logger.critical("--- Prediction of base class called. Did you forget to override it?")
        raise ModelException("Prediction of base class called. Did you forget to override it?")

    def evaluate(self, pred: pd.DataFrame, target: pd.DataFrame) -> float:
        """
        Evaluation function for the model. This function should be overwritten by the user.
        :param pred: predictions
        :param target: actual labels
        """
        # Evaluate function
        logger.critical("--- Evaluation of base class called. Did you forget to override it?")
        raise ModelException("Evaluation of base class called. Did you forget to override it?")

    def save(self, path: str) -> None:
        """
        Save function for the model. This function should be overwritten by the user.
        :param path: path to save the model to
        """
        logger.info("--- Nothing to save or not implemented")

    def load(self, path: str, only_hyperparameters: bool = False) -> None:
        """
        Load function for the model. This function should be overwritten by the user.
        :param path: path to load the model from
        :param only_hyperparameters: whether to only load the hyperparameters
        """
        logger.info("--- Nothing to load or not implemented")

    def reset_optimizer(self) -> None:
        """
        Reset the optimizer to the initial state. Useful for retraining the model. This function should be overwritten by the user.
        """
        logger.critical("--- Resetting optimizer of base class called. Did you forget to override it?")
        raise ModelException("Resetting optimizer of base class called. Did you forget to override it?")


class ModelException(Exception):
    """
    Exception class for the model.
    """

    def __init__(self, message: str) -> None:
        self.message = message
