import pandas as pd
import torch

from ..logger.logger import logger


class Model:
    """
    Model class with basic methods for training and evaluation. This class should be overwritten by the user.
    """

    def __init__(self, config: dict) -> None:
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

    # TODO Make train have X_train and X_test as input which are already splitted!
    def train(self, data: pd.DataFrame) -> None:
        """
        Train function for the model. This function should be overwritten by the user.
        :param data: labelled data
        """
        pass

    def pred(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prediction function for the model. This function should be overwritten by the user.
        :param data: unlabelled data
        :return:
        """
        return pd.DataFrame([1, 2])

    def save(self, path: str) -> None:
        """
        Save function for the model. This function should be overwritten by the user.
        :param path: path to save the model to
        """
        pass

    def evaluate(self, pred: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Evaluation function for the model. This function should be overwritten by the user.
        :param pred: predictions
        :param target: targets
        """
        # Evaluate function
        pass

    def load(self, path: str) -> None:
        """
        Load function for the model. This function should be overwritten by the user.
        :param path: path to load the model from
        """
        pass

    def get_type(self) -> str:
        """
        Get type function for the model. This function should be overwritten by the user.
        """
        pass


class ModelException(Exception):
    """
    Exception class for the model.
    """

    def __init__(self, message: str) -> None:
        self.message = message
