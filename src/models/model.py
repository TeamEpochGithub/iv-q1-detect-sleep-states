import numpy as np
import pandas as pd
import wandb

from ..logger.logger import logger
from ..util.hash_config import hash_config


class Model:
    """
    Model class with basic methods for training and evaluation. This class should be overwritten by the user.
    """

    def __init__(self, config: dict, name: str) -> None:
        self.model_type = "base-model"
        # Init function
        if config is None:
            self.config = None
        else:
            self.config = config
            self.hash = hash_config(config, length=5)

        self.name = name

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

    def pred(self, X_pred: np.ndarray) -> list[float, float]:
        """
        Prediction function for the model. This function should be overwritten by the user.
        :param X_pred: unlabeled data (step, features)
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

    def log_train_test(self, avg_losses: list, avg_val_losses: list, epochs: int) -> None:
        """
        Log the train and test loss to wandb.
        :param avg_losses: list of average train losses
        :param avg_val_losses: list of average test losses
        :param epochs: number of epochs
        """
        log_dict = {
            'epoch': list(range(epochs)),
            'train_loss': avg_losses,
            'val_loss': avg_val_losses
        }
        log_df = pd.DataFrame(log_dict)
        # Convert to a long format
        long_df = pd.melt(log_df, id_vars=['epoch'], var_name='loss_type', value_name='loss')

        table = wandb.Table(dataframe=long_df)
        # Field to column in df
        fields = {"step": "epoch", "lineVal": "loss", "lineKey": "loss_type"}
        custom_plot = wandb.plot_table(
            vega_spec_name="team-epoch-iv/trainval",
            data_table=table,
            fields=fields,
            string_fields={"title": "Train and validation loss of model " + self.name}
        )
        if wandb.run is not None:
            wandb.log({f"{self.name}": custom_plot})


class ModelException(Exception):
    """
    Exception class for the model.
    """

    def __init__(self, message: str) -> None:
        self.message = message
