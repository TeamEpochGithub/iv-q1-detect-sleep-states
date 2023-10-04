import torch

from ..logger.logger import logger


class Model:
    """
    Model class with basic methods for training and evaluation. This class should be overwritten by the user.
    """

    def __init__(self, config):
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
    def train(self, X_train, X_test, Y_train, Y_test):
        """
        Train function for the model. This function should be overwritten by the user.
        :param data: labelled data
        :return: None
        """
        # Get hyperparameters from config (epochs, lr, optimizer)
        logger.info("--- Training of model not necessary or not implemented")

    def pred(self, data):
        """
        Prediction function for the model. This function should be overwritten by the user.
        :param data: unlabelled data
        :return:
        """
        return [1, 2]

    def save(self, path):
        """
        Save function for the model. This function should be overwritten by the user.
        :param path: path to save the model to
        :return:
        """
        pass

    def evaluate(self, pred, target):
        """
        Evaluation function for the model. This function should be overwritten by the user.
        :param pred: predictions
        :param target: targets
        :return:
        """
        # Evaluate function
        pass

    def load(self, path):
        """
        Load function for the model. This function should be overwritten by the user.
        :param path: path to load the model from
        :return:
        """
        pass

    def get_type(self):
        """
        Get type function for the model. This function should be overwritten by the user.
        :return:
        """
        pass


class ModelException(Exception):
    """
    Exception class for the model.
    """

    def __init__(self, message):
        self.message = message
