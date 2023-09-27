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

    #TODO Make train have X_train and X_test as input which are already splitted!
    def train(self, data):
        """
        Train function for the model. This function should be overwritten by the user.
        :param data: labelled data
        :return: None
        """
        pass

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


class ModelException(Exception):
    """
    Exception class for the model.
    """

    def __init__(self, message):
        self.message = message

