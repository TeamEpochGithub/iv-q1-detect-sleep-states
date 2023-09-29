from ..loss.loss import Loss
from ..models.model import Model, ModelException
from ..util.state_to_event import find_events


class ClassicBaseModel(Model):
    """
    This is a sample model file. You can use this as a template for your own models.
    The model file should contain a class that inherits from the Model class.
    """

    def __init__(self, config):
        """
        Init function of the example model
        :param config: configuration to set up the model
        """
        super().__init__(config)
        self.load_config(config)

    def load_config(self, config):
        """
        Load config function for the model.
        :param config: configuration to set up the model
        :return:
        """

        # Error checks. Check if all necessary parameters are in the config.
        required = ["loss"]
        for req in required:
            if req not in config:
                raise ModelException("Config is missing required parameter: " + req)

        # Get default_config
        default_config = self.get_default_config()

        config["loss"] = Loss.get_loss(config["loss"])
        config["median_window"] = config.get("median_window", default_config["median_window"])
        config["threshold"] = config.get("threshold", default_config["threshold"])
        self.config = config

    def get_default_config(self):
        """
        Get default config function for the model.
        :return: default config
        """
        return {"median_window": 100, "threshold": .1}

    def train(self, data):
        """
        Train function for the model.
        :param data: labelled data
        :return:
        """

        # Get hyperparameters from config (epochs, lr, optimizer)
        print("----------------")
        print("Training classic baseline model not needed")
        print("----------------")

    def pred(self, data):
        """
        Prediction function for the model.
        :param data: unlabelled data for a single day window
        :return:
        """
        # Get the data from the data tuple
        state_pred = self.predict_state_labels(data)
        onset, awake = find_events(state_pred)
        return onset, awake

    def predict_state_labels(self, data):
        data['slope'] = abs(data['anglez'].diff()).clip(upper=10)
        data['movement'] = data['slope'].rolling(window=100).median()
        data['pred'] = (data['movement'] > .1).astype(float)
        return data['pred'].to_numpy()

    def evaluate(self, pred, target):
        """
        Evaluation function for the model.
        :param pred: predictions
        :param target: targets
        :return: avg loss of predictions
        """
        pass

    def save(self, path):
        """
        Save function for the model.
        :param path: path to save the model to
        :return:
        """
        pass

    def load(self, path):
        """
        Load function for the model.
        :param path: path to model checkpoint
        :return:
        """
        pass
