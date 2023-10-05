import numpy as np
import pandas as pd

from ..models.model import Model
from ..util.state_to_event import find_events


class ClassicBaseModel(Model):
    """
    This is a sample model file. You can use this as a template for your own models.
    The model file should contain a class that inherits from the Model class.
    """

    def __init__(self, config: dict, name: str) -> None:
        """
        Init function of the example model
        :param config: configuration to set up the model
        :param name: name of the model
        """
        super().__init__(config)
        self.model_type = "classic-base-model"
        self.load_config(config)

    def load_config(self, config: dict) -> None:
        """
        Load config function for the model.
        :param config: configuration to set up the model
        """

        # Get default_config
        default_config = self.get_default_config()

        config["median_window"] = config.get("median_window", default_config["median_window"])
        config["threshold"] = config.get("threshold", default_config["threshold"])
        self.config = config

    def get_default_config(self) -> dict:
        """
        Get default config function for the model.
        :return: default config
        """
        return {"median_window": 100, "threshold": .1}

    def pred(self, X_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Prediction function for the model.
        :param X_pred: unlabeled data for a single day window as pandas dataframe
        :return: two timestamps, or NaN if no sleep was detected
        """
        # Get the data from the data tuple
        state_pred = self.predict_state_labels(X_pred)
        onset, awake = find_events(state_pred)
        return onset, awake

    def predict_state_labels(self, data: pd.DataFrame) -> np.ndarray:
        data['slope'] = abs(data['anglez'].diff()).clip(upper=10)
        data['movement'] = data['slope'].rolling(window=100).median()
        data['pred'] = (data['movement'] > .1).astype(float)
        return data['pred'].to_numpy()
