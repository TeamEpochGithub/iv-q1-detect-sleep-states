import json

from src.models.event_segmentation_transformer import EventSegmentationTransformer

from ..feature_engineering.feature_engineering import FE
from ..logger.logger import logger
from ..models.event_seg_unet_1d_cnn import EventSegmentationUnet1DCNN
from ..models.spectrogram_2d_cnn import EventSegmentation2DCNN
from ..models.event_res_gru import EventResGRU
from ..models.spectrogram_2d_cnn_gru import EventSegmentation2DCNNGRU
from ..preprocessing.pp import PP
from ..pretrain.pretrain import Pretrain
from ..models.event_model import EventModel
from .. import data_info


class ModelConfigLoader:
    """Class to load the model training configuration from a JSON file"""

    def __init__(self, config_path: str, processed_out: str, processed_in: str, train_series: str, train_events: str, test_series: str, store_location: str = "") -> None:
        """
        Initialize the ModelConfigLoader class
        :param config_path: the path to the config.json file
        """
        self.out_path = processed_out
        self.in_path = processed_in
        self.train_series = train_series
        self.train_events = train_events
        self.test_series = test_series
        self.store_location = store_location

        # Read JSON from file
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Set the global variables
        self.set_globals()

    def get_config(self) -> dict:
        """
        Get the full configuration
        :return: the full configuration dict
        """
        return self.config

    def get_name(self) -> str:
        """
        Get the name of the model
        :return: the name of the model
        """
        return self.config["name"]

    def set_globals(self) -> None:
        """Set the global variables"""
        data_info.window_size = self.config.get("data_info").get(
            "window_size", data_info.window_size)
        data_info.downsampling_factor = self.config.get("data_info").get(
            "downsampling_factor", data_info.downsampling_factor)
        data_info.plot_summary = self.config.get("data_info").get(
            "plot_summary", data_info.plot_summary)
        data_info.latitude = self.config.get(
            "data_info").get("latitude", data_info.latitude)
        data_info.longitude = self.config.get(
            "data_info").get("longitude", data_info.longitude)

    def reset_globals(self) -> None:
        """Reset the global variables to the default values"""
        data_info.window_size_before = self.config.get(
            "data_info").get("window_size", 17280)
        data_info.window_size = data_info.window_size_before
        data_info.downsampling_factor = self.config.get(
            "data_info").get("downsampling_factor", 1)
        data_info.stage = "load_config"
        data_info.substage = "set_globals"
        data_info.plot_summary = self.config.get(
            "data_info").get("plot_summary", False)

        data_info.latitude = self.config.get(
            "data_info").get("latitude", 40.730610)
        data_info.longitude = self.config.get(
            "data_info").get("longitude", -73.935242)

        data_info.X_columns = {}
        data_info.y_columns = {}

        data_info.cv_current_fold = 0
        data_info.cv_unique_series = []

        data_info.X_columns = {}
        data_info.y_columns = {}

        data_info.cv_current_fold = 0

    def get_pp_steps(self, training: bool) -> list[PP]:
        """
        Get the preprocessing steps classes
        :param training: whether the preprocessing is for training or testing
        :return: the preprocessing steps and their names
        """
        return PP.from_config(self.config["preprocessing"], training)

    def get_fe_steps(self) -> list[FE]:
        """
        Get the feature engineering steps classes
        :return: the feature engineering steps and their names
        """
        return FE.from_config(self.config["feature_engineering"])

    def get_pp_fe_pretrain(self) -> str:
        """
        Gets the config of preprocessing, feature engineering and pretraining as a string. This is used to hash in the future.
        :return: the config of preprocessing, feature engineering and pretraining as a string
        """
        return str(self.config['preprocessing']) + str(self.config['feature_engineering']) + str(
            self.config["pretraining"])

    def get_pretraining(self) -> Pretrain:
        """
        Get the pretraining parameters
        :return: the pretraining object
        """
        return Pretrain.from_config(self.config["pretraining"])

    def get_pretrain_config(self) -> dict:
        """Get the data_info, preprocessing, feature_engineering and pretraining from the config

        This is necessary for the creating and retrieving the cached pretrain arrays.

        :return: the slice config with relevant parameters for pretraining
        """
        return {k: self.config[k] for k in ["data_info", "preprocessing", "feature_engineering", "pretraining"]
                if k in self.config}

    def set_model(self) -> EventModel:
        """
        Set the model from the config
        :return: the model
        """
        logger.info("Models: " + str(self.config.get("models")))
        model_name = self.config["name"]
        model_config = self.config["architecture"]
        match model_config["type"]:
            case "event-segmentation-transformer":
                curr_model = EventSegmentationTransformer(
                    model_config, model_name)
            case "event-res-gru":
                curr_model = EventResGRU(model_config, model_name)
            case "event-seg-unet-1d-cnn":
                curr_model = EventSegmentationUnet1DCNN(
                    model_config, model_name)
            case "Spectrogram_2D_Cnn":
                curr_model = EventSegmentation2DCNN(model_config, model_name)
            case "Spectrogram_Cnn_Gru":
                curr_model = EventSegmentation2DCNNGRU(model_config, model_name)
            case _:
                logger.critical("Model not found: " + model_config["type"])
                raise ConfigException(
                    "Model not found: " + model_config["type"])

        self.model = curr_model
        return curr_model

    def get_model(self) -> EventModel:
        """
        Get the model from the config
        :return: the model
        """
        return self.model

    def get_processed_out(self) -> str:
        """
        Get the path to the processed data output
        :return: the path to the processed data output
        """
        return self.out_path

    def get_processed_in(self) -> str:
        """
        Get the path to the processed data input
        :return: the path to the processed data input
        """
        return self.in_path

    def get_train_series_path(self) -> str:
        """
        Get the path to the training series
        :return: the path to the training series
        """
        return self.train_series

    def get_train_events_path(self) -> str:
        """
        Get the path to the training labels
        :return: the path to the training labels
        """
        return self.train_events

    def get_test_series_path(self) -> str:
        """
        Get the path to the test series
        :return: the path to the test series
        """
        return self.test_series

    def get_store_location(self) -> str:
        """
        Get the path to the model store directory
        :return: the path to the model store directory
        """
        return self.store_location


# ConfigException class
class ConfigException(Exception):
    """
    Exception class for configuration.
    Raises an exception when the config.json is not correct.
    """
    pass
