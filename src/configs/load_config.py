import json

from src.cv.cv import CV
from .. import data_info
from ..ensemble.ensemble import Ensemble
from .load_model_config import ModelConfigLoader
from ..logger.logger import logger


class ConfigLoader:
    """Class to load the configuration from a JSON file"""

    def __init__(self, config_path: str) -> None:
        """
        Initialize the ConfigLoader class
        :param config_path: the path to the config.json file
        """
        self.config_path = config_path

        # Read JSON from file
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Can't have ensemble and hpo in one config
        if "ensemble" in self.config and "hpo" in self.config:
            logger.critical(
                "Config cannot have both ensemble and hpo")
            raise ConfigException(
                "Config cannot have both ensemble and hpo")

        self.set_globals()
        if self.config.get("ensemble"):
            self.set_ensemble()
        if self.config.get("hpo"):
            self.set_hpo_config()
        self.name = self.config["name"]

    def get_config(self) -> dict:
        """
        Get the full configuration
        :return: the full configuration dict
        """
        return self.config

    def get_name(self) -> str:
        """
        Get the name of the config
        :return: the name of the config
        """

    def get_log_to_wandb(self) -> bool:
        """
        Get whether to log to Weights & Biases
        :return: whether to log to Weights & Biases
        """
        return self.config["log_to_wandb"]

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
        data_info.pred_with_cpu = self.get_pred_with_cpu()
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

    def get_train_series_path(self) -> str:
        """Get the path to the training series data

        :return: the path to the train_series.parquet file
        """
        return self.config["train_series_path"]

    def get_train_events_path(self) -> str:
        """
        Get the path to the training labels data
        :return: the path to the train_events.csv file
        """
        return self.config["train_events_path"]

    def get_test_series_path(self) -> str:
        """Get the path to the test series data

        :return: the path to the test_series.parquet file
        """
        return self.config["test_series_path"]

    def get_pred_with_cpu(self) -> bool:
        """Get whether to use CPU for prediction

        :return: whether to use CPU for prediction
        """
        return self.config["pred_with_cpu"]

    def get_processed_in(self) -> str:
        """Get the path to the preprocessing input data folder

        :return: the path to the preprocessing input data folder
        """
        return self.config["processed_loc_in"]

    def get_processed_out(self) -> str:
        """Get the path to the preprocessing output folder

        :return: the path to the preprocessing output folder
        """
        return self.config["processed_loc_out"]

    def get_model_store_loc(self) -> str:
        """Get the path to the model store directory

        :return: the path to the model store directory
        """
        return self.config["model_store_loc"]

    def get_cv(self) -> CV:
        """
        Get the cross validation method
        :return: the cross validation method
        """
        return CV(**self.config["cv"])

    def set_ensemble(self) -> None:
        """
        Reads each model config file and gets the ensemble from the config
        :return: the ensemble
        """
        # Get all models
        curr_models = []
        for model_file_name in self.config["ensemble"]["models"]:
            curr_path = self.config["model_config_loc"] + "/" + model_file_name
            curr_models.append(ModelConfigLoader(
                config_path=curr_path,
                processed_out=self.get_processed_out(),
                processed_in=self.get_processed_in(),
                train_series=self.get_train_series_path(),
                train_events=self.get_train_events_path(),
                test_series=self.get_test_series_path(),
                store_location=self.get_model_store_loc()))

        # Create ensemble
        ensemble = Ensemble(
            curr_models, self.config["ensemble"]["weights"], self.config["ensemble"]["comb_method"], self.config["ensemble"]["pred_only"])

        self.ensemble = ensemble

    def get_ensemble(self) -> Ensemble:
        """
        Get the ensemble from the config
        :return: the ensemble
        """
        if not hasattr(self, "ensemble"):
            return None
        return self.ensemble

    def get_train_optimal(self) -> bool:
        """
        Train optimal model from the config
        :return: whether to train optimal model
        """
        return self.config["train_optimal"]

    def cv(self) -> CV:
        """Get the cross validation method from the config

        :return: the cross validation method
        """
        if "cv" in self.config:
            return CV(**self.config["cv"])
        return None

    def get_hpo(self) -> bool:
        """
        Get the hyperparameter optimization parameters from the config
        :return: the hyperparameter optimization parameters
        """
        data_info.hpo = self.config.get("hpo")
        return self.config.get("hpo")

    def set_hpo_config(self) -> dict:
        """
        Set the hyperparameter optimization parameters from the config
        :return: the hyperparameter optimization parameters
        """

        curr_path = self.config["model_config_loc"] + "/" + self.config["hpo"]
        self.hpo_config = ModelConfigLoader(
            config_path=curr_path,
            processed_out=self.get_processed_out(),
            processed_in=self.get_processed_in(),
            train_series=self.get_train_series_path(),
            train_events=self.get_train_events_path(),
            test_series=self.get_test_series_path(),
            store_location=self.get_model_store_loc())

    def get_hpo_config(self) -> dict:
        """
        Get the hyperparameter optimization parameters from the config
        :return: the hyperparameter optimization parameters
        """
        return self.hpo_config

    # Function to retrieve train for submission
    def get_train_for_submission(self) -> bool:
        """Get whether to train for submission from the config

        :return: whether to train for submission
        """
        return self.config["train_for_submission"]

    # Function to retrieve scoring
    def get_scoring(self) -> bool:
        """Get whether to score from the config

        :return: whether to score
        """
        return self.config["scoring"]

    def get_browser_plot(self) -> bool:
        """Get whether to visualize from the config

        :return: whether to visualize
        """
        return self.config["visualize_preds"]["browser_plot"]

    def get_number_of_plots(self) -> int:
        """Get the number of plots from the config

        :return: the number of plots
        """
        return self.config["visualize_preds"]["n"]

    def get_store_plots(self) -> bool:
        """Get whether to store plots from the config

        :return: whether to store plots
        """
        return self.config["visualize_preds"]["save"]

    def get_similarity_filter(self) -> dict | None:
        """Get the similarity filter from the config

        :return: the similarity filter
        """
        return self.config.get("similarity_filter")


# ConfigException class
class ConfigException(Exception):
    """
    Exception class for configuration.
    Raises an exception when the config.json is not correct.
    """
    pass
