import json

from .. import data_info
from ..ensemble.ensemble import Ensemble
from ..hpo.hpo import HPO
from .load_model_config import ModelConfigLoader


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

        self.set_globals()
        self.set_ensemble()

    # Get full configuration
    def get_config(self) -> dict:
        """
        Get the full configuration
        :return: the full configuration dict
        """
        return self.config

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

    def get_train_series_path(self) -> str:
        """Get the path to the training series data

        :return: the path to the train_series.parquet file
        """
        return self.config["train_series_path"]

    def get_train_labels_path(self) -> str:
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
                train_labels=self.get_train_labels_path(),
                test_series=self.get_test_series_path()))

        # Create ensemble
        ensemble = Ensemble(
            curr_models, self.config["ensemble"]["weights"], self.config["ensemble"]["comb_method"])

        self.ensemble = ensemble

    def get_ensemble(self) -> Ensemble:
        """
        Get the ensemble from the config
        :return: the ensemble
        """
        return self.ensemble

    def get_hpo(self) -> HPO:
        """
        Get the hyperparameter tuning method from the config
        :return: the hyperparameter tuning method
        """
        hpo_class = None
        if self.config["hpo"]["method"] == "example_hpo":
            hpo_class = HPO(None, None, None)
        else:
            raise ConfigException("Hyperparameter tuning method not found: " +
                                  self.config["hpo"]["method"])

        return hpo_class

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
