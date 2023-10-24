import json

import pandas as pd
from torch import nn

from ..cv.cv import CV
from ..ensemble.ensemble import Ensemble
from ..feature_engineering.feature_engineering import FE
from ..feature_engineering.kurtosis import Kurtosis
from ..feature_engineering.mean import Mean
from ..feature_engineering.rotation import Rotation
from ..feature_engineering.skewness import Skewness
from ..feature_engineering.time import Time
from ..hpo.hpo import HPO
from ..logger.logger import logger
from ..loss.loss import Loss
from ..models.classic_base_model import ClassicBaseModel
from ..models.example_model import ExampleModel
from ..models.seg_simple_1d_cnn import SegmentationSimple1DCNN
from ..models.seg_unet_1d_cnn import SegmentationUnet1DCNN
from ..models.transformers.event_nan_regression_transformer import EventNaNRegressionTransformer
from ..models.transformers.event_regression_transformer import EventRegressionTransformer
from ..preprocessing.pp import PP
from ..pretrain.pretrain import Pretrain


class ConfigLoader:
    """Class to load the configuration from a JSON file"""

    def __init__(self, config_path: str) -> None:
        """Initialize the ConfigLoader class

        :param config_path: the path to the config.json file
        """
        self.config_path = config_path

        # Read JSON from file
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    # Get full configuration
    def get_config(self) -> dict:
        """Get the full configuration

        :return: the full configuration dict
        """
        return self.config

    def get_log_to_wandb(self) -> bool:
        """Get whether to log to Weights & Biases

        :return: whether to log to Weights & Biases
        """
        return self.config["log_to_wandb"]

    def get_train_series_path(self) -> str:
        """Get the path to the training series data

        :return: the path to the train_series.parquet file
        """
        return self.config["train_series_path"]

    def get_train_events_path(self) -> str:
        """Get the path to the training labels data

        :return: the path to the train_events.csv file
        """
        return self.config["train_events_path"]

    def get_test_series_path(self) -> str:
        """Get the path to the test series data

        :return: the path to the test_series.parquet file
        """
        return self.config["test_series_path"]

    def get_pp_steps(self, training: bool) -> list[PP]:
        """Get the preprocessing steps classes

        :param training: whether the preprocessing is for training or testing
        :return: the preprocessing steps and their names
        """
        return PP.from_config(self.config["preprocessing"], training)

    def get_pp_out(self) -> str:
        """Get the path to the preprocessing output folder

        :return: the path to the preprocessing output folder
        """
        return self.config["processed_loc_out"]

    def get_pred_with_cpu(self) -> bool:
        """Get whether to use CPU for prediction

        :return: whether to use CPU for prediction
        """
        return self.config["pred_with_cpu"]

    def get_pp_in(self) -> str:
        """Get the path to the preprocessing input data folder

        :return: the path to the preprocessing input data folder
        """
        return self.config["processed_loc_in"]

    def get_features(self) -> (dict[FE], list[str]):
        """Get the feature engineering steps classes

        :return: the feature engineering steps and their names
        """
        fe_steps: dict = {}
        fe_s: list = []
        for fe_step in self.config["feature_engineering"]:
            if fe_step == "kurtosis":
                fe_steps["kurtosis"] = Kurtosis(
                    self.config["feature_engineering"]["kurtosis"])
                window_sizes = self.config["feature_engineering"]["kurtosis"]["window_sizes"]
                window_sizes.sort()
                window_sizes = str(window_sizes).replace(" ", "")
                features = self.config["feature_engineering"]["kurtosis"]["features"]
                features.sort()
                features = str(features).replace(" ", "")
                fe_s.append(fe_step + features + window_sizes)
            elif fe_step == "skewness":
                fe_steps["skewness"] = Skewness(
                    self.config["feature_engineering"]["skewness"])
                window_sizes = self.config["feature_engineering"]["skewness"]["window_sizes"]
                window_sizes.sort()
                window_sizes = str(window_sizes).replace(" ", "")
                features = self.config["feature_engineering"]["skewness"]["features"]
                features.sort()
                features = str(features).replace(" ", "")
                fe_s.append(fe_step + features + window_sizes)
            elif fe_step == "mean":
                fe_steps["mean"] = Mean(
                    self.config["feature_engineering"]["mean"])
                window_sizes = self.config["feature_engineering"]["mean"]["window_sizes"]
                window_sizes.sort()
                window_sizes = str(window_sizes).replace(" ", "")
                features = self.config["feature_engineering"]["mean"]["features"]
                features.sort()
                features = str(features).replace(" ", "")
                fe_s.append(fe_step + features + window_sizes)
            elif fe_step == "time":
                fe_steps["time"] = Time(
                    self.config["feature_engineering"]["time"])
                fe_s.append(fe_step)
            elif fe_step == "rotation":
                fe_steps["rotation"] = Rotation(
                    self.config["feature_engineering"]["rotation"])
                fe_s.append(fe_step)
            else:
                logger.critical("Feature engineering step not found: " + fe_step)
                raise ConfigException(
                    "Feature engineering step not found: " + fe_step)

        return fe_steps, fe_s

    def get_pp_fe_pretrain(self) -> str:
        """Gets the config of preprocessing, feature engineering and pretraining as a string. This is used to hash in the future.
        :return: the config of preprocessing, feature engineering and pretraining as a string
        """
        return str(self.config['preprocessing']) + str(self.config['feature_engineering']) + str(
            self.config["pretraining"])

    def get_fe_out(self) -> str:
        """Get the path to the feature engineering output folder

        :return: the path to the feature engineering output folder
        """
        return self.config["fe_loc_out"]

    def get_fe_in(self) -> str:
        """Get the path to the feature engineering input data folder

        :return: the path to the feature engineering input data folder
        """
        return self.config["fe_loc_in"]

    def get_pretraining(self) -> Pretrain:
        """Get the pretraining parameters

        :return: the pretraining object
        """
        return Pretrain.from_config(self.config["pretraining"])

    # Function to retrieve model data
    def get_models(self, data_shape: tuple) -> dict:
        """Get the models from the config

        :param data_shape: the shape of the data
        :return: the models
        """
        models: dict = {}
        # Loop through models
        logger.info("Models: " + str(self.config.get("models")))
        for model_name in self.config["models"]:
            model_config = self.config["models"][model_name]
            curr_model = None
            match model_config["type"]:
                case "example-fc-model":
                    curr_model = ExampleModel(model_config, model_name)
                case "classic-base-model":
                    curr_model = ClassicBaseModel(model_config, model_name)
                case "seg-simple-1d-cnn":
                    curr_model = SegmentationSimple1DCNN(model_config, data_shape, model_name)
                case "seg-unet-1d-cnn":
                    curr_model = SegmentationUnet1DCNN(model_config, data_shape, model_name)
                case "event-nan-regression-transformer":
                    curr_model = EventNaNRegressionTransformer(model_config, model_name)
                case "event-regression-transformer":
                    curr_model = EventRegressionTransformer(model_config, data_shape, model_name)
                case _:
                    logger.critical("Model not found: " + model_config["type"])
                    raise ConfigException("Model not found: " + model_config["type"])

            models[model_name] = curr_model

        return models

    def get_model_store_loc(self) -> str:
        """Get the path to the model store directory

        :return: the path to the model store directory
        """
        return self.config["model_store_loc"]

    def get_ensemble(self, models: dict) -> Ensemble:
        """Get the ensemble from the config

        :param models: the models
        :return: the ensemble
        """

        curr_models: list = []
        # If length of weights and models is not equal, raise exception
        if len(self.config["ensemble"]["weights"]) != len(self.config["ensemble"]["models"]):
            logger.critical("Length of weights and models is not equal")
            raise ConfigException("Length of weights and models is not equal")

        if len(models) < len(self.config["ensemble"]["models"]):
            logger.critical("You cannot have more ensembles than models.")
            raise ConfigException("You cannot have more ensembles than models.")

        for model_name in self.config["ensemble"]["models"]:
            if model_name not in models:
                logger.critical(f"Model {model_name} not found in models.")
                raise ConfigException(f"Model {model_name} not found in models.")
            curr_models.append(models[model_name])

        # Create ensemble
        ensemble = Ensemble(
            curr_models, self.config["ensemble"]["weights"], self.config["ensemble"]["comb_method"], self.config["ensemble"])

        return ensemble

    def get_ensemble_loss(self) -> nn.Module:
        """Get the ensemble loss function from the config

        :return: the loss function
        """
        loss_class = None
        if self.config["ensemble_loss"] == "example_loss":
            loss_class = Loss().get_loss("example_loss")
        else:
            logger.critical("Loss function not found: " + self.config["loss"])
            raise ConfigException("Loss function not found: " + self.config["loss"])

        return loss_class

    def get_hpo(self) -> HPO:
        """Get the hyperparameter tuning method from the config

        :return: the hyperparameter tuning method
        """
        hpo_class = None
        if self.config["hpo"]["method"] == "example_hpo":
            hpo_class = HPO(None, None, None)
        else:
            raise ConfigException("Hyperparameter tuning method not found: " +
                                  self.config["hpo"]["method"])

        return hpo_class

    def get_cv(self, data: pd.DataFrame, labels: pd.Series) -> CV:
        """Get the cross validation method from the config

        :return: the cross validation method
        """
        return CV(data, labels, **self.config["cv"])

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


# ConfigException class
class ConfigException(Exception):
    """
    Exception class for configuration.
    Raises an exception when the config.json is not correct.
    """
    pass
