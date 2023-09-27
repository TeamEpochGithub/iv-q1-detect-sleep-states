# In this file the correct classes are retrieved for the configuration
import json

# Preprocessing imports
from ..preprocessing.mem_reduce import MemReduce
from ..preprocessing.add_noise import AddNoise
from ..preprocessing.split_windows import SplitWindows
from ..preprocessing.covert_datetime import ConvertDatetime
# Feature engineering imports
from ..feature_engineering.cumsum_accel import cumsum_accel

# Feature engineering imports
from ..feature_engineering.example_feature_engineering import ExampleFeatureEngineering

# Model imports
from ..models.example_model import ExampleModel

# Ensemble imports
from ..ensemble.ensemble import Ensemble

# Loss imports
from ..loss.loss import Loss

# HPO imports
from ..hpo.hpo import HPO


# CV imports
from ..cv.cv import CV


class ConfigLoader:

    # Initiate class using config path
    def __init__(self, config_path):
        self.config_path = config_path

        # Read JSON from file
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def get_config(self):
        return self.config

    # Function to retrieve preprocessing steps
    def get_pp_steps(self):
        self.pp_steps = []
        for pp_step in self.config["preprocessing"]:
            match pp_step:
                case "mem_reduce":
                    self.pp_steps.append(MemReduce())
                case "add_noise":
                    self.pp_steps.append(AddNoise())
                case "split_windows":
                    self.pp_steps.append(SplitWindows())
                case "convert_datetime":
                    self.pp_steps.append(ConvertDatetime())
                case _:
                    raise ConfigException("Preprocessing step not found: " + pp_step)
        return self.pp_steps, self.config["preprocessing"]

    # Function to retrieve preprocessing data location out path
    def get_pp_out(self):
        return self.config["processed_loc_out"]

    # Function to retrieve preprocessing data location in path
    def get_pp_in(self):
        return self.config["processed_loc_in"]

    # Function to retrieve feature engineering classes from feature engineering folder
    def get_features(self):
        self.fe_steps = {}
        for fe_step in self.config["feature_engineering"]:
            if fe_step == "example_feature_engineering":
                self.fe_steps["example_feature_engineering"] = ExampleFeatureEngineering(
                )
            elif fe_step == "cumsum_accel":
                self.fe_steps["cumsum_accel"] = cumsum_accel()
            else:
                raise ConfigException(
                    "Feature engineering step not found: " + fe_step)

        return self.fe_steps, self.config["feature_engineering"]

    # Function to retrieve feature engineering data location out path
    def get_fe_out(self):
        return self.config["fe_loc_out"]

    # Function to retrieve feature engineering data location in path
    def get_fe_in(self):
        return self.config["fe_loc_in"]

    # Function to retrieve model data
    def get_models(self):
        # Loop through models
        self.models = {}
        for model in self.config["models"]:
            model_config = self.config["models"][model]
            curr_model = None
            if model_config["type"] == "example_model":
                curr_model = ExampleModel(model_config)
            else:
                raise ConfigException(
                    "Model not found: " + model_config["type"])

            self.models[model] = curr_model

        return self.models

    # Function to retrieve ensemble data
    def get_ensemble(self):
        # If length of weights and models is not equal, raise exception
        if len(self.config["ensemble"]["weights"]) != len(self.config["ensemble"]["models"]):
            raise ConfigException(
                "Length of weights and models is not equal")

        # Create ensemble
        ensemble = Ensemble(
            [self.models[x] for x in self.config["ensemble"]["models"]],
            self.config["ensemble"]["weights"],
            self.config["ensemble"]["comb_method"])

        return ensemble

    # Function to retrieve loss function
    def get_loss(self):
        loss_class = None
        if self.config["loss"] == "example_loss":
            loss_class = Loss()
        else:
            raise ConfigException("Loss function not found: " +
                                  self.config["loss"])

        return loss_class

    # Function to retrieve hyperparameter tuning method
    def get_hpo(self):
        hpo_class = None
        if self.config["hpo"]["method"] == "example_hpo":
            hpo_class = HPO(None, None, None)
        else:
            raise ConfigException("Hyperparameter tuning method not found: " +
                                  self.config["hpo"]["method"])

        return hpo_class

    # Function to retrieve cross validation method
    def get_cv(self):
        cv_class = None
        if self.config["cv"]["method"] == "example_cv":
            cv_class = CV()
        else:
            raise ConfigException("Cross validation method not found: " +
                                  self.config["cv"]["method"])

        return cv_class

    # Function to retrieve scoring
    def get_scoring(self):
        return self.config["scoring"]


# ConfigException class
class ConfigException(Exception):
    pass
