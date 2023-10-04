# In this file the correct classes are retrieved for the configuration
import json

# CV imports
from ..cv.cv import CV
# Ensemble imports
from ..ensemble.ensemble import Ensemble
# Feature engineering imports
from ..feature_engineering.kurtosis import Kurtosis
from ..feature_engineering.mean import Mean
from ..feature_engineering.skewness import Skewness
# HPO imports
from ..hpo.hpo import HPO
from ..logger.logger import logger
# Loss imports
from ..loss.loss import Loss
from ..models.classic_base_model import ClassicBaseModel
# Model imports
from ..models.seg_simple_1d_cnn import SegmentationSimple1DCNN
from ..preprocessing.add_event_labels import AddEventLabels
from ..preprocessing.add_noise import AddNoise
from ..preprocessing.add_state_labels import AddStateLabels
# Preprocessing imports
from ..preprocessing.mem_reduce import MemReduce
from ..preprocessing.remove_unlabeled import RemoveUnlabeled
from ..preprocessing.split_windows import SplitWindows
from ..preprocessing.truncate import Truncate


class ConfigLoader:

    # Initiate class using config path
    def __init__(self, config_path):
        self.pp_steps = []
        self.config_path = config_path

        # Read JSON from file
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    # Get full configuration
    def get_config(self):
        return self.config

    # Get boolean for whether to use wandb
    def get_log_to_wandb(self):
        return self.config["log_to_wandb"]

    # Function to retrieve preprocessing steps
    def get_pp_steps(self, training=True):
        self.pp_steps = []
        for pp_step in self.config["preprocessing"]:
            match pp_step:
                case "mem_reduce":
                    self.pp_steps.append(MemReduce())
                case "add_noise":
                    self.pp_steps.append(AddNoise())
                case "split_windows":
                    self.pp_steps.append(SplitWindows())
                case "add_state_labels":
                    if training:
                        self.pp_steps.append(AddStateLabels())
                case "remove_unlabeled":
                    if training:
                        self.pp_steps.append(RemoveUnlabeled())
                case "truncate":
                    if training:
                        self.pp_steps.append(Truncate())
                case "add_event_labels":
                    if training:
                        self.pp_steps.append(AddEventLabels())
                case _:
                    logger.critical("Preprocessing step not found: " + pp_step)
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
        fe_steps = {}
        fe_s = []
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
            else:
                logger.critical("Feature engineering step not found: " + fe_step)
                raise ConfigException(
                    "Feature engineering step not found: " + fe_step)

        return fe_steps, fe_s

    # Function to retrieve feature engineering data location out path
    def get_fe_out(self):
        return self.config["fe_loc_out"]

    # Function to retrieve feature engineering data location in path
    def get_fe_in(self):
        return self.config["fe_loc_in"]

    # Function to retrieve pretraining data
    def get_pretraining(self):
        return self.config["pre_training"]

    # Function to retrieve model data
    def get_models(self, data_shape):
        # Loop through models
        self.models = {}
        logger.info("Models: " + str(self.config.get("models")))
        for model in self.config["models"]:
            model_config = self.config["models"][model]
            curr_model = None
            match model_config["type"]:
                case "example-fc-model":
                    curr_model = ExampleModel(model_config)
                case "classic-base-model":
                    curr_model = ClassicBaseModel(model_config)
                case "seg-simple-1d-cnn":
                    curr_model = SegmentationSimple1DCNN(model_config, data_shape)
                case _:
                    logger.critical("Model not found: " + model_config["type"])
                    raise ConfigException("Model not found: " + model_config["type"])
            self.models[model] = curr_model

        return self.models

    # Getter for model store location
    def get_model_store_loc(self):
        return self.config["model_store_loc"]

    # Function to retrieve ensemble data
    def get_ensemble(self, models):

        curr_models = []
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
            curr_models, self.config["ensemble"]["weights"], self.config["ensemble"]["comb_method"])

        return ensemble

    # Function to retrieve loss function

    def get_ensemble_loss(self):
        loss_class = None
        if self.config["ensemble_loss"] == "example_loss":
            loss_class = Loss().get_loss("example_loss")
        else:
            logger.critical("Loss function not found: " + self.config["loss"])
            raise ConfigException("Loss function not found: " + self.config["loss"])

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

    # Function to retrieve train for submission
    def get_train_for_submission(self):
        return self.config["train_for_submission"]

    # Function to retrieve scoring
    def get_scoring(self):
        return self.config["scoring"]


# ConfigException class
class ConfigException(Exception):
    pass
