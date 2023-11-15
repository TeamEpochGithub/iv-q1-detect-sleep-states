# Create a class for ensemble learning

# Imports
import os
import numpy as np
from tqdm import tqdm
from src.get_processed_data import get_processed_data
from src.pretrain.pretrain import Pretrain
from src.util.get_pretrain_cache import get_pretrain_split_cache

from src.util.printing_utils import print_section_separator
from src.util.hash_config import hash_config
from src.util.state_to_event import pred_to_event_state

from ..logger.logger import logger
from typing import Any, List
from ..configs.load_model_config import ModelConfigLoader
from .. import data_info


class Ensemble:

    # Init function
    def __init__(self, model_configs: List[ModelConfigLoader] = None, weight_matrix: List[int] = None, combination_method: str = "addition", pred_only: bool = False) -> None:
        self.pred_only = pred_only
        self.combination_method = combination_method

        if model_configs is None:
            self.model_configs = []
        else:
            self.model_configs = model_configs

        if weight_matrix is None:
            weight_matrix = np.ones(len(self.model_configs))

        # Apply softmax to weight matrix
        self.weight_matrix = np.exp(
            weight_matrix) / np.sum(np.exp(weight_matrix))

        if len(self.weight_matrix) != len(self.model_configs):
            logger.critical(
                "Weight matrix length does not match number of models")
            raise ValueError(
                "Weight matrix length does not match number of models")
        elif np.sum(self.weight_matrix) != 1:
            logger.critical("Weight matrix must sum to 1")
            raise ValueError("Weight matrix must sum to 1")
        elif np.any(self.weight_matrix) <= 0:
            logger.critical("Weight matrix must be positive")
            raise ValueError("Weight matrix must be positive")

    def get_models(self):
        """
        Get the models from the ensemble
        :return: the models
        """
        return self.model_configs

    def get_pred_only(self):
        """
        Get whether to only predict with the ensemble
        :return: whether to only predict with the ensemble
        """
        return self.pred_only

    def get_test_idx(self) -> np.ndarray:
        """
        Get the test indices from the ensemble
        :return: the test indices
        """
        return self.test_idx

    def pred(self, config_loader, pred_with_cpu: bool, training: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Prediction function for the ensemble.
        Feeds the models data window-by-window, averages their predictions
        and converts the window-relative steps to absolute steps since the start of the series

        :param data: 3D tensor with shape (window, n_timesteps, n_features)
        :param pred_with_cpu: whether to use the cpu for prediction
        :return: 3D array with shape (window, 2), with onset and wakeup steps (nan if no detection)
        """

        # Initialize models
        store_location = config_loader.get_model_store_loc()
        logger.info("Model store location: " + store_location)
        logger.info("Predicting with ensemble")

        # Run each model
        predictions = None
        confidences = []
        # model_pred is (onset, wakeup) tuples for each window
        for _, model_config in enumerate(self.model_configs):

            config_loader.reset_globals()
            model_pred = self.pred_model(
                model_config_loader=model_config, store_location=store_location, pred_with_cpu=pred_with_cpu, training=training)

            # Model_pred is tuple of np.array(onset, awake) for each window
            # Split the series of tuples into two column
            if predictions is not None:
                predictions = np.concatenate((predictions, (model_pred.reshape(
                    model_pred.shape[0], model_pred.shape[1], 2, 1))), axis=3)
            else:
                predictions = model_pred.reshape(
                    model_pred.shape[0], model_pred.shape[1], 2, 1)

        if self.combination_method == "confidence_average":
            predictions = np.average(predictions, axis=3)
            all_predictions = []
            all_confidences = []
            for pred in tqdm(predictions, desc="Converting predictions to events", unit="window"):
                # Convert to relative window event timestamps
                events = pred_to_event_state(pred, thresh=0)

                # Add step offset based on repeat factor.
                if data_info.downsampling_factor <= 1:
                    offset = 0
                elif data_info.downsampling_factor % 2 == 0:
                    offset = (data_info.downsampling_factor / 2.0) - 0.5
                else:
                    offset = data_info.downsampling_factor // 2
                steps = (events[0] + offset, events[1] + offset)
                confidences = (events[2], events[3])
                all_predictions.append(steps)
                all_confidences.append(confidences)

            # Return numpy array
            return np.array(all_predictions), np.array(all_confidences)

        # TODO: consider how to combine non-Nan and NaNs in the predictions #146

        # Weight the predictions
        predictions = np.array(predictions)
        aggregate = np.average(
            predictions, axis=0, weights=self.weight_matrix)

        # Weight the confidences
        confidences = np.array(confidences)
        aggregate_confidences = np.average(
            confidences, axis=0, weights=self.weight_matrix)

        return aggregate, aggregate_confidences

    def pred_model(
        self,
        model_config_loader: ModelConfigLoader,
        store_location: str,
        pred_with_cpu: bool = True,
        training: bool = False
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:

        model_name = model_config_loader.get_name()

        # ------------------------------------------- #
        #    Preprocessing and feature Engineering    #
        # ------------------------------------------- #

        print_section_separator(
            "Preprocessing and feature engineering", spacing=0)
        data_info.stage = "preprocessing & feature engineering"
        featured_data = get_processed_data(
            model_config_loader, training=training, save_output=True)

        # ------------------------ #
        #         Pretrain         #
        # ------------------------ #

        print_section_separator("Pretrain", spacing=0)
        data_info.stage = "pretraining"

        logger.info(
            "Get pretraining parameters from config and initialize pretrain")
        pretrain: Pretrain = model_config_loader.get_pretraining()

        logger.info("Pretraining with scaler " + str(pretrain.scaler.kind) +
                    " and test size of " + str(pretrain.test_size))

        # Split data into train/test and validation
        # Use numpy.reshape to turn the data into a 3D tensor with shape (window, n_timesteps, n_features)
        logger.info("Splitting data into train and test...")
        data_info.substage = "pretrain_split"

        if training:
            x_train, x_test, y_train, y_test, train_idx, test_idx, groups = get_pretrain_split_cache(
                model_config_loader, featured_data, save_output=True)
            self.test_idx = test_idx

            logger.info("X Train data shape (size, window_size, features): " + str(
                x_train.shape) + " and y Train data shape (size, window_size, features): " + str(y_train.shape))
            logger.info("X Test data shape (size, window_size, features): " + str(
                x_test.shape) + " and y Test data shape (size, window_size, features): " + str(y_test.shape))
        else:
            # Hash of concatenated string of preprocessing, feature engineering and pretraining
            scaler_hash = hash_config(
                model_config_loader.get_pretrain_config(), length=5)
            if pretrain.scaler.scaler:
                pretrain.scaler.load(
                    store_location + "/scaler-" + scaler_hash + ".pkl")

            x_test = pretrain.preprocess(featured_data)

        logger.info("Creating model using ModelConfigLoader")
        model = model_config_loader.set_model()

        # Hash of concatenated string of preprocessing, feature engineering and pretraining
        initial_hash = hash_config(
            model_config_loader.get_pp_fe_pretrain(), length=5)
        data_info.substage = f"training model: {model_name}"

        # Get filename of model
        model_filename_opt = store_location + "/optimal_" + \
            model_name + "-" + initial_hash + model.hash + ".pt"

        # If this file exists, load instead of start training
        if os.path.isfile(model_filename_opt):
            logger.info("Found existing trained optimal model: " +
                        model_name + " with location " + model_filename_opt)
            model.load(model_filename_opt, only_hyperparameters=False)
        else:
            logger.critical("Not all models have been trained yet")
            raise ValueError(
                f"Not all models have been trained yet, missing {model_filename_opt}")

        model = model_config_loader.get_model()
        # If the model has the device attribute, it is a pytorch model and we want to pass the pred_cpu argument.
        if hasattr(model, 'device'):
            model_pred = model.pred(
                x_test, pred_with_cpu=pred_with_cpu, raw_output=True)
        else:
            model_pred = model.pred(x_test, raw_output=True)

        return model_pred
