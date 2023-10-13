import copy
import numpy as np
import torch
import wandb

from src.logger.logger import logger
from src.models.transformers.trainers.base_trainer import Trainer

from ...loss.loss import Loss
from ..model import Model, ModelException
from ...optimizer.optimizer import Optimizer
from .architecture.transformer_encoder import TSTransformerEncoderClassiregressor
from ...util.patching import patch_x_data, patch_y_data  # , unpatch_data
from typing import List
from torch import nn
from tqdm import tqdm
from numpy import ndarray, dtype
from typing import Any


class EventNaNRegressionTransformer(Model):
    """
    This is the model file for the stacked transformer model.
    """

    def __init__(self, config: dict, name: str):
        """
        Init function of the example model
        :param config: configuration to set up the model
        """
        super().__init__(config, name)
        # Init model
        self.name = name
        self.events_transformer_config = self.load_transformer_config(
            config).copy()
        self.nans_transformer_config = self.load_transformer_config(
            config).copy()
        self.nans_transformer_config["act_out"] = "sigmoid"
        self.model_events = TSTransformerEncoderClassiregressor(
            **self.events_transformer_config)
        self.model_nans = TSTransformerEncoderClassiregressor(
            **self.nans_transformer_config)
        self.load_config(config)

        # Check if gpu is available, else return an exception
        if not torch.cuda.is_available():
            raise ModelException("GPU not available")

        print("GPU Found: " + torch.cuda.get_device_name(0))
        self.device = torch.device("cuda")

    def load_config(self, config):
        """
        Load config function for the model.
        :param config: configuration to set up the model
        :return:
        """
        # Error checks. Check if all necessary parameters are in the config.
        required = ["loss_events", "loss_nans", "epochs_events",
                    "epochs_nans", "optimizer_events", "optimizer_nans"]
        for req in required:
            if req not in config:
                logger.critical(
                    "------ Config is missing required parameter: " + req)
                raise ModelException(
                    "Config is missing required parameter: " + req)

        # Get default_config
        default_config = self.get_default_config()
        config = copy.deepcopy(config)
        config["loss_events"] = Loss.get_loss(config["loss_events"])
        config["loss_nans"] = Loss.get_loss(config["loss_nans"])
        config["batch_size"] = config.get(
            "batch_size", default_config["batch_size"])
        config["lr_events"] = config.get(
            "lr_events", default_config["lr_events"])
        config["lr_nans"] = config.get("lr_nans", default_config["lr_nans"])
        config["optimizer_events"] = Optimizer.get_optimizer(
            config["optimizer_events"], config["lr_events"], self.model_events)
        config["optimizer_nans"] = Optimizer.get_optimizer(
            config["optimizer_nans"], config["lr_nans"], self.model_nans)
        config["patch_size"] = config.get(
            "patch_size", default_config["patch_size"])

        self.config = config

    def get_default_config(self):
        """
        Get default config function for the model.
        :return: default config
        """
        return {"batch_size": 32, "lr_events": 0.001, "lr_nans": 0.001, 'patch_size': 36}

    def load_transformer_config(self, config):
        """
        Load config function for the model.
        :param config: configuration to set up the model
        :return:
        """
        # Check if all necessary parameters are in the config.
        default_config = self.get_default_transformer_config()
        new_config = default_config.copy()
        for key in default_config:
            if key in config:
                new_config[key] = config[key]

        return new_config

    def get_default_transformer_config(self):
        """
        Get default config function for the model.
        :return: default config
        """
        return {
            'feat_dim': 72,
            'max_len': 480,
            'd_model': 192,
            'n_heads': 6,
            'n_layers': 5,
            'dim_feedforward': 2048,
            'num_classes': 2,
            'dropout': 0.1,
            'pos_encoding': "learnable",
            'act_int': "relu",
            'act_out': "relu",
            'norm': "BatchNorm",
            'freeze': False,
        }

    def train(self, X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array):
        """
        Train function for the model.
        :param data: labelled data
        :return:
        """

        # Get hyperparameters from config (epochs, lr, optimizer)
        logger.info(f"Training model: {type(self).__name__}")
        logger.info(f"Hyperparameters: {self.config}")

        # Load hyperparameters
        criterion_events = self.config["loss_events"]
        criterion_nans = self.config["loss_nans"]
        optimizer_events = self.config["optimizer_events"]
        optimizer_nans = self.config["optimizer_nans"]
        epochs_events = self.config["epochs_events"]
        epochs_nans = self.config["epochs_nans"]
        batch_size = self.config["batch_size"]

        # Print the shapes and types of train and test
        logger.debug(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.debug(
            f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logger.debug(
            f"X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")
        logger.debug(
            f"X_test type: {X_test.dtype}, y_test type: {y_test.dtype}")

        # Remove labels
        y_train = y_train[:, :, 1:]
        y_test = y_test[:, :, 1:]

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

        # Do patching
        patch_size = self.config["patch_size"]

        # Patch the data for the features
        X_train = patch_x_data(X_train, patch_size)
        X_test = patch_x_data(X_test, patch_size)

        # Patch the data for the labels
        y_train = patch_y_data(y_train, patch_size)
        y_test = patch_y_data(y_test, patch_size)

        # Regression
        y_train = y_train[:, 0]
        y_test = y_test[:, 0]

        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        # Create a dataloader from the dataset
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size)

        # Train events
        logger.info("Training events model")
        trainer = Trainer(epochs=epochs_events,
                                 criterion=criterion_events)
        avg_train_loss_event, avg_val_loss_event = trainer.fit(
            train_dataloader, test_dataloader, self.model_events, optimizer_events, self.name)
        if wandb.run is not None:
            self.log_train_test(avg_train_loss_event,
                                avg_val_loss_event, epochs_events)

        # Train nans
        logger.info("Training nans model")
        trainer = Trainer(epochs=epochs_nans, criterion=criterion_nans)
        avg_train_loss_nan, avg_val_loss_nan = trainer.fit(
            train_dataloader, test_dataloader, self.model_nans, optimizer_nans, self.name)
        if wandb.run is not None:
            self.log_train_test(avg_train_loss_nan,
                                avg_val_loss_nan, epochs_nans)

    def pred(self, data: np.ndarray[Any, dtype[Any]]) -> ndarray[Any, dtype[Any]]:
        """
        Prediction function for the model.
        :param data: unlabelled data
        :return:
        """

        # Push to device
        self.model_events.to(self.device).double()
        self.model_nans.to(self.device).double()

        # Turn data from (window_size, features) to (1, window_size, features)
        data = torch.from_numpy(data)  # .unsqueeze(0)

        # Patch data
        patch_size = self.config["patch_size"]
        data = patch_x_data(data, patch_size)

        test_dataset = torch.utils.data.TensorDataset(
            data, torch.zeros((data.shape[0], data.shape[1])))

        # Create a dataloader from the dataset
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["batch_size"])

        # Make predictions
        prediction_events = np.empty((0, 2))
        prediction_nans = np.empty((0, 2))
        with tqdm(test_dataloader, unit="batch", disable=False) as tepoch:
            for idx, data in enumerate(tepoch):
                prediction_events = self._pred_one_batch(
                    data, prediction_events, self.model_events)
                prediction_nans = self._pred_one_batch(
                    data, prediction_nans, self.model_nans)

        # Set the threshold for the nans
        threshold = 0.5

        # Create mask where if prediction_nans is above threshold, prediction_events is nans
        for i in range(len(prediction_events)):
            if prediction_nans[i, 0] > threshold or prediction_nans[i, 1] > threshold:
                prediction_events[i, 0] = np.NAN
                prediction_events[i, 1] = np.NAN

        return prediction_events

    def _pred_one_batch(self, data: torch.utils.data.DataLoader, preds: List[float], model: nn.Module) -> List[float]:
        """
        Predict one batch and return the predictions.
        :param data: data to predict on
        :param preds: predictions to append to
        :param model: model to predict with
        :return: predictions
        """

        # Make predictions without gradient
        with torch.no_grad():
            data[0] = data[0].double()
            padding_mask = torch.ones((data[0].shape[0], data[0].shape[1])) > 0
            output = model(data[0].to(self.device),
                           padding_mask.to(self.device))
            preds = np.concatenate((preds, output.cpu().numpy()), axis=0)
        return preds

    def save(self, path: str) -> None:
        """
        Save function for the model.
        :param path: path to save the model to
        :return:
        """
        checkpoint = {
            'model_state_dict_events': self.model_events.state_dict(),
            'model_state_dict_nans': self.model_nans.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        logger.info("Model saved to: " + path)

    def load(self, path: str, only_hyperparameters: bool = False) -> None:
        """
        Load function for the model.
        :param path: path to model checkpoint
        :return:
        """
        self.model_events = TSTransformerEncoderClassiregressor(
            **self.events_transformer_config)
        self.model_nans = TSTransformerEncoderClassiregressor(
            **self.nans_transformer_config)
        checkpoint = torch.load(path)
        self.model_events.load_state_dict(
            checkpoint['model_state_dict_events'])
        self.model_nans.load_state_dict(checkpoint['model_state_dict_nans'])
        self.config = checkpoint['config']
