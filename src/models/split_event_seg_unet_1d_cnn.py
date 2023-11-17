import copy
from typing import Any

import numpy as np
import torch
import wandb
from numpy import ndarray, dtype
from timm.scheduler import CosineLRScheduler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .architectures.seg_unet_1d_cnn import SegUnet1D
from .model import Model, ModelException
from .trainers.event_trainer import EventTrainer
from .. import data_info
from ..logger.logger import logger
from ..loss.loss import Loss
from ..optimizer.optimizer import Optimizer
from ..util.state_to_event import pred_to_event_state


class SplitEventSegmentationUnet1DCNN(Model):
    """
    This model is an event segmentation model based on the Unet 1D CNN. It uses the architecture from the SegSimple1DCNN class.
    """

    def __init__(self, config: dict, name: str) -> None:
        """
        Init function of the example model
        :param config: configuration to set up the model
        :param name: name of the model
        """
        super().__init__(config, name)

        # Check if gpu is available, else return an exception
        if not torch.cuda.is_available():
            logger.warning("GPU not available - using CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            logger.info(
                f"--- Device set to model {self.name}: " + torch.cuda.get_device_name(0))

        self.model_type = "event-segmentation"

        # We load the model architecture here. 2 Out channels, one for onset, one for offset event state prediction
        self.model_onset = SegUnet1D(
            in_channels=len(data_info.X_columns), window_size=data_info.window_size, out_channels=1, model_type=self.model_type, **self.config.get("network_params", {}))
        self.model_awake = SegUnet1D(
            in_channels=len(data_info.X_columns), window_size=data_info.window_size, out_channels=1, model_type=self.model_type, **self.config.get("network_params", {}))

        # Load config
        self.load_config(config)

        # Print model summary
        if wandb.run is not None:
            if data_info.plot_summary:
                from torchsummary import summary
                summary(self.model_onset.cuda(), input_size=(
                    len(data_info.X_columns), data_info.window_size))
                summary(self.model_awake.cuda(), input_size=(
                    len(data_info.X_columns), data_info.window_size))

    def load_config(self, config: dict) -> None:
        """
        Load config function for the model.
        :param config: configuration to set up the model
        """
        config = copy.deepcopy(config)

        # Error checks. Check if all necessary parameters are in the config.
        required = ["loss", "optimizer"]
        for req in required:
            if req not in config:
                logger.critical(
                    "------ Config is missing required parameter: " + req)
                raise ModelException(
                    "Config is missing required parameter: " + req)

        # Get default_config
        default_config = self.get_default_config()
        config["mask_unlabeled"] = config.get(
            "mask_unlabeled", default_config["mask_unlabeled"])
        if config["mask_unlabeled"]:
            config["loss"] = Loss.get_loss(config["loss"], reduction="none")
        else:
            if config["loss"] == "kldiv-torch":
                config["loss"] = Loss.get_loss(config["loss"], reduction="batchmean")
            else:
                config["loss"] = Loss.get_loss(config["loss"], reduction="mean")
        config["batch_size"] = config.get(
            "batch_size", default_config["batch_size"])
        config["epochs"] = config.get("epochs", default_config["epochs"])
        config["lr"] = config.get("lr", default_config["lr"])
        config["early_stopping"] = config.get("early_stopping", default_config["early_stopping"])
        config["threshold"] = config.get("threshold", default_config["threshold"])
        config["weight_decay"] = config.get("weight_decay", default_config["weight_decay"])
        config["optimizer_onset"] = Optimizer.get_optimizer(config["optimizer"], config["lr"], config["weight_decay"], self.model_onset)
        config["optimizer_awake"] = Optimizer.get_optimizer(config["optimizer"], config["lr"], config["weight_decay"], self.model_awake)
        if "lr_schedule" in config:
            config["lr_schedule"] = config.get("lr_schedule", default_config["lr_schedule"])
            config["scheduler_onset"] = CosineLRScheduler(config["optimizer_onset"], **self.config["lr_schedule"])
            config["scheduler_awake"] = CosineLRScheduler(config["optimizer_awake"], **self.config["lr_schedule"])
        config["activation_delay"] = config.get("activation_delay", default_config["activation_delay"])
        config["network_params"] |= self.get_default_config()["network_params"]
        self.config = config

    def load_network_params(self, config: dict) -> dict:
        return config["network_params"] | self.get_default_config()["network_params"]

    def get_default_config(self) -> dict:
        """
        Get default config function for the model.
        :return: default config
        """
        return {
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 10,
            "early_stopping": -1,
            "threshold": 0.5,
            "weight_decay": 0.0,
            "mask_unlabeled": False,
            "lr_schedule": {
                "t_initial": 100,
                "warmup_t": 5,
                "warmup_lr_init": 0.000001,
                "lr_min": 2e-8
            },
            "activation_delay": 0,
            "network_params": {
                "activation": "relu",
                "hidden_layers": 8,
                "kernel_size": 7,
                "depth": 2,
            }
        }

    def get_type(self) -> str:
        """
        Get type function for the model.
        :return: the type of the model
        """
        return self.model_type

    def train(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        """
        Train function for the model.
        :param X_train: the training data
        :param X_test: the test data
        :param y_train: the training labels
        :param y_test: the test labels
        """
        # Get hyperparameters from config (epochs, lr, optimizer)
        # Load hyperparameters
        criterion = self.config["loss"]
        optimizer_onset = self.config["optimizer_onset"]
        optimizer_awake = self.config["optimizer_awake"]
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        early_stopping = self.config["early_stopping"]
        mask_unlabeled = self.config["mask_unlabeled"]
        activation_delay = self.config["activation_delay"]
        if "scheduler" in self.config:
            scheduler_onset = self.config["scheduler_onset"]
            scheduler_awake = self.config["scheduler_awake"]
        else:
            scheduler_onset = None
            scheduler_awake = None
        if early_stopping > 0:
            logger.info(
                f"--- Early stopping enabled with patience of {early_stopping} epochs.")

        # X_train and X_test are of shape (n, channels, window_size)
        X_train = torch.from_numpy(X_train).permute(0, 2, 1)
        X_test = torch.from_numpy(X_test).permute(0, 2, 1)

        # Get only the 2 event state features
        y_train = torch.from_numpy(y_train).permute(0, 2, 1)
        y_test = torch.from_numpy(y_test).permute(0, 2, 1)

        # Print the shapes and types of train and test
        logger.info(
            f"--- X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(
            f"--- X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logger.info(
            f"--- X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")
        logger.info(
            f"--- X_test type: {X_test.dtype}, y_test type: {y_test.dtype}")

        # Create dataloaders for awake and onset

        # Dataset for onset
        if mask_unlabeled:
            train_dataset_onset = torch.utils.data.TensorDataset(
                X_train, y_train[:, (data_info.y_columns["awake"], data_info.y_columns["state-onset"]), :])
            test_dataset_onset = torch.utils.data.TensorDataset(
                X_test, y_test[:, (data_info.y_columns["awake"], data_info.y_columns["state-onset"]), :])
        else:
            train_dataset_onset = torch.utils.data.TensorDataset(
                X_train, y_train[:, data_info.y_columns["state-onset"], :])
            test_dataset_onset = torch.utils.data.TensorDataset(
                X_test, y_test[:, data_info.y_columns["state-onset"], :])

        # Dataset for awake
        if mask_unlabeled:
            train_dataset_awake = torch.utils.data.TensorDataset(
                X_train, y_train[:, (data_info.y_columns["awake"], data_info.y_columns["state-wakeup"]), :])
            test_dataset_awake = torch.utils.data.TensorDataset(
                X_test, y_test[:, (data_info.y_columns["awake"], data_info.y_columns["state-wakeup"]), :])
        else:
            train_dataset_awake = torch.utils.data.TensorDataset(
                X_train, y_train[:, data_info.y_columns["state-wakeup"], :])
            test_dataset_awake = torch.utils.data.TensorDataset(
                X_test, y_test[:, data_info.y_columns["state-wakeup"], :])

        # Create dataloaders for awake and onset
        train_dataloader_onset = torch.utils.data.DataLoader(
            train_dataset_onset, batch_size=batch_size)
        test_dataloader_onset = torch.utils.data.DataLoader(
            test_dataset_onset, batch_size=batch_size)

        train_dataloader_awake = torch.utils.data.DataLoader(
            train_dataset_awake, batch_size=batch_size)
        test_dataloader_awake = torch.utils.data.DataLoader(
            test_dataset_awake, batch_size=batch_size)

        # Train the onset model
        logger.info("--- Training onset model")

        trainer_onset = EventTrainer(
            epochs, criterion, mask_unlabeled, early_stopping)
        avg_losses_onset, avg_val_losses_onset, total_epochs_onset = trainer_onset.fit(
            train_dataloader_onset, test_dataloader_onset, self.model_onset, optimizer_onset, self.name + "_onset", scheduler=scheduler_onset,
            activation_delay=activation_delay)

        # Train the awake model
        logger.info("--- Training awake model")
        trainer_awake = EventTrainer(
            epochs, criterion, mask_unlabeled, early_stopping)
        avg_losses_awake, avg_val_losses_awake, total_epochs_awake = trainer_awake.fit(
            train_dataloader_awake, test_dataloader_awake, self.model_awake, optimizer_awake, self.name + "_awake", scheduler=scheduler_awake,
            activation_delay=activation_delay)

        # Log full train and test plot
        if wandb.run is not None:
            self.log_train_test(
                avg_losses_onset[:total_epochs_onset], avg_val_losses_onset[:total_epochs_onset], total_epochs_onset, "onset")
            self.log_train_test(
                avg_losses_awake[:total_epochs_awake], avg_val_losses_awake[:total_epochs_awake], total_epochs_awake, "awake")
        logger.info("--- Training of model complete!")

        self.config["total_epochs_onset"] = total_epochs_onset
        self.config["total_epochs_awake"] = total_epochs_awake

    def train_full(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model on the full dataset.
        :param X_train: the training data
        :param y_train: the training labels
        """
        criterion = self.config["loss"]
        optimizer_onset = self.config["optimizer_onset"]
        optimizer_awake = self.config["optimizer_awake"]
        epochs_onset = self.config["total_epochs_onset"]
        epochs_awake = self.config["total_epochs_awake"]
        batch_size = self.config["batch_size"]
        mask_unlabeled = self.config["mask_unlabeled"]
        activation_delay = self.config["activation_delay"]
        if "scheduler" in self.config:
            scheduler_onset = self.config["scheduler_onset"]
            scheduler_awake = self.config["scheduler_awake"]

        else:
            scheduler_onset = None
            scheduler_awake = None

        logger.info("--- Running for " + str(epochs_onset) + " epochs_onset.")
        logger.info("--- Running for " + str(epochs_awake) + " epochs_awake.")

        X_train = torch.from_numpy(X_train).permute(0, 2, 1)

        # Get only the event state features
        y_train = torch.from_numpy(y_train).permute(0, 2, 1)

        # Dataset for onset
        if mask_unlabeled:
            train_dataset_onset = torch.utils.data.TensorDataset(
                X_train, y_train[:, (data_info.y_columns["awake"], data_info.y_columns["state-onset"]), :])
        else:
            train_dataset_onset = torch.utils.data.TensorDataset(
                X_train, y_train[:, data_info.y_columns["state-onset"], :])

        # Dataset for awake
        if mask_unlabeled:
            train_dataset_awake = torch.utils.data.TensorDataset(
                X_train, y_train[:, (data_info.y_columns["awake"], data_info.y_columns["state-wakeup"]), :])
        else:
            train_dataset_awake = torch.utils.data.TensorDataset(
                X_train, y_train[:, data_info.y_columns["state-wakeup"], :])

        # Create dataloaders for awake and onset
        train_dataloader_onset = torch.utils.data.DataLoader(
            train_dataset_onset, batch_size=batch_size)

        train_dataloader_awake = torch.utils.data.DataLoader(
            train_dataset_awake, batch_size=batch_size)

        # Print the shapes and types of train and test
        logger.info(
            f"--- X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(
            f"--- X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")

        # Train the onset model
        logger.info("--- Training onset model full")
        trainer_onset = EventTrainer(
            epochs_onset, criterion, mask_unlabeled, -1)
        trainer_onset.fit(train_dataloader_onset, None, self.model_onset,
                          optimizer_onset, self.name + "_onset_full", scheduler=scheduler_onset, activation_delay=activation_delay)

        # Train the awake model
        logger.info("--- Training awake model full")
        trainer_awake = EventTrainer(
            epochs_awake, criterion, mask_unlabeled, -1)
        trainer_awake.fit(train_dataloader_awake, None, self.model_awake,
                          optimizer_awake, self.name + "_awake_full", scheduler=scheduler_awake, activation_delay=activation_delay)

        logger.info("--- Full train complete!")

    def pred(self, data: np.ndarray, pred_with_cpu: bool) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Prediction function for the model.
        :param data: unlabelled data
        :return: the predictions
        """
        # Prediction function
        logger.info(f"--- Predicting results with model {self.name}")
        # Run the model on the data and return the predictions

        if pred_with_cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # Set models to eval for inference
        self.model_onset.eval()
        self.model_awake.eval()

        self.model_onset.to(device)
        self.model_awake.to(device)

        # Print data shape
        logger.info(f"--- Data shape of predictions dataset: {data.shape}")

        # Create a DataLoader for batched inference
        dataset = TensorDataset(torch.from_numpy(data).permute(0, 2, 1))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        # Onset predictions
        predictions_onset = []
        with torch.no_grad():
            for batch_data in tqdm(dataloader, "Predicting", unit="batch"):
                batch_data = batch_data[0].to(device)

                # Make a batch prediction
                batch_prediction = self.model_onset(batch_data)

                if pred_with_cpu:
                    batch_prediction = batch_prediction.numpy()
                else:
                    batch_prediction = batch_prediction.cpu().numpy()

                predictions_onset.append(batch_prediction)

        # Concatenate the predictions from all batches for onset
        predictions_onset = np.concatenate(predictions_onset, axis=0)

        # Awake predictions
        predictions_awake = []
        with torch.no_grad():
            for batch_data in tqdm(dataloader, "Predicting", unit="batch"):
                batch_data = batch_data[0].to(device)

                # Make a batch prediction
                batch_prediction = self.model_awake(batch_data)

                if pred_with_cpu:
                    batch_prediction = batch_prediction.numpy()
                else:
                    batch_prediction = batch_prediction.cpu().numpy()

                predictions_awake.append(batch_prediction)

        # Concatenate the predictions from all batches for awake
        predictions_awake = np.concatenate(predictions_awake, axis=0)

        # Concatenate the predictions from awake and onset (batch, steps, 1) + (batch, steps, 1) = (batch, steps, 2)
        predictions = np.concatenate(
            (predictions_onset, predictions_awake), axis=1)

        # Apply upsampling to the predictions
        if data_info.downsampling_factor > 1:
            predictions = np.repeat(
                predictions, data_info.downsampling_factor, axis=2)

        all_predictions = []
        all_confidences = []
        # Convert to events
        for pred in tqdm(predictions, desc="Converting predictions to events", unit="window"):
            # Convert to relative window event timestamps
            events = pred_to_event_state(pred, thresh=self.config["threshold"])

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

    def save(self, path: str) -> None:
        """
        Save function for the model.
        :param path: path to save the model to
        """
        checkpoint = {
            'onset_model_state_dict': self.model_onset.state_dict(),
            'awake_model_state_dict': self.model_awake.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        logger.info("--- Model saved to: " + path)

    def load(self, path: str, only_hyperparameters: bool = False) -> None:
        """
        Load function for the model.
        :param path: path to model checkpoint
        :param only_hyperparameters: whether to only load the hyperparameters
        """
        if self.device == torch.device("cpu"):
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(path)
        self.config = checkpoint['config']
        if only_hyperparameters:
            self.reset_weights()
            self.reset_optimizer()
            self.reset_scheduler()
            logger.info(
                "Loading hyperparameters and instantiate new model from: " + path)
            return

        self.model_onset.load_state_dict(checkpoint['onset_model_state_dict'])
        self.model_awake.load_state_dict(checkpoint['awake_model_state_dict'])
        self.reset_optimizer()
        self.reset_scheduler()
        logger.info("Model fully loaded from: " + path)

    def reset_optimizer(self) -> None:

        """
        Reset the optimizer to the initial state. Useful for retraining the model.
        """
        self.config['optimizer_onset'] = type(self.config['optimizer_onset'])(self.model_onset.parameters(), lr=self.config['lr'])
        self.config['optimizer_awake'] = type(self.config['optimizer_awake'])(self.model_awake.parameters(), lr=self.config['lr'])

    def reset_weights(self) -> None:
        """
        Reset the weights of the model. Useful for retraining the model.
        """
        self.model_onset = SegUnet1D(
            in_channels=len(data_info.X_columns), window_size=data_info.window_size, out_channels=1, model_type=self.model_type, **self.config.get("network_params", {}))
        self.model_awake = SegUnet1D(
            in_channels=len(data_info.X_columns), window_size=data_info.window_size, out_channels=1, model_type=self.model_type, **self.config.get("network_params", {}))

    def reset_scheduler(self) -> None:
        """
        Reset the scheduler to the initial state. Useful for retraining the model.
        """
        if 'scheduler' in self.config:
            self.config['scheduler_onset'] = CosineLRScheduler(self.config['optimizer_onset'], **self.config["lr_schedule"])
            self.config['scheduler_awake'] = CosineLRScheduler(self.config['optimizer_awake'], **self.config["lr_schedule"])
