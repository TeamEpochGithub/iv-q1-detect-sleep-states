import copy
import numpy as np
import torch
import wandb

from src.logger.logger import logger
from src.loss.loss import Loss
from src.models.trainers.event_trainer import EventTrainer
from src.optimizer.optimizer import Optimizer
from src.util.state_to_event import pred_to_event_state

from torch import nn
from tqdm import tqdm
from numpy import ndarray, dtype
from typing import Any
from .architecture.transformer_pool import TransformerPool
from .base_transformer import BaseTransformer
from ... import data_info
from torch.utils.data import TensorDataset, DataLoader


class EventSegmentationTransformer(BaseTransformer):
    """
    This is the model file for the event segmentation transformer model.
    """

    def __init__(self, config: dict, name: str) -> None:
        """
        Init function of the event segmentation transformer model
        :param config: configuration to set up the model
        :param data_shape: shape of the data (channels, sequence_size)
        :param name: name of the model
        """
        super().__init__(config=config, name=name)
        # Init model
        self.model_type = "event-segmentation-transformer"

        # Load transformer config and model
        self.transformer_config["t_type"] = "event"
        self.transformer_config["num_class"] = 1
        self.model_onset = TransformerPool(**self.transformer_config)
        self.model_awake = TransformerPool(**self.transformer_config)

        # Load model class config
        self.load_config(**self.config)

        # Initialize weights
        for p in self.model_onset.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.model_awake.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def train(self, X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array) -> None:
        """
        Train function for the model.
        :param X_train: the training data
        :param X_test: the test data
        :param y_train: the training labels
        :param y_test: the test labels
        """

        # Get hyperparameters from config (epochs, lr, optimizer)
        logger.info(f"Training model: {type(self).__name__}")
        logger.info(f"Hyperparameters: {self.config}")

        # Load hyperparameters
        criterion = self.config["loss"]
        optimizer_onset = self.config["optimizer_onset"]
        optimizer_awake = self.config["optimizer_awake"]
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        mask_unlabeled = self.config["mask_unlabeled"]
        early_stopping = self.config["early_stopping"]
        if early_stopping > 0:
            logger.info(
                f"--- Early stopping enabled with patience of {early_stopping} epochs.")

        # Print the shapes and types of train and test
        logger.debug(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.debug(
            f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logger.debug(
            f"X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")
        logger.debug(
            f"X_test type: {X_test.dtype}, y_test type: {y_test.dtype}")

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

        # Dataset for onset
        if mask_unlabeled:
            train_dataset_onset = TensorDataset(
                X_train, y_train[:, :, (data_info.y_columns["awake"], data_info.y_columns["state-onset"])])
            test_dataset_onset = TensorDataset(
                X_test, y_test[:, :, (data_info.y_columns["awake"], data_info.y_columns["state-onset"])])
        else:
            train_dataset_onset = TensorDataset(
                X_train, y_train[:, :, data_info.y_columns["state-onset"]])
            test_dataset_onset = TensorDataset(
                X_test, y_test[:, :, data_info.y_columns["state-onset"]])

        # Dataset for awake
        if mask_unlabeled:
            train_dataset_awake = TensorDataset(
                X_train, y_train[:, :, (data_info.y_columns["awake"], data_info.y_columns["state-wakeup"])])
            test_dataset_awake = TensorDataset(
                X_test, y_test[:, :, (data_info.y_columns["awake"], data_info.y_columns["state-wakeup"])])
        else:
            train_dataset_awake = TensorDataset(
                X_train, y_train[:, :, data_info.y_columns["state-wakeup"]])
            test_dataset_awake = TensorDataset(
                X_test, y_test[:, :, data_info.y_columns["state-wakeup"]])

        # Create a dataloader from the dataset for onset
        train_dataloader_onset = DataLoader(
            train_dataset_onset, batch_size=batch_size)
        test_dataloader_onset = DataLoader(
            test_dataset_onset, batch_size=batch_size)

        # Create a dataloader from the dataset for awake
        train_dataloader_awake = DataLoader(
            train_dataset_awake, batch_size=batch_size)
        test_dataloader_awake = DataLoader(
            test_dataset_awake, batch_size=batch_size)

        # Train the onset model
        logger.info("--- Training onset model")
        trainer_onset = EventTrainer(
            epochs, criterion, mask_unlabeled, early_stopping)
        avg_losses_onset, avg_val_losses_onset, total_epochs_onset = trainer_onset.fit(
            train_dataloader_onset, test_dataloader_onset, self.model_onset, optimizer_onset, self.name + "_onset")

        # Train the awake model
        logger.info("--- Training awake model")
        trainer_awake = EventTrainer(
            epochs, criterion, mask_unlabeled, early_stopping)
        avg_losses_awake, avg_val_losses_awake, total_epochs_awake = trainer_awake.fit(
            train_dataloader_awake, test_dataloader_awake, self.model_awake, optimizer_awake, self.name + "_awake")

        self.config["total_epochs_onset"] = total_epochs_onset
        self.config["total_epochs_awake"] = total_epochs_awake

        if wandb.run is not None:
            # Log onset
            self.log_train_test(avg_losses_onset, avg_val_losses_onset, len(
                avg_losses_onset), "_onset")

            # Log awake
            self.log_train_test(avg_losses_awake, avg_val_losses_awake, len(
                avg_losses_awake), "_awake")

        logger.info("--- Train complete!")

    def train_full(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model on the full dataset.
        :param X_train: the training data
        :param y_train: the training labels
        """
        # Get hyperparameters from config (epochs, lr, optimizer)
        logger.info(f"Training model: {type(self).__name__}")
        logger.info(f"Hyperparameters: {self.config}")

        # Load hyperparameters
        criterion = self.config["loss"]
        optimizer_onset = self.config["optimizer_onset"]
        optimizer_awake = self.config["optimizer_awake"]
        epochs_onset = self.config["total_epochs_onset"]
        epochs_awake = self.config["total_epochs_awake"]
        batch_size = self.config["batch_size"]
        mask_unlabeled = self.config["mask_unlabeled"]

        # Print the shapes and types of train and test
        logger.info("--- Running for " + str(epochs_onset) + " epochs_onset.")
        logger.info("--- Running for " + str(epochs_awake) + " epochs_awake.")

        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)

        # Dataset for onset
        if mask_unlabeled:
            train_dataset_onset = TensorDataset(
                X_train, y_train[:, :, (data_info.y_columns["awake"], data_info.y_columns["state-onset"])])
        else:
            train_dataset_onset = TensorDataset(
                X_train, y_train[:, :, data_info.y_columns["state-onset"]])

        # Dataset for awake
        if mask_unlabeled:
            train_dataset_awake = TensorDataset(
                X_train, y_train[:, :, (data_info.y_columns["awake"], data_info.y_columns["state-wakeup"])])
        else:
            train_dataset_awake = TensorDataset(
                X_train, y_train[:, :, data_info.y_columns["state-wakeup"]])

        # Create dataloaders for awake and onset
        train_dataloader_onset = DataLoader(
            train_dataset_onset, batch_size=batch_size)

        train_dataloader_awake = DataLoader(
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
                          optimizer_onset, self.name + "_onset_full")

        # Train the awake model
        logger.info("--- Training awake model full")
        trainer_awake = EventTrainer(
            epochs_awake, criterion, mask_unlabeled, -1)
        trainer_awake.fit(train_dataloader_awake, None, self.model_awake,
                          optimizer_awake, self.name + "_awake_full")

        logger.info("--- Full train complete!")

    def pred(self, data: np.ndarray[Any, dtype[Any]], pred_with_cpu: bool = False) -> ndarray[Any, dtype[Any]]:
        """
        Prediction function for the model.
        :param data: unlabelled data
        :param with_cpu: whether to use cpu
        :return: predictions of the model (windows, labels)
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
        dataset = TensorDataset(torch.from_numpy(data))
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
        # Permute np array to (batch, 1, steps)
        predictions_onset = predictions_onset.transpose(0, 2, 1)

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
        predictions_awake = predictions_awake.transpose(0, 2, 1)

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
            logger.info(
                "Loading hyperparameters and instantiate new model from: " + path)
            return

        self.model_onset.load_state_dict(checkpoint['onset_model_state_dict'])
        self.model_awake.load_state_dict(checkpoint['awake_model_state_dict'])
        self.reset_optimizer()
        logger.info("Model fully loaded from: " + path)

    def load_config(self, loss: str, epochs: int, optimizer: str, **kwargs: dict) -> None:
        """
        Load config function for the model.
        :param loss: loss function
        :param epochs: number of epochs
        :param optimizer: optimizer
        :param kwargs: other parameters
        """

        # Get default_config
        default_config = self.get_default_config()

        # Copy kwargs
        config = copy.deepcopy(kwargs)

        # Add parameters
        config["batch_size"] = config.get(
            "batch_size", default_config["batch_size"])
        config["lr"] = config.get("lr", default_config["lr"])
        config["early_stopping"] = config.get(
            "early_stopping", default_config["early_stopping"])
        config["threshold"] = config.get(
            "threshold", default_config["threshold"])

        # Add loss, epochs and optimizer to config
        config["mask_unlabeled"] = config.get(
            "mask_unlabeled", default_config["mask_unlabeled"])
        if config["mask_unlabeled"]:
            config["loss"] = Loss.get_loss(loss, reduction="none")
        else:
            config["loss"] = Loss.get_loss(loss, reduction="mean")
        config["optimizer_onset"] = Optimizer.get_optimizer(
            optimizer, config["lr"], model=self.model_onset)
        config["optimizer_awake"] = Optimizer.get_optimizer(
            optimizer, config["lr"], model=self.model_awake)
        config["epochs"] = epochs
        config["trained_epochs_onset"] = epochs
        config["trained_epochs_awake"] = epochs

        self.config = config

    def reset_optimizer(self) -> None:
        """
        Reset the optimizer to the initial state. Useful for retraining the model.
        """
        self.config['optimizer_onset'] = type(self.config['optimizer_onset'])(
            self.model_onset.parameters(), lr=self.config['optimizer_onset'].param_groups[0]['lr'])
        self.config[('optimizer_awake')] = type(self.config['optimizer_awake'])(
            self.model_awake.parameters(), lr=self.config['optimizer_awake'].param_groups[0]['lr'])

    def reset_weights(self) -> None:
        """
        Reset the weights of the model. Useful for retraining the model.
        """
        self.model_onset = TransformerPool(**self.transformer_config)
        self.model_awake = TransformerPool(**self.transformer_config)
