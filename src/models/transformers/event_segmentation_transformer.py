import copy
import numpy as np
import torch
import wandb

from src.logger.logger import logger
from src.loss.loss import Loss
from src.models.transformers.trainers.segmentation_trainer import SegmentationTrainer
from src.optimizer.optimizer import Optimizer
from src.util.state_to_event import pred_to_event_state

from torch import nn
from tqdm import tqdm
from numpy import ndarray, dtype
from typing import Any
from .architecture.transformer_pool import TransformerPool
from .base_transformer import BaseTransformer


class EventSegmentationTransformer(BaseTransformer):
    """
    This is the model file for the event segmentation transformer model.
    """

    def __init__(self, config: dict, data_shape: tuple, name: str) -> None:
        """
        Init function of the event segmentation transformer model
        :param config: configuration to set up the model
        :param data_shape: shape of the data (channels, sequence_size)
        :param name: name of the model
        """
        super().__init__(config=config, data_shape=data_shape, name=name)
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

        # Print the shapes and types of train and test
        logger.debug(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.debug(
            f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logger.debug(
            f"X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")
        logger.debug(
            f"X_test type: {X_test.dtype}, y_test type: {y_test.dtype}")

        # Y should have all  (Preprocessing steps: 1. Add event labels)
        assert y_train.shape[2] == 2

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

        # Create a dataset from X and y for onset
        train_dataset_onset = torch.utils.data.TensorDataset(
            X_train, y_train[:, :, 0])
        test_dataset_onset = torch.utils.data.TensorDataset(
            X_test, y_test[:, :, 0])

        # Create a dataset from X and y for awake
        train_dataset_awake = torch.utils.data.TensorDataset(
            X_train, y_train[:, :, 1])
        test_dataset_awake = torch.utils.data.TensorDataset(
            X_test, y_test[:, :, 1])

        # Create a dataloader from the dataset for onset
        train_dataloader_onset = torch.utils.data.DataLoader(
            train_dataset_onset, batch_size=batch_size)
        test_dataloader_onset = torch.utils.data.DataLoader(
            test_dataset_onset, batch_size=batch_size)

        # Create a dataloader from the dataset for awake
        train_dataloader_awake = torch.utils.data.DataLoader(
            train_dataset_awake, batch_size=batch_size)
        test_dataloader_awake = torch.utils.data.DataLoader(
            test_dataset_awake, batch_size=batch_size)

        # Train events
        logger.info("Training onset model")
        trainer_onset = SegmentationTrainer(epochs=epochs, criterion=criterion)
        avg_train_loss_onset, avg_val_loss_onset, self.config["trained_epochs_onset"] = trainer_onset.fit(
            train_dataloader_onset, test_dataloader_onset, self.model_onset, optimizer_onset, self.name + "_onset")

        # Train awake
        logger.info("Training awake model")
        trainer_awake = SegmentationTrainer(epochs=epochs, criterion=criterion)
        avg_train_loss_awake, avg_val_loss_awake, self.config["trained_epochs_awake"] = trainer_awake.fit(
            train_dataloader_awake, test_dataloader_awake, self.model_awake, optimizer_awake, self.name + "_awake")

        if wandb.run is not None:
            # Log onset
            self.log_train_test(avg_train_loss_onset, avg_val_loss_onset, len(
                avg_train_loss_onset), "_onset")

            # Log awake
            self.log_train_test(avg_train_loss_awake, avg_val_loss_awake, len(
                avg_train_loss_awake), "_onset")

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
        optimizer = self.config["optimizer"]
        epochs = self.config["trained_epochs"]
        batch_size = self.config["batch_size"]

        # Print the shapes and types of train and test
        logger.debug(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.debug(
            f"X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")

        # Y should have all  (Preprocessing steps: 1. Add event labels)
        assert y_train.shape[2] == 2

        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)

        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

        # Create a dataloader from the dataset
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size)

        # Train events
        logger.info("Training events model")
        trainer = SegmentationTrainer(epochs=epochs,
                                      criterion=criterion)
        trainer.fit(
            train_dataloader, None, self.model, optimizer, self.name)

    def pred(self, data: np.ndarray[Any, dtype[Any]], with_cpu: bool = False) -> ndarray[Any, dtype[Any]]:
        """
        Prediction function for the model.
        :param data: unlabelled data
        :param with_cpu: whether to use cpu
        :return: predictions of the model (windows, labels)
        """

        # Check which device to use
        if with_cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # Push to device
        self.model_onset.to(device).float()
        self.model_awake.to(device).float()

        # Turn data into numpy array
        data = torch.from_numpy(data).to(device)

        test_dataset = torch.utils.data.TensorDataset(
            data, torch.zeros((data.shape[0], data.shape[1])))

        # Create a dataloader from the dataset
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["batch_size"])

        # Make predictions
        predictions_onset = np.empty((0, self.data_shape[1], 1))
        predictions_awake = np.empty((0, self.data_shape[1], 1))
        with tqdm(test_dataloader, unit="batch", disable=False) as tepoch:
            for _, data in enumerate(tepoch):
                predictions_onset = self._pred_one_batch(
                    data, predictions_onset, self.model_onset)
                predictions_awake = self._pred_one_batch(
                    data, predictions_awake, self.model_awake)

        # Prediction shape is (windows, seq_len // downsample_factor, num_class)
        # Apply upsampling to the predictions
        downsampling_factor = 17280 // self.data_shape[1]
        if downsampling_factor > 1:
            predictions_onset = np.repeat(
                predictions_onset, downsampling_factor, axis=1)
            predictions_awake = np.repeat(
                predictions_awake, downsampling_factor, axis=1)

        # Concatenate predictions and return (n, seq_len, num_class) + (n, seq_len, num_class) -> (n, seq_len, num_class * 2)
        predictions = np.concatenate(
            (predictions_onset, predictions_awake), axis=2)
        # Swap axes to (n, num_class * 2, seq_len)
        predictions = np.swapaxes(predictions, 1, 2)

        all_predictions = []
        all_confidences = []
        # Convert to events
        for pred in tqdm(predictions, desc="Converting predictions to events", unit="window"):
            # Convert to relative window event timestamps
            events = pred_to_event_state(pred, thresh=0)

            # Add step offset based on repeat factor.
            offset = ((downsampling_factor / 2.0) - 0.5 if downsampling_factor %
                      2 == 0 else downsampling_factor // 2) if downsampling_factor > 1 else 0
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
        logger.info("Model saved to: " + path)

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

        if not only_hyperparameters:
            self.model_onset.load_state_dict(
                checkpoint['onset_model_state_dict'])
            self.model_awake.load_state_dict(
                checkpoint['awake_model_state_dict'])
        else:
            self.reset_optimizer()

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

        # Add loss, epochs and optimizer to config
        config["loss"] = Loss.get_loss(loss)
        config["optimizer_onset"] = Optimizer.get_optimizer(
            optimizer, config["lr"], model=self.model_onset)
        config["optimizer_awake"] = Optimizer.get_optimizer(
            optimizer, config["lr"], model=self.model_awake)
        config["epochs"] = epochs
        config["trained_epochs_onset"] = epochs
        config["trained_epochs_awake"] = epochs

        self.config = config
