import copy
import numpy as np
import torch
import wandb

from src.logger.logger import logger
from src.models.transformers.trainers.segmentation_trainer import SegmentationTrainer
from src.util.state_to_event import find_events

from ...loss.loss import Loss
from ..model import Model
from ...optimizer.optimizer import Optimizer
from typing import List
from torch import nn
from tqdm import tqdm
from numpy import ndarray, dtype
from typing import Any
from .architecture.transformer_pool import TransformerPool


class SegmentationTransformer(Model):
    """
    This is the model file for the patch pool event regression transformer model.
    """

    def __init__(self, config: dict, data_shape: tuple, name: str, pred_with_cpu: bool = False) -> None:
        """
        Init function of the example model
        :param config: configuration to set up the model
        :param data_shape: shape of the data (channels, sequence_size)
        :param name: name of the model
        """
        super().__init__(config, name)
        # Init model
        self.name = name
        self.transformer_config = self.load_transformer_config(config).copy()
        self.transformer_config["seq_len"] = data_shape[1]
        self.transformer_config["no_head"] = True
        self.transformer_config["tokenizer_args"]["channels"] = data_shape[0]
        self.model = TransformerPool(tokenizer_args=self.transformer_config["tokenizer_args"],
                                     **((self.transformer_config, self.transformer_config.pop("tokenizer_args"))[0]))
        self.data_shape = data_shape

        # Initialize weights
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.transformer_config = self.load_transformer_config(config).copy()
        self.transformer_config["tokenizer_args"]["channels"] = data_shape[0]
        self.load_config(**config)
        self.config["trained_epochs"] = self.config["epochs"]
        self.config["seq_len"] = data_shape[1]

        # Check if gpu is available, else return an exception
        if not torch.cuda.is_available():
            logger.warning("GPU not available - using CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            logger.info(
                f"--- Device set to model {self.name}: " + torch.cuda.get_device_name(0))

        self.pred_with_cpu = pred_with_cpu

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
        config["patch_size"] = config.get(
            "patch_size", default_config["patch_size"])

        # Add loss, epochs and optimizer to config
        config["loss"] = Loss.get_loss(loss)
        config["optimizer"] = Optimizer.get_optimizer(
            optimizer, config["lr"], model=self.model)
        config["epochs"] = epochs

        self.config = config

    def get_default_config(self) -> dict[str, int | str]:
        """
        Get default config function for the model.
        :return: default config
        """
        return {"batch_size": 32, "lr": 0.000035, 'patch_size': 36}

    def load_transformer_config(self, config: dict[str, int | float | str]) -> dict[str, int | float | str]:
        """
        Load transformer config function for the model.
        :param config: configuration to set up the transformer architecture
        :return: transformer config
        """
        # Check if all necessary parameters are in the config.
        default_config = self.get_default_transformer_config()
        new_config = default_config.copy()
        for key in default_config:
            if key in config:
                new_config[key] = config[key]

        return new_config

    def get_default_transformer_config(self) -> dict[str, int | float | str]:
        """
        Get default config function for the model.
        :return: default config
        """
        return {
            'heads': 12,
            'emb_dim': 92,
            'forward_dim': 2048,
            'dropout': 0.1,
            'n_layers': 12,
            "tokenizer": "patch",
            'tokenizer_args': {},
            'seq_len': 17280,
            'num_class': 3,
            'pooling': 'none'
        }

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
        optimizer = self.config["optimizer"]
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

        # One hot segmentation (Preprocessing steps: 1. Add state labels, 2. One hot encode) -> Remove state labels
        y_train = y_train[:, :, 1:]
        y_test = y_test[:, :, 1:]

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

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
        trainer = SegmentationTrainer(epochs=epochs,
                                      criterion=criterion)
        avg_train_loss, avg_val_loss, self.config["trained_epochs"] = trainer.fit(
            train_dataloader, test_dataloader, self.model, optimizer, self.name)
        if wandb.run is not None:
            self.log_train_test(avg_train_loss,
                                avg_val_loss, len(avg_train_loss))

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

        # Remove labels
        y_train = y_train[:, :, 1:]

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

    def pred(self, data: np.ndarray[Any, dtype[Any]]) -> ndarray[Any, dtype[Any]]:
        """
        Prediction function for the model.
        :param data: unlabelled data
        :param with_cpu: whether to use cpu
        :return: predictions of the model (windows, labels)
        """

        # Check which device to use
        if self.pred_with_cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # Push to device
        self.model.to(device).float()

        # Turn data into numpy array
        data = torch.from_numpy(data).to(device)

        test_dataset = torch.utils.data.TensorDataset(
            data, torch.zeros((data.shape[0], data.shape[1])))

        # Create a dataloader from the dataset
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["batch_size"])

        # Make predictions
        predictions = np.empty((0, self.data_shape[1], 3))
        with tqdm(test_dataloader, unit="batch", disable=False) as tepoch:
            for _, data in enumerate(tepoch):
                predictions = self._pred_one_batch(
                    data, predictions, self.model)

        # Prediction shape is (windows, seq_len // downsample_factor, num_class)
        # Apply upsampling to the predictions
        downsampling_factor = 17280 // self.data_shape[1]
        if downsampling_factor > 1:
            predictions = np.repeat(predictions, downsampling_factor, axis=1)

        all_predictions = []

        # Convert to events
        for pred in tqdm(predictions, desc="Converting predictions to events", unit="window"):
            # Convert to relative window event timestamps
            pred = np.argmax(pred, axis=1)
            events = find_events(pred, median_filter_size=15)
            all_predictions.append(events)

        # Return numpy array
        return np.array(all_predictions)

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
            data[0] = data[0].float()
            output = model(data[0].to(self.device))
            preds = np.concatenate((preds, output.cpu().numpy()), axis=0)
        return preds

    def save(self, path: str) -> None:
        """
        Save function for the model.
        :param path: path to save the model to
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
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
        self.transformer_config["no_head"] = True
        self.transformer_config['seq_len'] = self.config['seq_len']
        self.model = TransformerPool(tokenizer_args=self.transformer_config["tokenizer_args"],
                                     **((self.transformer_config, self.transformer_config.pop("tokenizer_args"))[0]))
        if not only_hyperparameters:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.reset_optimizer()

    def reset_optimizer(self) -> None:
        """
        Reset the optimizer to the initial state. Useful for retraining the model.
        """
        self.config['optimizer'] = type(self.config['optimizer'])(
            self.model.parameters(), lr=self.config['optimizer'].param_groups[0]['lr'])
