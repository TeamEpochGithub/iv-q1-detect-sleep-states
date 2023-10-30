import numpy as np
import torch
import wandb

from src.logger.logger import logger
from src.models.transformers.trainers.segmentation_trainer import SegmentationTrainer
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
        self.model = TransformerPool(tokenizer_args=self.transformer_config["tokenizer_args"],
                                     **((self.transformer_config, self.transformer_config.pop("tokenizer_args"))[0]))
        self.transformer_config = self.load_transformer_config(config).copy()
        self.transformer_config["tokenizer_args"]["channels"] = data_shape[0]

        # Load model class config
        self.load_config(**self.config)

        # Initialize weights
        for p in self.model.parameters():
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

        # Y should have all  (Preprocessing steps: 1. Add event labels)
        assert y_train.shape[2] == 2

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
        self.model.to(device).float()

        # Turn data into numpy array
        data = torch.from_numpy(data).to(device)

        test_dataset = torch.utils.data.TensorDataset(
            data, torch.zeros((data.shape[0], data.shape[1])))

        # Create a dataloader from the dataset
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["batch_size"])

        # Make predictions
        predictions = np.empty((0, self.data_shape[1], 2))
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
        all_confidences = []
        # Convert to events
        for pred in tqdm(predictions, desc="Converting predictions to events", unit="window"):
            # Convert to relative window event timestamps
            # TODO Add automatic thresholding to the model
            events = pred_to_event_state(pred, thresh=0)
            steps = (events[0], events[1])
            confidences = (events[2], events[3])
            all_predictions.append(steps)
            all_confidences.append(confidences)

        # Return numpy array
        return np.array(all_predictions), np.array(all_confidences)
    
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
        self.transformer_config["t_type"] = "event"
        self.transformer_config['seq_len'] = self.config['seq_len']
        self.model = TransformerPool(tokenizer_args=self.transformer_config["tokenizer_args"],
                                     **((self.transformer_config, self.transformer_config.pop("tokenizer_args"))[0]))
        if not only_hyperparameters:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.reset_optimizer()
