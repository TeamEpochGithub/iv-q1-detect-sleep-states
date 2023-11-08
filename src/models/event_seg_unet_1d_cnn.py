import copy
from typing import Any

import numpy as np
import torch
import wandb
from numpy import ndarray, dtype
from sklearn.metrics import roc_curve
from timm.scheduler import CosineLRScheduler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .architectures.seg_unet_1d_cnn import SegUnet1D
from .trainers.event_trainer import EventTrainer
from .. import data_info
from ..logger.logger import logger
from ..loss.loss import Loss
from ..models.model import Model, ModelException
from ..optimizer.optimizer import Optimizer
from ..util.state_to_event import pred_to_event_state


class EventSegmentationUnet1DCNN(Model):
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
        self.model = SegUnet1D(in_channels=len(data_info.X_columns), window_size=data_info.window_size, out_channels=2, model_type=self.model_type,
                                     **self.load_network_params(config))

        # Load optimizer
        self.load_optimizer()

        # Load config
        self.load_config(config)

        # Print model summary
        if wandb.run is not None:
            from torchsummary import summary
            summary(self.model.cuda(), input_size=(
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
        config["early_stopping"] = config.get(
            "early_stopping", default_config["early_stopping"])
        config["threshold"] = config.get(
            "threshold", default_config["threshold"])
        config["weight_decay"] = config.get(
            "weight_decay", default_config["weight_decay"])
        if "lr_schedule" in config:
            config["lr_schedule"] = config.get("lr_schedule", default_config["lr_schedule"])
            config["scheduler"] = CosineLRScheduler(config["optimizer"], **self.config["lr_schedule"])
        config["activation_delay"] = config.get("activation_delay", default_config["activation_delay"])
        config["network_params"] = config.get("network_params", default_config["network_params"])
        self.config = config

    def load_optimizer(self) -> None:
        """
        Load optimizer function for the model.
        """
        # Load optimizer
        self.config["optimizer"] = Optimizer.get_optimizer(self.config["optimizer"], self.config["lr"],
                                                           self.config.get("weight_decay", self.get_default_config()["weight_decay"]), self.model)

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
        optimizer = self.config["optimizer"]
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        mask_unlabeled = self.config["mask_unlabeled"]
        early_stopping = self.config["early_stopping"]
        activation_delay = self.config["activation_delay"]
        if "scheduler" in self.config:
            scheduler = self.config["scheduler"]
        else:
            scheduler = None

        if early_stopping > 0:
            logger.info(
                f"--- Early stopping enabled with patience of {early_stopping} epochs.")
        # Use the optimal threshold when the threshold is set to a negative value
        use_optimal_threshold = self.config["threshold"] < 0

        # Only copy if we need to find the optimal threshold later
        if use_optimal_threshold:
            X_train_start = X_train.copy()
            Y_train_start = y_train.copy()

        X_train = torch.from_numpy(X_train).permute(0, 2, 1)
        X_test = torch.from_numpy(X_test).permute(0, 2, 1)

        # Get only the 2 event state features
        if mask_unlabeled:
            y_train = y_train[:, :, np.array(
                [data_info.y_columns["awake"], data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]])]
            y_test = y_test[:, :, np.array(
                [data_info.y_columns["awake"], data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]])]
        else:
            y_train = y_train[:, :, np.array(
                [data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]])]
            y_test = y_test[:, :, np.array(
                [data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]])]
        y_train = torch.from_numpy(y_train).permute(0, 2, 1)
        y_test = torch.from_numpy(y_test).permute(0, 2, 1)

        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        # Print the shapes and types of train and test
        logger.info(
            f"--- X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(
            f"--- X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logger.info(
            f"--- X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")
        logger.info(
            f"--- X_test type: {X_test.dtype}, y_test type: {y_test.dtype}")

        # Create a dataloader from the dataset
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size)

        # Train the model
        logger.info("--- Training model " + self.name)
        trainer = EventTrainer(
            epochs, criterion, mask_unlabeled, early_stopping)
        avg_losses, avg_val_losses, total_epochs = trainer.fit(
            trainloader=train_dataloader, testloader=test_dataloader, model=self.model, optimizer=optimizer, name=self.name, scheduler=scheduler,
            activation_delay=activation_delay)

        # Log full train and test plot
        if wandb.run is not None:
            self.log_train_test(avg_losses, avg_val_losses, len(avg_losses))
        logger.info("--- Training of model complete!")

        # Set total_epochs in config if broken by the early stopping
        self.config["total_epochs"] = total_epochs

        # Find optimal threshold if necessary
        if use_optimal_threshold:
            logger.info("--- Finding optimal threshold for model.")
            self.find_optimal_threshold(X_train_start, Y_train_start)
            logger.info(
                f"--- Optimal threshold is {self.config['threshold']:.4f}.")
        else:
            logger.info(
                f"--- Using threshold of {self.config['threshold']:.4f} from the config.")

    def train_full(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model on the full dataset.
        :param X_train: the training data
        :param y_train: the training labels
        """
        criterion = self.config["loss"]
        optimizer = self.config["optimizer"]
        epochs = self.config["total_epochs"]
        batch_size = self.config["batch_size"]
        mask_unlabeled = self.config["mask_unlabeled"]
        activation_delay = self.config["activation_delay"]
        if "scheduler" in self.config:
            scheduler = self.config["scheduler"]
        else:
            scheduler = None

        logger.info("--- Running for " + str(epochs) + " epochs.")

        X_train = torch.from_numpy(X_train).permute(0, 2, 1)

        if mask_unlabeled:
            y_train = y_train[:, :, np.array(
                [data_info.y_columns["awake"], data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]])]
        else:
            y_train = y_train[:, :, np.array(
                [data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]])]
        y_train = torch.from_numpy(y_train).permute(0, 2, 1)

        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

        # Print the shapes and types of train and test
        logger.info(
            f"--- X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(
            f"--- X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")

        # Create a dataloader from the dataset
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size)

        # Train the model
        logger.info("--- Training model full " + self.name)
        trainer = EventTrainer(epochs, criterion, mask_unlabeled, -1)
        avg_losses, avg_val_losses, total_epochs = trainer.fit(trainloader=train_dataloader, testloader=None,
                                                               model=self.model, optimizer=optimizer, name=self.name, scheduler=scheduler, activation_delay=activation_delay)

        logger.info("--- Full train complete!")

        # Log the results to wandb
        if wandb.run is not None:
            self.log_train_test(avg_losses[:total_epochs], avg_val_losses[:total_epochs], total_epochs)

    def pred(self, data: np.ndarray, pred_with_cpu: bool) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Prediction function for the model.
        :param data: unlabeled data (step, features)
        :param pred_with_cpu: whether to predict with cpu or gpu
        :return: the predictions in format: (predictions, confidences)
        """
        # Prediction function
        logger.info(f"--- Predicting results with model {self.name}")
        # Run the model on the data and return the predictions

        if pred_with_cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # Set model to eval for inference
        self.model.eval()

        self.model.to(device)

        # Print data shape
        logger.info(f"--- Data shape of predictions dataset: {data.shape}")

        # Create a DataLoader for batched inference
        dataset = TensorDataset(torch.from_numpy(data).permute(0, 2, 1))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch_data in tqdm(dataloader, "Predicting", unit="batch"):
                batch_data = batch_data[0].to(device)

                # Make a batch prediction
                batch_prediction = self.model(batch_data)

                if pred_with_cpu:
                    batch_prediction = batch_prediction.numpy()
                else:
                    batch_prediction = batch_prediction.cpu().numpy()

                predictions.append(batch_prediction)

        # Concatenate the predictions from all batches
        predictions = np.concatenate(predictions, axis=0)

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

    def evaluate(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Evaluation function for the model.
        :param pred: predictions
        :param target: targets
        :return: avg loss of predictions
        """
        # Evaluate function
        logger.info("--- Evaluating model")
        # Calculate the loss of the predictions
        criterion = self.config["loss"]
        loss = criterion(pred, target)
        return loss

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

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.reset_optimizer()
        self.reset_scheduler()
        logger.info("Model fully loaded from: " + path)

    def find_optimal_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        """Finds and sets the optimal threshold for the model.

        :param X: the data with shape (window, window_size, n_features)
        :param y: the labels with shape (window, window_size, n_features)
        :return: the optimal threshold ∈ [0, 1]
        """
        self.config["threshold"] = -10000
        y_pred = self.pred(X, False)

        # Get the sum of the confidences
        confidences_sum = np.sum(y_pred[1], axis=1)

        # Make an array where it is true if awake is 0 or 1 and false if awake is 2
        make_pred_windowed = np.where(
            y[:, :, data_info.y_columns['awake']] == 2, False, True)

        # Get a single boolean for each window if it should make a prediction or not
        make_pred = np.any(make_pred_windowed, axis=1)

        # Create ROC curve
        fpr, tpr, thresholds = roc_curve(make_pred, confidences_sum)

        # Get optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        self.config["threshold"] = optimal_threshold
        return self.config["threshold"]

    def reset_weights(self) -> None:
        """
        Reset the weights of the model.
        """
        self.model = SegUnet1D(in_channels=len(data_info.X_columns), window_size=data_info.window_size, out_channels=2,
                               model_type=self.model_type, **self.load_network_params(self.config))

    def reset_scheduler(self) -> None:
        """
        Reset the scheduler to the initial state. Useful for retraining the model.
        """
        if 'scheduler' in self.config:
            self.config['scheduler'] = CosineLRScheduler(self.config['optimizer'], **self.config["lr_schedule"])

    def reset_optimizer(self) -> None:

        """
        Reset the optimizer to the initial state. Useful for retraining the model.
        """
        self.config['optimizer'] = type(self.config['optimizer'])(self.model.parameters(), lr=self.config['lr'])
