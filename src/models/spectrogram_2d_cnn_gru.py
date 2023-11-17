import copy
from typing import Any

import numpy as np
import torch
import wandb
from numpy import ndarray, dtype
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .architectures.spectrogram_encoder_decoder import SpectrogramEncoderDecoder
from .model import Model, ModelException
from .trainers.event_trainer import EventTrainer
from .. import data_info
from ..logger.logger import logger
from ..loss.loss import Loss
from ..optimizer.optimizer import Optimizer
from timm.scheduler import CosineLRScheduler
from ..util.state_to_event import pred_to_event_state
from .architectures.spectrogram_cnn_gru import MultiResidualBiGRUwSpectrogramCNN


class EventSegmentation2DCNNGRU(Model):
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

        self.model_type = "Spectrogram_2D_Cnn"

        # We load the model architecture here. 2 Out channels, one for onset, one for offset event state prediction
        if self.config.get("use_awake_channel", False):
            self.model = MultiResidualBiGRUwSpectrogramCNN(in_channels=len(data_info.X_columns),
                                           out_channels=3, model_type=self.model_type, config=self.config)
        else:
            self.model = MultiResidualBiGRUwSpectrogramCNN(in_channels=len(data_info.X_columns),
                                           out_channels=2, model_type=self.model_type, config=self.config)
        data_info.downsampling_factor = self.config.get('hop_length', 1)
        data_info.window_size = 17280//data_info.downsampling_factor
        # Load config
        self.load_config(config)
        # Print model summary
        if wandb.run is not None:
            if data_info.plot_summary:
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
        config["hidden_layers"] = config.get(
            "hidden_layers", default_config["hidden_layers"])
        config["kernel_size"] = config.get(
            "kernel_size", default_config["kernel_size"])
        config["depth"] = config.get("depth", default_config["depth"])
        config["early_stopping"] = config.get(
            "early_stopping", default_config["early_stopping"])
        config["threshold"] = config.get(
            "threshold", default_config["threshold"])
        config["weight_decay"] = config.get(
            "weight_decay", default_config["weight_decay"])

        self.config = config
        self.config["optimizer"] = Optimizer.get_optimizer(
            self.config["optimizer"], self.config["lr"], self.config["weight_decay"], self.model)
        if "lr_schedule" in self.config:
            config["scheduler"] = CosineLRScheduler(config["optimizer"], **self.config["lr_schedule"])

    def get_default_config(self) -> dict:
        """
        Get default config function for the model.
        :return: default config
        """
        return {
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 10,
            "hidden_layers": 32,
            "kernel_size": 7,
            "depth": 2,
            "early_stopping": -1,
            "threshold": 0.5,
            "weight_decay": 0.0,
            "mask_unlabeled": False
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

        # make sure to put enmo and anglez first in the feature order
        desired_feature_order = ['f_enmo', 'f_anglez'] + [feature for feature in data_info.X_columns if feature not in ['f_enmo', 'f_anglez']]
        # get the indices of the desired feature order
        desired_feature_order_indices = np.array([data_info.X_columns[feature] for feature in desired_feature_order])
        # now use the indices to reorder the X_train and X_test
        X_train = X_train[:, desired_feature_order_indices, :]
        X_test = X_test[:, desired_feature_order_indices, :]

        # make a list to get which indices to use for the y data
        if mask_unlabeled:
            index_list = [data_info.y_columns["awake"], data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]]
        else:
            index_list = [data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]]

        # add the awake column again if you want to use the awake channel
        if self.config.get('use_awake_channel', False):
            index_list.append(data_info.y_columns["awake"])

        # Index the y data with the index list
        y_train = y_train[:, :, np.array(index_list)]
        y_test = y_test[:, :, np.array(index_list)]

        # if clip awake clip the awake columns values
        if self.config.get('clip_awake', False):
            if y_train.shape[2] == 3:
                y_train[:, :, 2] = np.clip(y_train[:, :, 2], 0, 1)
                y_test[:, :, 2] = np.clip(y_test[:, :, 2], 0, 1)
            elif y_train.shape[2] == 4:
                y_train[:, :, 3] = np.clip(y_train[:, :, 3], 0, 1)
                y_test[:, :, 3] = np.clip(y_test[:, :, 3], 0, 1)

        # our pretrain downsampling puts the median of 12 items into one item
        # this loop also does that
        y_train_downsampled = []
        for i in range(y_train.shape[0]):
            downsampled_channels = []
            for j in range(y_train.shape[2]):
                downsampled_channels.append(np.median(y_train[i, :, j].reshape(-1, self.config.get('hop_length', 1)), axis=1))
            y_train_downsampled.append(np.array(downsampled_channels))
        y_train = torch.from_numpy(np.array(y_train_downsampled)).permute(0, 2, 1)
        del y_train_downsampled

        # same loop as above to downsample the test data
        y_test_downsampled = []
        for i in range(y_test.shape[0]):
            downsampled_channels = []
            for j in range(y_test.shape[2]):
                downsampled_channels.append(np.median(y_test[i, :, j].reshape(-1, self.config.get('hop_length', 1)), axis=1))
            y_test_downsampled.append(np.array(downsampled_channels))
        y_test = torch.from_numpy(np.array(y_test_downsampled)).permute(0, 2, 1)
        del y_test_downsampled

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

        # Create a custom dataset class to apply the spectrogram to the data

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
            activation_delay=self.config.get('activation_delay', 10))

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
        if "scheduler" in self.config:
            scheduler = self.config["scheduler"]
        else:
            scheduler = None

        logger.info("--- Running for " + str(epochs) + " epochs.")

        X_train = torch.from_numpy(X_train).permute(0, 2, 1)

        if mask_unlabeled:
            index_list = [data_info.y_columns["awake"], data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]]
        else:
            index_list = [data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]]

        # add the awake column again if you want to use the awake channel
        if self.config.get('use_awake_channel', False):
            index_list.append(data_info.y_columns["awake"])

        # Index the y data with the index list
        y_train = y_train[:, :, np.array(index_list)]

        # if clip awake clip the awake columns values
        if self.config.get('clip_awake', False):
            if y_train.shape[2] == 3:
                y_train[:, :, 2] = np.clip(y_train[:, :, 2], 0, 1)
            elif y_train.shape[2] == 4:
                y_train[:, :, 3] = np.clip(y_train[:, :, 3], 0, 1)

        # downsample the y data
        y_train_downsampled = []
        for i in range(y_train.shape[0]):
            downsampled_channels = []
            for j in range(y_train.shape[2]):
                downsampled_channels.append(np.median(y_train[i, :, j].reshape(-1, self.config.get('hop_length', 1)), axis=1))
            y_train_downsampled.append(np.array(downsampled_channels))
        y_train = torch.from_numpy(np.array(y_train_downsampled))
        del y_train_downsampled

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
        trainer.fit(trainloader=train_dataloader, testloader=None,
                    model=self.model, optimizer=optimizer, name=self.name, scheduler=scheduler)

        logger.info("--- Full train complete!")

    def pred(self, data: np.ndarray, pred_with_cpu: bool, raw_output: bool = False) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
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
        self.model.eval()
        self.model.to(device)
        # Print data shape
        logger.info(f"--- Data shape of predictions dataset: {data.shape}")

        # Create a DataLoader for batched inference
        dataset = TensorDataset(torch.from_numpy(data).permute(0, 2, 1))
        dataloader = DataLoader(dataset, batch_size=self.config.get('batch_size', 1), shuffle=False)

        # Onset predictions
        predictions = []
        with torch.no_grad():
            for batch_data in tqdm(dataloader, "Predicting", unit="batch"):
                batch_data = batch_data[0].to(device)

                # Make a batch prediction
                batch_prediction, _ = self.model(batch_data)

                if pred_with_cpu:
                    batch_prediction = batch_prediction.numpy()
                else:
                    batch_prediction = batch_prediction.cpu().numpy()

                predictions.append(batch_prediction)

        # Concatenate the predictions from all batches for onset
        predictions = np.concatenate(predictions, axis=0)

        # Apply upsampling to the predictions
        if self.config.get('hop_length', 1) > 1:
            predictions = np.repeat(
                predictions, self.config.get('hop_length', 1), axis=1)
        y_test = np.load('y_test.npy')
        if raw_output:
            return predictions

        all_predictions = []
        all_confidences = []
        # Convert to events
        for pred in tqdm(predictions, desc="Converting predictions to events", unit="window"):
            # Convert to relative window event timestamps
            if pred.shape[1] == 3:
                events = pred_to_event_state(pred[:, :-1], thresh=self.config["threshold"])
            else:
                events = pred_to_event_state(pred, thresh=self.config["threshold"])
            # Add step offset based on repeat factor.
            if self.config.get('hop_length', 1) <= 1:
                offset = 0
            elif self.config.get('hop_length', 1) % 2 == 0:
                offset = (self.config.get('hop_length', 1) / 2.0) - 0.5
            else:
                offset = self.config.get('hop_length', 1) // 2
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
            logger.info(
                "Loading hyperparameters and instantiate new model from: " + path)
            return

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.reset_optimizer()
        logger.info("Model fully loaded from: " + path)

    def reset_optimizer(self) -> None:
        """
        Reset the optimizer to the initial state. Useful for retraining the model.
        """
        self.config['optimizer_onset'] = type(self.config['optimizer'])(
            self.model.parameters(), lr=self.config['optimizer'].param_groups[0]['lr'])

    def reset_scheduler(self) -> None:
        """
        Reset the scheduler to the initial state. Useful for retraining the model.
        """
        if 'scheduler' in self.config:
            self.config['scheduler'] = CosineLRScheduler(self.config['optimizer'], **self.config["lr_schedule"])

    def reset_weights(self) -> None:
        """
        Reset the weights of the model. Useful for retraining the model.
        """
        if self.config.get("use_awake_channel", False):
            self.model = MultiResidualBiGRUwSpectrogramCNN(in_channels=len(data_info.X_columns),
                                            out_channels=3, model_type=self.model_type, config=self.config)
        else:
            self.model = MultiResidualBiGRUwSpectrogramCNN(in_channels=len(data_info.X_columns), 
                                           out_channels=2, model_type=self.model_type, config=self.config)
