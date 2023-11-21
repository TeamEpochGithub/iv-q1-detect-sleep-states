import copy

import numpy as np
import pandas as pd
import torch
import wandb
from timm.scheduler import CosineLRScheduler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from src.models.trainers.event_state_trainer import EventStateTrainer

from src.loss.loss import Loss
from src.models.model_exception import ModelException
from src.models.trainers.event_trainer import EventTrainer
from src.optimizer.optimizer import Optimizer
from src.util.state_to_event import pred_to_event_state
from .. import data_info
from ..logger.logger import logger
from ..util.hash_config import hash_config


class EventModel:
    """
    Model class with basic methods for training and evaluation. This class should be overwritten by the user.
    """

    def __init__(self, config: dict, name: str) -> None:
        self.model_type = "base-model"
        self.model = None
        # Init function
        if config is None:
            self.config = None
        else:
            # Deepcopy config
            self.config = copy.deepcopy(config)
            self.hash = hash_config(config, length=5)

        self.name = name

    def get_type(self) -> str:
        """
        Get type function for the model.
        :return: type of the model
        """
        return self.model_type

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
        config["use_auxiliary_awake"] = config.get(
            "use_auxiliary_awake", default_config["use_auxiliary_awake"])
        config["mask_unlabeled"] = config.get(
            "mask_unlabeled", default_config["mask_unlabeled"])
        if config["mask_unlabeled"]:
            config["loss"] = Loss.get_loss(config["loss"], reduction="none")
        else:
            if config["loss"] == "kldiv-torch":
                config["loss"] = Loss.get_loss(
                    config["loss"], reduction="batchmean")
            else:
                config["loss"] = Loss.get_loss(
                    config["loss"], reduction="mean")
        config["batch_size"] = config.get(
            "batch_size", default_config["batch_size"])
        config["lr"] = config.get("lr", default_config["lr"])
        config["optimizer"] = Optimizer.get_optimizer(
            config["optimizer"], config["lr"], 0, self.model)
        if "lr_schedule" in config:
            config["lr_schedule"] = config.get(
                "lr_schedule", default_config["lr_schedule"])
            config["scheduler"] = CosineLRScheduler(
                config["optimizer"], **self.config["lr_schedule"])
        config["epochs"] = config.get("epochs", default_config["epochs"])
        config["early_stopping"] = config.get(
            "early_stopping", default_config["early_stopping"])
        config["activation_delay"] = config.get(
            "activation_delay", default_config["activation_delay"])
        config["network_params"] = config.get("network_params", dict())
        config["threshold"] = config.get(
            "threshold", default_config["threshold"])
        self.config = config

    def get_default_config(self) -> dict:
        """
        Get default config function for the model. This function should be overwritten by the user.
        :return: default config
        """
        logger.info("--- No default configuration of model or not implemented")
        return {}

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
        if "scheduler" in self.config:
            scheduler = self.config["scheduler"]
        else:
            scheduler = None
        early_stopping = self.config["early_stopping"]
        activation_delay = self.config["activation_delay"]
        use_auxiliary_awake = self.config["use_auxiliary_awake"]
        if early_stopping > 0:
            logger.info(
                f"--- Early stopping enabled with patience of {early_stopping} epochs.")

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)

        # Get only the 2 event state features
        labels_list = [data_info.y_columns["state-onset"],
                       data_info.y_columns["state-wakeup"]]
        if mask_unlabeled:
            # Add awake label to front of the list
            labels_list.insert(0, data_info.y_columns["awake"])
        if use_auxiliary_awake:
            # Add awake label to end of the list
            labels_list.append(data_info.y_columns["awake"])
        labels_list = np.array(labels_list)

        y_train = torch.from_numpy(y_train[:, :, labels_list])
        y_test = torch.from_numpy(y_test[:, :, labels_list])

        # Turn last column into one hot encoding of awake so that it can be used as auxiliary awake
        if use_auxiliary_awake:
            # Change all 3's for last column to 2's
            y_train[:, :, -1] = torch.where(
                y_train[:, :, -1] == 3, torch.tensor(2), y_train[:, :, -1])
            y_test[:, :, -1] = torch.where(
                y_test[:, :, -1] == 3, torch.tensor(2), y_test[:, :, -1])

            awake = y_train[:, :, -1]
            awake = torch.nn.functional.one_hot(awake.to(torch.int64))
            y_train = torch.cat((y_train[:, :, :-1], awake.float()), dim=2)

            awake = y_test[:, :, -1]
            awake = torch.nn.functional.one_hot(awake.to(torch.int64))
            y_test = torch.cat((y_test[:, :, :-1], awake.float()), dim=2)

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

        if use_auxiliary_awake:
            trainer = EventStateTrainer(
                epochs, criterion, early_stopping=early_stopping, mask_unlabeled=mask_unlabeled)
        else:
            trainer = EventTrainer(
                epochs, criterion, early_stopping=early_stopping, mask_unlabeled=mask_unlabeled)
        avg_losses, avg_val_losses, total_epochs = trainer.fit(
            trainloader=train_dataloader, testloader=test_dataloader, model=self.model, optimizer=optimizer, name=self.name, scheduler=scheduler,
            activation_delay=activation_delay)

        if wandb.run is not None:
            self.log_train_test(
                avg_losses[:total_epochs], avg_val_losses[:total_epochs], total_epochs)

        logger.info("--- Training of model complete!")
        self.config["total_epochs"] = total_epochs

    def train_full(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
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
        if "scheduler" in self.config:
            scheduler = self.config["scheduler"]
        else:
            scheduler = None
        early_stopping = self.config["early_stopping"]
        activation_delay = self.config["activation_delay"]
        use_auxiliary_awake = self.config["use_auxiliary_awake"]
        if early_stopping > 0:
            logger.info(
                f"--- Early stopping enabled with patience of {early_stopping} epochs.")

        x_train = torch.from_numpy(x_train)

        # Get only the 2 event state features
        labels_list = [data_info.y_columns["state-onset"],
                       data_info.y_columns["state-wakeup"]]
        if mask_unlabeled:
            # Add awake label to front of the list
            labels_list.insert(0, data_info.y_columns["awake"])
        if use_auxiliary_awake:
            # Add awake label to end of the list
            labels_list.append(data_info.y_columns["awake"])
        labels_list = np.array(labels_list)

        y_train = torch.from_numpy(y_train[:, :, labels_list])

        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

        # Print the shapes and types of train and test
        logger.info(
            f"--- X_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        logger.info(
            f"--- X_train type: {x_train.dtype}, y_train type: {y_train.dtype}")

        # Create a dataloader from the dataset
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size)

        if use_auxiliary_awake:
            trainer = EventStateTrainer(
                epochs, criterion, early_stopping=early_stopping, mask_unlabeled=mask_unlabeled)
        else:
            trainer = EventTrainer(
                epochs, criterion, early_stopping=early_stopping, mask_unlabeled=mask_unlabeled)
        trainer.fit(
            trainloader=train_dataloader, testloader=None, model=self.model, optimizer=optimizer, name=self.name, scheduler=scheduler,
            activation_delay=activation_delay)
        logger.info("Full train complete!")

    def pred(self, data: np.ndarray, pred_with_cpu: bool, raw_output: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Prediction function for the model.
        :param data: unlabelled data
        :param pred_with_cpu: whether to use cpu or gpu
        :return: the predictions and confidences, as numpy arrays
        """
        # Prediction function
        logger.info(f"--- Predicting results with model {self.name}")
        # Run the model on the data and return the predictions

        if pred_with_cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        self.model.eval()
        self.model.to(device)

        # Print data shape
        logger.info(f"--- Data shape of predictions dataset: {data.shape}")

        # Create a DataLoader for batched inference
        dataset = TensorDataset(torch.from_numpy(data))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch_data in tqdm(dataloader, "Predicting", unit="batch"):
                batch_data = batch_data[0].to(device)

                # Make a batch prediction
                if str(self.model).startswith("MultiResidualBiGRU"):
                    batch_prediction, _ = self.model(batch_data)
                else:
                    batch_prediction = self.model(batch_data)

                # If auxiliary awake is used, take only the first 2 columns
                if self.config["use_auxiliary_awake"]:
                    batch_prediction = batch_prediction[:, :, :2]

                if pred_with_cpu:
                    batch_prediction = batch_prediction.numpy()
                else:
                    batch_prediction = batch_prediction.cpu().numpy()

                predictions.append(batch_prediction)

        # Concatenate the predictions from all batches
        predictions = np.concatenate(predictions, axis=0)

        # # Apply upsampling to the predictions
        downsampling_factor = data_info.downsampling_factor

        # TODO Try other interpolation methods (linear / cubic)
        # if downsampling_factor > 1:
        #     predictions = np.repeat(predictions, downsampling_factor, axis=1)

        # # Define the original time points
        # original_time_points = np.linspace(0, 1, data_info.window_size)
        #
        # # Define the new time points for upsampled data
        # upsampled_time_points = np.linspace(0, 1, data_info.window_size_before)
        #
        # # Create an array to store upsampled data
        # upsampled_data = np.zeros((predictions.shape[0], data_info.window_size_before, predictions.shape[2]))
        #
        # # Apply interpolation along axis=1 for each channel
        # for channel_idx in range(predictions.shape[2]):
        #     for row_idx in range(predictions.shape[0]):
        #         interpolation_function = interp1d(original_time_points, predictions[row_idx, :, channel_idx], kind='linear', fill_value='extrapolate')
        #         upsampled_data[row_idx, :, channel_idx] = interpolation_function(upsampled_time_points)

        steps_sinc = np.arange(0, data_info.window_size_before, data_info.downsampling_factor)
        u_sinc = np.arange(0, data_info.window_size_before, 1)

        upsampled_data = np.zeros((predictions.shape[0], data_info.window_size_before, predictions.shape[2]))

        # Find the period
        T = steps_sinc[1] - steps_sinc[0]
        # Use broadcasting correctly
        sincM = (u_sinc - steps_sinc[:, np.newaxis]) / T
        res_sinc = np.sinc(sincM)

        for channel_idx in range(predictions.shape[2]):
            for row_idx in tqdm(range(predictions.shape[0]), "Upsampling using sinc interpolation", unit="window"):
                y_sinc = np.dot(predictions[row_idx, :, channel_idx], res_sinc)
                upsampled_data[row_idx, :, channel_idx] = y_sinc

        predictions = upsampled_data

        # Return raw output if necessary
        if raw_output:
            return predictions

        all_predictions = []
        all_confidences = []
        # Convert to events
        for pred in tqdm(predictions, desc="Converting predictions to events", unit="window"):
            # Pred should be 2d array with shape (window_size, 2)
            assert pred.shape[
                       1] == 2, "Prediction should be 2d array with shape (window_size, 2)"

            # Convert to relative window event timestamps
            events = pred_to_event_state(pred, thresh=self.config["threshold"])

            # Add step offset based on repeat factor.
            if downsampling_factor > 1:
                offset = ((downsampling_factor / 2.0) - 0.5 if downsampling_factor % 2 == 0 else downsampling_factor // 2)
            else:
                offset = 0
            steps = (events[0] + offset, events[1] + offset)
            confidences = (events[2], events[3])
            all_predictions.append(steps)
            all_confidences.append(confidences)

        # Return numpy array
        return np.array(all_predictions), np.array(all_confidences)

    def evaluate(self, pred: pd.DataFrame, target: pd.DataFrame) -> float:
        """
        Evaluation function for the model. This function should be overwritten by the user.
        :param pred: predictions
        :param target: actual labels
        """
        # Evaluate function
        logger.critical(
            "--- Evaluation of base class called. Did you forget to override it?")
        raise ModelException(
            "Evaluation of base class called. Did you forget to override it?")

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

    def reset_weights(self) -> None:
        """
        Reset the weights of the model. Useful for retraining the model. This function should be overwritten by the user.
        """
        logger.critical(
            "--- Resetting weights of base class called. Did you forget to override it?")
        raise ModelException(
            "Resetting weights of base class called. Did you forget to override it?")

    def reset_optimizer(self) -> None:
        """
        Reset the optimizer to the initial state. Useful for retraining the model.
        """
        self.config['optimizer'] = type(self.config['optimizer'])(
            self.model.parameters(), lr=self.config['lr'])

    def reset_scheduler(self) -> None:
        """
        Reset the scheduler to the initial state. Useful for retraining the model.
        """
        if 'scheduler' in self.config:
            self.config['scheduler'] = CosineLRScheduler(
                self.config['optimizer'], **self.config["lr_schedule"])

    def log_train_test(self, avg_losses: list | np.ndarray, avg_val_losses: list | np.ndarray, epochs: int, name: str = "") -> None:
        """
        Log the train and test loss to wandb.
        :param avg_losses: list of average train losses
        :param avg_val_losses: list of average test losses
        :param epochs: number of epochs
        """
        log_dict = {
            'epoch': list(range(epochs)),
            'train_loss': avg_losses,
            'val_loss': avg_val_losses
        }
        log_df = pd.DataFrame(log_dict)
        # Convert to a long format
        long_df = pd.melt(
            log_df, id_vars=['epoch'], var_name='loss_type', value_name='loss')

        table = wandb.Table(dataframe=long_df)

        # Field to column in df
        fields = {"step": "epoch", "lineVal": "loss", "lineKey": "loss_type"}
        custom_plot = wandb.plot_table(
            vega_spec_name="team-epoch-iv/trainval",
            data_table=table,
            fields=fields,
            string_fields={"title": data_info.substage + " - Train and validation loss of model " + self.name + "_" + name}
        )
        if wandb.run is not None:
            wandb.log({f"{data_info.substage, name}": custom_plot})
