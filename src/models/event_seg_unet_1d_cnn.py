import copy
from typing import Any

import numpy as np
import torch
import wandb
from numpy import ndarray, dtype
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .architectures.seg_unet_1d_cnn import SegUnet1D
from ..logger.logger import logger
from ..loss.loss import Loss
from ..models.model import Model, ModelException
from ..optimizer.optimizer import Optimizer
from ..util.state_to_event import pred_to_event_state


class EventSegmentationUnet1DCNN(Model):
    """
    This model is a event segmentation model based on the Unet 1D CNN. It uses the architecture from the SegSimple1DCNN class.
    """

    def __init__(self, config: dict, data_shape: tuple, name: str) -> None:
        """
        Init function of the example model
        :param config: configuration to set up the model
        :param data_shape: shape of the X data (channels, window_size)
        :param name: name of the model
        """
        super().__init__(config, name)

        # Check if gpu is available, else return an exception
        if not torch.cuda.is_available():
            logger.warning("GPU not available - using CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            logger.info(f"--- Device set to model {self.name}: " + torch.cuda.get_device_name(0))

        self.model_type = "event-segmentation"
        self.data_shape = data_shape

        # Load config
        self.load_config(config)

        # We load the model architecture here. 2 Out channels, one for onset, one for offset event state prediction
        self.model = SegUnet1D(in_channels=data_shape[0], window_size=data_shape[1], out_channels=2, model_type=self.model_type, config=self.config)

        # Load optimizer
        self.load_optimizer()

        # Print model summary
        if wandb.run is not None:
            from torchsummary import summary
            summary(self.model.cuda(), input_size=(data_shape[0], data_shape[1]))

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
                logger.critical("------ Config is missing required parameter: " + req)
                raise ModelException("Config is missing required parameter: " + req)

        # Get default_config
        default_config = self.get_default_config()
        config["loss"] = Loss.get_loss(config["loss"])
        config["batch_size"] = config.get("batch_size", default_config["batch_size"])
        config["epochs"] = config.get("epochs", default_config["epochs"])
        config["lr"] = config.get("lr", default_config["lr"])
        config["hidden_layers"] = config.get("hidden_layers", default_config["hidden_layers"])
        config["kernel_size"] = config.get("kernel_size", default_config["kernel_size"])
        config["depth"] = config.get("depth", default_config["depth"])
        config["early_stopping"] = config.get("early_stopping", default_config["early_stopping"])
        config["threshold"] = config.get("threshold", default_config["threshold"])
        config["weight_decay"] = config.get("weight_decay", default_config["weight_decay"])
        self.config = config

    def load_optimizer(self) -> None:
        """
        Load optimizer function for the model.
        """
        # Load optimizer
        self.config["optimizer"] = Optimizer.get_optimizer(self.config["optimizer"], self.config["lr"], self.config["weight_decay"], self.model)

    def get_default_config(self) -> dict:
        """
        Get default config function for the model.
        :return: default config
        """
        return {"batch_size": 32, "lr": 0.001, "epochs": 10, "hidden_layers": 32, "kernel_size": 7, "depth": 2, "early_stopping": -1, "threshold": 0.5, "weight_decay": 0.0}

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
        early_stopping = self.config["early_stopping"]
        if early_stopping > 0:
            logger.info(f"--- Early stopping enabled with patience of {early_stopping} epochs.")

        # TODO Change
        X_train = torch.from_numpy(X_train[:, :, :]).permute(0, 2, 1)
        X_test = torch.from_numpy(X_test[:, :, :]).permute(0, 2, 1)

        # Get only the 2 event state features
        y_train = y_train[:, :, -2:]
        y_test = y_test[:, :, -2:]
        y_train = torch.from_numpy(y_train).permute(0, 2, 1)
        y_test = torch.from_numpy(y_test).permute(0, 2, 1)

        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        # Print the shapes and types of train and test
        logger.info(f"--- X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"--- X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logger.info(f"--- X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")
        logger.info(f"--- X_test type: {X_test.dtype}, y_test type: {y_test.dtype}")

        # Create a dataloader from the dataset
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        # Add model and data to device cuda
        # self.model.half()
        self.model.to(self.device)

        # Define wandb metrics
        if wandb.run is not None:
            wandb.define_metric("epoch")
            wandb.define_metric(f"Train {str(criterion)} of {self.name}", step_metric="epoch")
            wandb.define_metric(f"Validation {str(criterion)} of {self.name}", step_metric="epoch")

        # Initialize place holder arrays for train and test loss and early stopping
        total_epochs = 0
        avg_losses = []
        avg_val_losses = []
        counter = 0
        lowest_val_loss = np.inf
        best_model = self.model.state_dict()
        stopped = False

        # Train the model
        for epoch in range(epochs):
            self.model.train()
            avg_loss = 0
            avg_val_loss = 0
            total_batch_loss = 0
            total_val_batch_loss = 0
            # Train loop
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for i, (x, y) in enumerate(tepoch):
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                    # Clear gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(x)
                    loss = criterion(outputs, y)

                    # Backward and optimize
                    loss.backward()
                    optimizer.step()

                    # Get the current loss
                    current_loss = loss.item()
                    total_batch_loss += current_loss
                    avg_loss = total_batch_loss / (i + 1)

                    # Log to console
                    tepoch.set_description(f" Train Epoch {epoch}")
                    tepoch.set_postfix(loss=avg_loss)

            # Calculate the validation loss and set the model to eval
            self.model.eval()

            with torch.no_grad():
                with tqdm(test_dataloader, unit="batch") as vepoch:
                    for i, (vx, vy) in enumerate(vepoch):
                        vx = vx.to(self.device)
                        vy = vy.to(self.device)
                        voutputs = self.model(vx)
                        vloss = criterion(voutputs, vy)

                        current_loss = vloss.item()
                        total_val_batch_loss += current_loss
                        avg_val_loss = total_val_batch_loss / (i + 1)

                        vepoch.set_description(f" Test  Epoch {epoch}")
                        vepoch.set_postfix(loss=avg_val_loss)

            # Print the avg training and validation loss of 1 epoch in a clean way.
            descr = f"------ Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            logger.debug(descr)

            # Add average losses and epochs to list
            avg_losses.append(avg_loss)
            avg_val_losses.append(avg_val_loss)
            total_epochs += 1

            # Log train test loss to wandb
            if wandb.run is not None:
                wandb.log({f"Train {str(criterion)} of {self.name}": avg_loss, f"Validation {str(criterion)} of {self.name}": avg_val_loss, "epoch": epoch})

            # Early stopping
            if early_stopping > 0:
                # Save model if validation loss is lower than previous lowest validation loss
                if avg_val_loss < lowest_val_loss:
                    lowest_val_loss = avg_val_loss
                    best_model = self.model.state_dict()
                    counter = 0
                else:
                    counter += 1
                    if counter >= early_stopping:
                        logger.info("--- Patience reached of " + str(early_stopping) + " epochs. Current epochs run = " + str(
                            total_epochs) + " Stopping training and loading best model for " + str(total_epochs - early_stopping) + ".")
                        self.model.load_state_dict(best_model)
                        stopped = True
                        break

        # Log full train and test plot
        if wandb.run is not None:
            self.log_train_test(avg_losses, avg_val_losses, total_epochs)
        logger.info("--- Training of model complete!")

        # Set total_epochs in config if broken by the early stopping
        if stopped:
            total_epochs -= early_stopping
        self.config["total_epochs"] = total_epochs

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

        logger.info("--- Running for " + str(epochs) + " epochs.")

        X_train = torch.from_numpy(X_train).permute(0, 2, 1)

        # Get only the event state features
        y_train = y_train[:, :, -2:]
        y_train = torch.from_numpy(y_train).permute(0, 2, 1)
        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

        # Print the shapes and types of train and test
        logger.info(f"--- X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"--- X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")

        # Create a dataloader from the dataset
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

        # Add model and data to device cuda
        # self.model.half()
        self.model.to(self.device)

        # Define wandb metrics
        if wandb.run is not None:
            wandb.define_metric("epoch")
            wandb.define_metric(f"Train {str(criterion)} on whole dataset of {self.name}", step_metric="epoch")

        for epoch in range(epochs):
            self.model.train()
            total_batch_loss = 0
            avg_loss = 0
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for i, (x, y) in enumerate(tepoch):
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                    # Clear gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(x)
                    loss = criterion(outputs, y)

                    # Backward and optimize
                    loss.backward()
                    optimizer.step()

                    # Get the current loss
                    current_loss = loss.item()
                    total_batch_loss += current_loss
                    avg_loss = total_batch_loss / (i + 1)

                    # Log to console
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(loss=avg_loss)

            # Print the avg training and validation loss of 1 epoch in a clean way.
            descr = f"------ Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}"
            logger.debug(descr)

            # pbar.set_description(descr)

            # Log train full
            if wandb.run is not None:
                wandb.log({f"Train {str(criterion)} on whole dataset of {self.name}": avg_loss, "epoch": epoch})
        logger.info("--- Full train complete!")

    def pred(self, data: np.ndarray, with_cpu: bool) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Prediction function for the model.
        :param data: unlabelled data
        :param with_cpu: whether to use cpu or gpu
        :return: the predictions
        """
        # Prediction function
        logger.info(f"--- Predicting results with model {self.name}")
        # Run the model on the data and return the predictions

        if with_cpu:
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

                if with_cpu:
                    batch_prediction = batch_prediction.numpy()
                else:
                    batch_prediction = batch_prediction.cpu().numpy()

                predictions.append(batch_prediction)

        # Concatenate the predictions from all batches
        predictions = np.concatenate(predictions, axis=0)

        # Apply upsampling to the predictions
        downsampling_factor = 17280 // self.data_shape[1]
        if downsampling_factor > 1:
            predictions = np.repeat(predictions, downsampling_factor, axis=2)

        all_predictions = []
        all_confidences = []
        # Convert to events
        for pred in tqdm(predictions, desc="Converting predictions to events", unit="window"):
            # Convert to relative window event timestamps
            # TODO Add automatic thresholding to the model
            events = pred_to_event_state(pred, thresh=self.config["threshold"])

            # Add step offset based on repeat factor.
            offset = ((downsampling_factor / 2.0) - 0.5 if downsampling_factor % 2 == 0 else downsampling_factor // 2) if downsampling_factor > 1 else 0
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
            self.model = SegUnet1D(in_channels=self.data_shape[0], window_size=self.data_shape[1], out_channels=2, model_type=self.model_type, config=self.config)
            self.reset_optimizer()
            logger.info("Loading hyperparameters and instantiate new model from: " + path)
            return

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.reset_optimizer()
        logger.info("Model fully loaded from: " + path)
        return

    def reset_optimizer(self) -> None:

        """
        Reset the optimizer to the initial state. Useful for retraining the model.
        """
        self.config['optimizer'] = type(self.config['optimizer'])(self.model.parameters(), lr=self.config['optimizer'].param_groups[0]['lr'])
