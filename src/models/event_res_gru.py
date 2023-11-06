import copy

import numpy as np
import torch
import wandb
from timm.scheduler import CosineLRScheduler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.util.state_to_event import pred_to_event_state
from .architectures.multi_res_bi_GRU import MultiResidualBiGRU
from .. import data_info
from ..logger.logger import logger
from ..loss.loss import Loss
from ..models.model import Model, ModelException
from ..optimizer.optimizer import Optimizer


class EventResGRU(Model):
    """
    This is a sample model file. You can use this as a template for your own models.
    The model file should contain a class that inherits from the Model class.
    """

    def __init__(self, config: dict, name: str) -> None:
        """
        Init function of the example model
        :param config: configuration to set up the model
        :param input_size: the number of features in the data
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

        self.model_type = "segmentation"
        self.num_features = len(data_info.X_columns)

        # Create model
        self.model = MultiResidualBiGRU(self.num_features, **config['network_params'])
        if wandb.run is not None:
            from torchsummary import summary
            summary(self.model.cuda(), input_size=(data_info.window_size, self.num_features))

        self.load_config(config)

    def load_config(self, config: dict) -> None:
        """
        Load config function for the model.
        :param config: configuration to set up the model
        """
        config = copy.deepcopy(config)

        # Error checks. Check if all necessary parameters are in the config.
        required = ["loss", "optimizer", "lr_schedule"]
        for req in required:
            if req not in config:
                logger.critical("------ Config is missing required parameter: " + req)
                raise ModelException("Config is missing required parameter: " + req)

        # Get default_config
        default_config = self.get_default_config()
        config["loss"] = Loss.get_loss(config["loss"])
        config["batch_size"] = config.get("batch_size", default_config["batch_size"])
        config["lr"] = config.get("lr", default_config["lr"])
        config["optimizer"] = Optimizer.get_optimizer(config["optimizer"], config["lr"], 0, self.model)
        config["lr_schedule"] = config.get("lr_schedule")
        config["epochs"] = config.get("epochs", default_config["epochs"])
        config["early_stopping"] = config.get("early_stopping", default_config["early_stopping"])
        config["activation_delay"] = config.get("activation_delay", default_config["activation_delay"])
        config["network_params"] = config.get("network_params", dict())
        config["threshold"] = config.get("threshold", default_config["threshold"])
        self.config = config

    def get_default_config(self) -> dict:
        return {
            "batch_size": 1,
            "lr": 0.1,
            "epochs": 100,
            "early_stopping": 3,
            "activation_delay": 0,
            "threshold": 0.0
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
        # in the docs for this function it says that t_initial is the number of epochs
        # but in the critical point code it is multiplied by the number of samples
        scheduler = CosineLRScheduler(optimizer, **self.config["lr_schedule"])

        early_stopping = self.config["early_stopping"]
        if early_stopping > 0:
            logger.info(f"--- Early stopping enabled with patience of {early_stopping} epochs.")

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)

        # Flatten y_train and y_test so we only get the regression labels
        # TODO get the proper labels from the data
        y_train = y_train[:, :, -2:]
        y_test = y_test[:, :, -2:]
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

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

        total_epochs = 0
        avg_losses = []
        avg_val_losses = []
        counter = 0
        lowest_val_loss = np.inf
        best_model = self.model.state_dict()
        stopped = False
        # Train the model

        for epoch in range(epochs):
            self.model.train(True)
            total_loss = 0
            use_activation = epoch > self.config["activation_delay"]
            with tqdm(train_dataloader, unit="batch") as pbar:
                for i, (x, y) in enumerate(pbar):
                    h = None
                    pbar.set_description(f"------ Epoch [{epoch + 1}/{epochs}]")
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                    # Clear gradients
                    optimizer.zero_grad()
                    scheduler.step(epoch)
                    # Forward pass
                    outputs, _ = self.model(x, h, use_activation=use_activation)
                    loss = criterion(outputs, y)

                    # Backward and optimize
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e-1)
                    optimizer.step()

                    # Calculate the avg loss for 1 epoch
                    total_loss += loss.item()
                    avg_loss = total_loss / (i + 1)
                    pbar.set_postfix(loss=avg_loss)

            # Calculate the validation loss
            self.model.train(False)
            total_val_loss = 0
            with torch.no_grad():
                with tqdm(test_dataloader, unit="batch") as pbar:
                    for i, (vx, vy) in enumerate(pbar):
                        pbar.set_description(f"------ Epoch [{epoch + 1}/{epochs}]")
                        h = None
                        vx = vx.to(self.device)
                        vy = vy.to(self.device)
                        voutputs, _ = self.model(vx, h)
                        vloss = criterion(voutputs, vy)
                        total_val_loss += vloss.item()
                        avg_val_loss = total_val_loss / (i + 1)
                        pbar.set_postfix(loss=avg_val_loss)

            # Print the avg training and validation loss of 1 epoch in a clean way.
            descr = f"------ Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            logger.debug(descr)
            pbar.set_description(descr)

            # Add average losses to list
            avg_losses.append(avg_loss)
            avg_val_losses.append(avg_val_loss)
            total_epochs += 1

            # Log train test loss to wandb
            if wandb.run is not None:
                wandb.log({f"Train {str(criterion)} of {self.name}": avg_loss,
                           f"Validation {str(criterion)} of {self.name}": avg_val_loss,
                           "epoch": epoch})

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
                        # TODO use the save function to save this model
                        stopped = True
                        break

        if wandb.run is not None:
            self.log_train_test(avg_losses, avg_val_losses, total_epochs)

        logger.info("--- Training of model complete!")
        if stopped:
            total_epochs -= early_stopping
        self.config["total_epochs"] = total_epochs

    def train_full(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
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
        epochs = self.config["total_epochs"]
        batch_size = self.config["batch_size"]
        # in the docs for this function it says that t_initial is the number of epochs
        # but in the critical point code it is multiplied by the number of samples
        scheduler = CosineLRScheduler(optimizer, **self.config["lr_schedule"])
        X_train = torch.from_numpy(X_train)

        # Flatten y_train and y_test so we only get the regression labels
        # TODO get the proper labels from the data
        y_train = y_train[:, :, -2:]
        y_train = torch.from_numpy(y_train)

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
            wandb.define_metric(f"Train {str(criterion)} of {self.name}", step_metric="epoch")
            wandb.define_metric(f"Validation {str(criterion)} of {self.name}", step_metric="epoch")

        # Train the model

        for epoch in range(epochs):
            self.model.train(True)
            total_loss = 0
            use_activation = epoch > self.config["activation_delay"]
            with tqdm(train_dataloader, unit="batch") as pbar:
                for i, (x, y) in enumerate(pbar):
                    h = None
                    pbar.set_description(f"------ Epoch [{epoch + 1}/{epochs}]")
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                    # Clear gradients
                    optimizer.zero_grad()
                    scheduler.step(epoch)
                    # Forward pass
                    outputs, _ = self.model(x, h, use_activation=use_activation)
                    loss = criterion(outputs, y)

                    # Backward and optimize
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e-1)
                    optimizer.step()

                    # Calculate the avg loss for 1 epoch
                    total_loss += loss.item()
                    avg_loss = total_loss / (i + 1)
                    pbar.set_postfix(loss=avg_loss)

            # Print the avg training and validation loss of 1 epoch in a clean way.
            descr = f"------ Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}"
            logger.debug(descr)
            pbar.set_description(descr)

            if wandb.run is not None:
                wandb.log({f"Train {str(criterion)} on whole dataset of {self.name}": avg_loss, "epoch": epoch})
        logger.info("--- Training of model complete!")

    def pred(self, data: np.ndarray, pred_with_cpu: bool):
        """
        Prediction function for the model.
        :param data: unlabelled data
        :param pred_with_cpu: whether to use cpu or gpu
        :return: the predictions
        """
        # Prediction function
        logger.info(f"--- Predicting results with model {self.name}")
        # Run the model on the data and return the predictions

        if pred_with_cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

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
                batch_prediction, _ = self.model(batch_data)

                if pred_with_cpu:
                    batch_prediction = batch_prediction.numpy()
                else:
                    batch_prediction = batch_prediction.cpu().numpy()

                predictions.append(batch_prediction)

        # Concatenate the predictions from all batches
        predictions = np.concatenate(predictions, axis=0)

        # Apply upsampling to the predictions
        downsampling_factor = 17280 // data.shape[1]
        if downsampling_factor > 1:
            predictions = np.repeat(predictions, downsampling_factor, axis=1)

        all_predictions = []
        all_confidences = []
        # Convert to events
        for pred in tqdm(predictions, desc="Converting predictions to events", unit="window"):
            # Convert to relative window event timestamps
            events = pred_to_event_state(pred.T, thresh=self.config["threshold"])

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
            self.model = MultiResidualBiGRU(self.num_features, **self.config['network_params'])
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

    def reset_weights(self) -> None:
        """
        Reset the weights of the model. Useful for retraining the model.
        """
        self.model = MultiResidualBiGRU(self.num_features, **self.config['network_params'])
