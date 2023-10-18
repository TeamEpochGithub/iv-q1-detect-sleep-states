import copy
from typing import Any

import numpy as np
import torch
import wandb
from numpy import ndarray, dtype
from tqdm import tqdm

from ..logger.logger import logger
from ..loss.loss import Loss
from ..models.model import Model, ModelException
from ..optimizer.optimizer import Optimizer
from ..util.state_to_event import find_events
from .architectures.simple_GRU import SimpleGRU


class SimpleGRUModel(Model):
    """
    This is a sample model file. You can use this as a template for your own models.
    The model file should contain a class that inherits from the Model class.
    """

    def __init__(self, config: dict, input_size: tuple, name: str) -> None:
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

        self.model_type = "regression"
        self.input_size = input_size
        self.model = SimpleGRU(config, self.input_size[0])
        # Load model
        self.load_config(config)

        # If we log the run to weights and biases, we can
        if wandb.run is not None:
            from torchsummary import summary
            summary(self.model.cuda(), input_size=(input_size[1], input_size[0]))
            # pass

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
        config["lr"] = config.get("lr", default_config["lr"])
        config["optimizer"] = Optimizer.get_optimizer(config["optimizer"], config["lr"], self.model)
        config["epochs"] = config.get("epochs", default_config["epochs"])
        self.config = config

    def get_default_config(self) -> dict:
        """
        Get default config function for the model.
        :return: default config
        """
        return {"batch_size": 1, "lr": 0.1, "epochs": 20}

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

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)

        # Flatten y_train and y_test so we only get the regression labels
        y_train = y_train[:, 0, 1]
        y_test = y_test[:, 0, 1]
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
        wandb.define_metric("epoch")
        wandb.define_metric(f"Train {str(criterion)} of {self.name}", step_metric="epoch")
        wandb.define_metric(f"Validation {str(criterion)} of {self.name}", step_metric="epoch")

        avg_losses = []
        avg_val_losses = []
        # Train the model

        pbar = tqdm(range(epochs))
        for epoch in pbar:
            self.model.train(True)
            avg_loss = 0
            for i, (x, y) in enumerate(train_dataloader):
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # Clear gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(x)
                loss = criterion(outputs.squeeze(), y)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Calculate the avg loss for 1 epoch
                avg_loss += loss.item() / len(train_dataloader)

            # Calculate the validation loss
            self.model.train(False)
            avg_val_loss = 0
            with torch.no_grad():
                for i, (vx, vy) in enumerate(test_dataloader):
                    vx = vx.to(self.device)
                    vy = vy.to(self.device)
                    voutputs = self.model(vx)
                    vloss = criterion(voutputs, vy)
                    avg_val_loss += vloss.item() / len(test_dataloader)

            # Print the avg training and validation loss of 1 epoch in a clean way.
            descr = f"------ Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            logger.debug(descr)
            pbar.set_description(descr)

            # Add average losses to list
            avg_losses.append(avg_loss)
            avg_val_losses.append(avg_val_loss)

            # Log train test loss to wandb
            if wandb.run is not None:
                wandb.log({f"Train {str(criterion)} of {self.name}": avg_loss, f"Validation {str(criterion)} of {self.name}": avg_val_loss, "epoch": epoch})

        # Log full train and test plot
        self.log_train_test(avg_losses, avg_val_losses, epochs)
        logger.info("--- Training of model complete!")

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
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]

        X_train = torch.from_numpy(X_train)

        # Flatten y_train and y_test so we only get the regression labels
        y_train = y_train[:, 0, 1]
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
        wandb.define_metric("epoch")
        wandb.define_metric(f"Train {str(criterion)} of {self.name}", step_metric="epoch")
        wandb.define_metric(f"Validation {str(criterion)} of {self.name}", step_metric="epoch")

        avg_losses = []
        # Train the model

        for epoch in range(epochs):
            self.model.train(True)
            avg_loss = 0
            pbar = tqdm(enumerate(train_dataloader))
            for i, (x, y) in pbar:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # Clear gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(x)
                loss = criterion(outputs.squeeze(), y)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Calculate the avg loss for 1 epoch
                avg_loss += loss.item() / len(train_dataloader)

            # Print the avg training and validation loss of 1 epoch in a clean way.
            descr = f"------ Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}"
            logger.debug(descr)
            pbar.set_description(descr)

            # Add average losses to list
            avg_losses.append(avg_loss)

        logger.info("--- Training of model complete!")

    def pred(self, data: np.ndarray, with_cpu: bool) -> ndarray[Any, dtype[Any]]:
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

        self.model.to(device)
        # Convert data to tensor
        data = torch.from_numpy(data).to(device)

        # Print data shape
        logger.info(f"--- Data shape of predictions: {data.shape}")

        # Make a prediction
        prediction = None
        with torch.no_grad():
            # make predcitions in batches
            for sample in data:
                if prediction is None:
                    prediction = self.model(sample.unsqueeze(0))
                else:
                    prediction = torch.cat([prediction, self.model(sample.unsqueeze(0))])

        if with_cpu:
            prediction = prediction.numpy()
        else:
            prediction = prediction.cpu().numpy()

        logger.info(f"--- Done making predictions with model {self.name}")

        return np.array(prediction).T

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
            self.model = SimpleGRU(config=self.config, input_size=self.input_size[0])
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
