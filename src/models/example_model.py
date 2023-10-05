import pandas as pd
import torch

from .architectures.simple_model import SimpleModel
from ..logger.logger import logger
from ..loss.loss import Loss
from ..models.model import Model, ModelException
from ..optimizer.optimizer import Optimizer


# TODO This model is currently outdated...
class ExampleModel(Model):
    """
    This is a sample model file. You can use this as a template for your own models.
    The model file should contain a class that inherits from the Model class.
    """

    def __init__(self, config: dict) -> None:
        """
        Init function of the example model
        :param config: configuration to set up the model
        """
        super().__init__(config)

        # Check if gpu is available, else return an exception
        if not torch.cuda.is_available():
            logger.critical("GPU not available")
            raise ModelException("GPU not available")

        logger.info(f"--- Device set to model {type(self).__name__}: " + torch.cuda.get_device_name(0))
        self.device = torch.device("cuda")

        self.model_type = "state-prediction"
        # Load model
        self.model = SimpleModel(2, 10, 1, config)
        self.load_config(config)

    def load_config(self, config: dict) -> None:
        """
        Load config function for the model.
        :param config: configuration to set up the model
        """
        # Error checks. Check if all necessary parameters are in the config.
        required = ["loss", "epochs", "optimizer"]
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
        self.config = config

    def get_default_config(self) -> dict:
        """
        Get default config function for the model.
        :return: default config
        """
        return {"batch_size": 1, "lr": 0.001}

    def train(self, X_train: pd.DataFrame, X_test: pd.DataFrame, Y_train: pd.DataFrame, Y_test: pd.DataFrame) -> None:
        """
        Train function for the model.
        :param X_train: the training data
        :param X_test: the test data
        :param Y_train: the training labels
        :param Y_test: the test labels
        """

        # Get hyperparameters from config (epochs, lr, optimizer)
        # Load hyperparameters
        criterion = self.config["loss"]
        optimizer = self.config["optimizer"]
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]

        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

        # Print the shapes and types of train and test
        logger.info(f"--- X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        logger.info(f"--- X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
        logger.info(f"--- X_train type: {X_train.dtype}, Y_train type: {Y_train.dtype}")
        logger.info(f"--- X_test type: {X_test.dtype}, Y_test type: {Y_test.dtype}")

        # Create a dataloader from the dataset
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        # Add model and data to device cuda
        self.model.to("cuda")

        # Train the model
        for epoch in range(epochs):
            self.model.train(True)
            avg_loss = 0
            for i, (x, y) in enumerate(train_dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                # Clear gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(x)
                loss = criterion(outputs, y)

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
            logger.info(f"------ Epoch [{epoch + 1}/{epochs}],"
                        f" Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    def pred(self, X_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Prediction function for the model.
        :param X_pred: unlabeled data
        :return: the predictions
        """
        # Prediction function
        logger.info("--- Predicting model")
        # Run the model on the data and return the predictions

        # Push to device
        self.model.to(self.device)

        # Make a prediction
        with torch.no_grad():
            prediction = self.model(X_pred)
        return prediction

    def evaluate(self, pred: pd.DataFrame, target: pd.DataFrame) -> float:
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

    def load(self, path: str) -> None:
        """
        Load function for the model.
        :param path: path to model checkpoint
        """
        self.model = SimpleModel(2, 10, 1, self.config)
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']

        logger.info("Model loaded from: " + path)
