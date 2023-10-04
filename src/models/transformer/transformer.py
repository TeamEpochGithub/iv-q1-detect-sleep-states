import torch

from src.logger.logger import logger
from src.models.transformer.trainer import Trainer

from ...loss.loss import Loss
from ..model import Model, ModelException
from ...optimizer.optimizer import Optimizer
from .transformer_encoder import TSTransformerEncoderClassiregressor
import torchinfo


class Transformer(Model):
    """
    This is the model file for the transformer model.
    """

    def __init__(self, config):
        """
        Init function of the example model
        :param config: configuration to set up the model
        """
        super().__init__(config)
        # Init model
        self.transformer_config = self.load_transformer_config(config)
        self.model = TSTransformerEncoderClassiregressor(
            **self.transformer_config)
        self.load_config(config)

        # Check if gpu is available, else return an exception
        if not torch.cuda.is_available():
            raise ModelException("GPU not available")

        print("GPU Found: " + torch.cuda.get_device_name(0))
        self.device = torch.device("cuda")

    def load_config(self, config):
        """
        Load config function for the model.
        :param config: configuration to set up the model
        :return:
        """
        # Error checks. Check if all necessary parameters are in the config.
        required = ["loss", "epochs", "optimizer"]
        for req in required:
            if req not in config:
                logger.critical(
                    "------ Config is missing required parameter: " + req)
                raise ModelException(
                    "Config is missing required parameter: " + req)

        # Get default_config
        default_config = self.get_default_config()

        config["loss"] = Loss.get_loss(config["loss"])
        config["batch_size"] = config.get(
            "batch_size", default_config["batch_size"])
        config["lr"] = config.get("lr", default_config["lr"])
        config["optimizer"] = Optimizer.get_optimizer(
            config["optimizer"], config["lr"], self.model)
        config["patch_size"] = config.get(
            "patch_size", default_config["patch_size"])

        self.config = config

    def get_default_config(self):
        """
        Get default config function for the model.
        :return: default config
        """
        return {"batch_size": 32, "lr": 0.001, 'patch_size': 36}

    def load_transformer_config(self, config):
        """
        Load config function for the model.
        :param config: configuration to set up the model
        :return:
        """
        # Check if all necessary parameters are in the config.
        default_config = self.get_default_transformer_config()
        new_config = default_config.copy()
        for key in default_config:
            if key in config:
                new_config[key] = config[key]

        return new_config

    def get_default_transformer_config(self):
        """
        Get default config function for the model.
        :return: default config
        """
        return {
            'feat_dim': 72,
            'max_len': 480,
            'd_model': 192,
            'n_heads': 6,
            'num_layers': 5,
            'dim_feedforward': 2048,
            'num_classes': 480,
            'dropout': 0.1,
            'pos_encoding': "learnable",
            'activation': "relu",
            'norm': "BatchNorm",
            'freeze': False,
        }

    def train(self, X_train, X_test, y_train, y_test):
        """
        Train function for the model.
        :param data: labelled data
        :return:
        """

        # Get hyperparameters from config (epochs, lr, optimizer)
        print("----------------")
        print(f"Training model: {type(self).__name__}")
        print(f"Hyperparameters: {self.config}")
        print("----------------")

        # Load hyperparameters
        # criterion = self.config["loss"]
        # optimizer = self.config["optimizer"]
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]

        # Print the shapes and types of train and test
        logger.info(
            f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(
            f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logger.info(
            f"X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")
        logger.info(
            f"X_test type: {X_test.dtype}, y_test type: {y_test.dtype}")

        # Remove labels
        y_train = y_train[:, :, 0]
        y_test = y_test[:, :, 0]

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

        # Do patching
        patch_size = self.config["patch_size"]

        # Patch the data for the features
        X_train = torch.reshape(
            X_train, (X_train.shape[0], X_train.shape[1] // patch_size, patch_size, X_train.shape[2]))
        X_train = torch.reshape(
            X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2] * X_train.shape[3]))
        X_test = torch.reshape(
            X_test, (X_test.shape[0], X_test.shape[1] // patch_size, patch_size, X_test.shape[2]))
        X_test = torch.reshape(
            X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2] * X_test.shape[3]))

        # Patch the data for the labels
        y_train = torch.reshape(
            y_train, (y_train.shape[0], y_train.shape[1] // patch_size, patch_size))
        y_train = torch.transpose(y_train, 1, 2)
        y_train = torch.max(y_train, 1).values

        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        # Create a dataloader from the dataset
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size)

        # Torch summary
        logger.info(torchinfo.summary(self.model))
        trainer = Trainer(epochs=epochs)
        trainer.fit(train_dataloader, self.model, self.config["optimizer"])
        accuracy = trainer.evaluate(test_dataloader, self.model)
        print(f"Accuracy: {accuracy}")

    def pred(self, data):
        """
        Prediction function for the model.
        :param data: unlabelled data
        :return:
        """
        # Prediction function
        print("Predicting model")
        # Run the model on the data and return the predictions

        # Push to device
        self.model.to(self.device)

        # Make a prediction
        with torch.no_grad():
            prediction = self.model(data)
        return prediction

    def evaluate(self, pred, target):
        """
        Evaluation function for the model.
        :param pred: predictions
        :param target: targets
        :return: avg loss of predictions
        """
        self.model.eval()

        # Evaluate function
        print("Evaluating model")
        # Calculate the loss of the predictions
        criterion = self.config["loss"]
        loss = criterion(pred, target)
        return loss

    def save(self, path):
        """
        Save function for the model.
        :param path: path to save the model to
        :return:
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        print("Model saved to: " + path)

    def load(self, path):
        """
        Load function for the model.
        :param path: path to model checkpoint
        :return:
        """
        self.model = TSTransformerEncoderClassiregressor(
            **self.transformer_config)
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
