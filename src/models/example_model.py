import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.loss import Loss
from src.models.model import Model, ModelException
from src.optimizer.optimizer import Optimizer


class ExampleModel(Model):
    """
    This is a sample model file. You can use this as a template for your own models.
    The model file should contain a class that inherits from the Model class.
    """

    def __init__(self, config):
        """
        Init function of the example model
        :param config: configuration to set up the model
        """
        super().__init__(config)
        self.model = SimpleModel(10, 10, 2, config)

    def load_config(self, config):
        """
        Load config function for the model.
        :param config: configuration to set up the model
        :return:
        """

        print(config)
        # Error checks. Check if all necessary parameters are in the config.
        required = ["loss", "epochs", "optimizer"]
        for req in required:
            if req not in config:
                raise ModelException("Config is missing required parameter: " + req)

        #Get default_config
        default_config = self.get_default_config()

        config["loss"] = Loss.get_loss(config["loss"])
        config["batch_size"] = config.get("batch_size", default_config["batch_size"])
        config["lr"] = config.get("lr", default_config["lr"])
        config["optimizer"] = Optimizer.get_optimizer(config["optimizer"], self.model.model.parameters(), config["lr"])
        self.config = config

        print("Config loaded")
        print(self.config)

    def get_default_config(self):
        """
        Get default config function for the model.
        :return: default config
        """
        return {"batch_size": 1, "lr": 0.001}

    def train(self, data):
        """
        Train function for the model.
        :param data: labelled data
        :return:
        """
        # Define loss function from config based on current model name from loss class
        print(self.config)
        loss = Loss.get_loss(self.config["loss"])

        # Get hyperparameters from config (epochs, lr, optimizer)

        # Train function
        print("Training model")

    def pred(self, data):
        """
        Prediction function for the model.
        :param data: unlabelled data
        :return:
        """
        # Prediction function
        print("Predicting model")
        return [1, 2]

    def evaluate(self, pred, target):
        """
        Evaluation function for the model.
        :param pred: predictions
        :param target: targets
        :return:
        """
        # Evaluate function
        print("Evaluating model")
        return 0.5


class SimpleModel(nn.Module):
    """
    Pytorch implementation of a really simple baseline model.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, config):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
