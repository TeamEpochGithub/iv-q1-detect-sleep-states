import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.loss import Loss
from src.models.model import Model


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

    # Create different training function
    def train(self, data):
        """
        Train function for the model.
        :param data: labelled data
        :return:
        """
        # Define loss function from config based on current model name from loss class
        loss = Loss.get_loss(self.config["models"][self.__name__]["loss"])

        #Get hyperparameters from config (epochs, lr, optimizer)

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
