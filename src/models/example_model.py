import torch
import torch.nn as nn

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
        # Load model
        self.model = SimpleModel(2, 10, 1, config)
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
                raise ModelException("Config is missing required parameter: " + req)

        # Get default_config
        default_config = self.get_default_config()

        config["loss"] = Loss.get_loss(config["loss"])
        config["batch_size"] = config.get("batch_size", default_config["batch_size"])
        config["lr"] = config.get("lr", default_config["lr"])
        config["optimizer"] = Optimizer.get_optimizer(config["optimizer"], config["lr"], self.model)
        self.config = config

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

        # Get hyperparameters from config (epochs, lr, optimizer)
        print("----------------")
        print(f"Training model: {type(self).__name__}")
        print(f"Hyperparameters: {self.config}")
        print("----------------")

        # Load hyperparameters
        criterion = self.config["loss"]
        optimizer = self.config["optimizer"]
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]

        # For now only look at enmo and anglez feature of data
        X = data[["enmo", "anglez"]].to_numpy()

        # Create a y with random regression values
        y = torch.rand(len(X), 1)

        # Create a tensor from X
        X = torch.from_numpy(X).float()

        # For now we split 50-50 into validation and test
        X_train = X[:len(X) // 2]
        y_train = y[:len(y) // 2]
        X_test = X[len(X) // 2:]
        y_test = y[len(y) // 2:]

        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        # Print the shapes and types of train and test
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print(f"X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")
        print(f"X_test type: {X_test.dtype}, y_test type: {y_test.dtype}")

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
            print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

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

    def load(self, path):
        """
        Load function for the model.
        :param path: path to model checkpoint
        :return:
        """
        self.model = SimpleModel(2, 10, 1, self.config)
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']


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
