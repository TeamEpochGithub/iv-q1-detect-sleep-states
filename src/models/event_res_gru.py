import copy

import numpy as np
import torch
import wandb
from timm.scheduler import CosineLRScheduler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.util.state_to_event import pred_to_event_state
from .architectures.multi_res_bi_GRU import MultiResidualBiGRU
from .trainers.event_trainer import EventTrainer
from .. import data_info
from ..logger.logger import logger
from ..loss.loss import Loss
from ..models.model import Model, ModelException
from ..optimizer.optimizer import Optimizer


class EventResGRU(Model):
    """
    Event segmentation residual-GRU model.
    """

    def __init__(self, config: dict, name: str) -> None:
        """
        Init function of the Event segmentation residual-GRU model.
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
        required = ["loss", "optimizer"]
        for req in required:
            if req not in config:
                logger.critical("------ Config is missing required parameter: " + req)
                raise ModelException("Config is missing required parameter: " + req)

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
        config["batch_size"] = config.get("batch_size", default_config["batch_size"])
        config["lr"] = config.get("lr", default_config["lr"])
        config["optimizer"] = Optimizer.get_optimizer(config["optimizer"], config["lr"], 0, self.model)
        if "lr_schedule" in config:
            config["lr_schedule"] = config.get("lr_schedule", default_config["lr_schedule"])
            config["scheduler"] = CosineLRScheduler(config["optimizer"], **self.config["lr_schedule"])
        config["epochs"] = config.get("epochs", default_config["epochs"])
        config["early_stopping"] = config.get("early_stopping", default_config["early_stopping"])
        config["activation_delay"] = config.get("activation_delay", default_config["activation_delay"])
        config["network_params"] = config.get("network_params", dict())
        config["threshold"] = config.get("threshold", default_config["threshold"])
        self.config = config

    def get_default_config(self) -> dict:
        return {
            "batch_size": 1,
            "lr": 0.001,
            "epochs": 100,
            "early_stopping": 3,
            "activation_delay": 0,
            "threshold": 0.0,
            "mask_unlabeled": False,
            "lr_schedule": {
                "t_initial": 100,
                "warmup_t": 5,
                "warmup_lr_init": 0.000001,
                "lr_min": 2e-8
            },
            "network_params": {
                "activation": "relu",
                "hidden_layers": 8
            }
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
        if "scheduler" in self.config:
            scheduler = self.config["scheduler"]
        else:
            scheduler = None
        early_stopping = self.config["early_stopping"]
        activation_delay = self.config["activation_delay"]
        if early_stopping > 0:
            logger.info(f"--- Early stopping enabled with patience of {early_stopping} epochs.")

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)

        # Get only the 2 event state features
        if mask_unlabeled:
            y_train = y_train[:, :, np.array(
                [data_info.y_columns["awake"], data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]])]
            y_test = y_test[:, :, np.array(
                [data_info.y_columns["awake"], data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]])]
        else:
            y_train = y_train[:, :, np.array(
                [data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]])]
            y_test = y_test[:, :, np.array(
                [data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]])]

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

        trainer = EventTrainer(
            epochs, criterion, early_stopping=early_stopping, mask_unlabeled=mask_unlabeled)
        avg_losses, avg_val_losses, total_epochs = trainer.fit(
            trainloader=train_dataloader, testloader=test_dataloader, model=self.model, optimizer=optimizer, name=self.name, scheduler=scheduler,
            activation_delay=activation_delay)

        if wandb.run is not None:
            self.log_train_test(avg_losses[:total_epochs], avg_val_losses[:total_epochs], total_epochs)

        logger.info("--- Training of model complete!")
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
        epochs = self.config["total_epochs"]
        batch_size = self.config["batch_size"]
        optimizer = self.config["optimizer"]
        self.reset_scheduler()
        if "scheduler" in self.config:
            scheduler = self.config["scheduler"]
        else:
            scheduler = None

        activation_delay = self.config["activation_delay"]

        # Create a dataset from X and y
        X_train = torch.from_numpy(X_train)
        cols = np.array([data_info.y_columns["state-onset"], data_info.y_columns["state-wakeup"]])
        y_train = torch.from_numpy(y_train[:, :, cols])

        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

        # Print the shapes and types of train and test
        logger.info(f"--- X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"--- X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")
        # Create a dataloader from the dataset
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

        # Train the model
        logger.info("--- Training model full " + self.name + " for " + str(epochs) + " epochs")
        trainer = EventTrainer(epochs, criterion)
        trainer.fit(trainloader=train_dataloader, testloader=None, model=self.model, optimizer=optimizer, name=self.name, scheduler=scheduler,
                    activation_delay=activation_delay)
        logger.info("Full train complete!")

    def pred(self, data: np.ndarray, pred_with_cpu: bool) -> tuple[np.ndarray, np.ndarray]:
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
        downsampling_factor = data_info.downsampling_factor
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
        Evaluation function for the model.F
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
            self.reset_scheduler()
            logger.info("Loading hyperparameters and instantiate new model from: " + path)
            return

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.reset_optimizer()
        self.reset_scheduler()
        logger.info("Model fully loaded from: " + path)

    def reset_weights(self) -> None:
        """
        Reset the weights of the model. Useful for retraining the model.
        """
        self.model = MultiResidualBiGRU(self.num_features, **self.config['network_params'])

    def reset_optimizer(self) -> None:

        """
        Reset the optimizer to the initial state. Useful for retraining the model.
        """
        self.config['optimizer'] = type(self.config['optimizer'])(self.model.parameters(), lr=self.config['lr'])

    def reset_scheduler(self) -> None:
        """
        Reset the scheduler to the initial state. Useful for retraining the model.
        """
        if 'scheduler' in self.config:
            self.config['scheduler'] = CosineLRScheduler(self.config['optimizer'], **self.config["lr_schedule"])
