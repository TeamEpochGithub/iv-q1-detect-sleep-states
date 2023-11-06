import copy
import numpy as np
import torch
from src import data_info

from src.logger.logger import logger

from ...loss.loss import Loss
from ..model import Model
from ...optimizer.optimizer import Optimizer
from typing import List
from torch import nn
from numpy import ndarray, dtype
from typing import Any
from .architecture.transformer_pool import TransformerPool
from ..model import ModelException


class BaseTransformer(Model):
    """
    This is the model file for a base implementation of the transformer, individual architectures and training can be implemented on top of this.
    """

    def __init__(self, config: dict, name: str) -> None:
        """
        Init function of the example model
        :param config: configuration to set up the model
        :param data_shape: shape of the data (channels, sequence_size)
        :param name: name of the model
        """
        super().__init__(config, name)

        # Init model
        self.name = name

        # Config for the transformer architecture
        self.transformer_config = self.load_transformer_config(config).copy()
        self.transformer_config["seq_len"] = data_info.window_size
        self.transformer_config["tokenizer_args"]["channels"] = len(data_info.X_columns)

        # Config for the model class
        self.config["seq_len"] = data_info.window_size

        # Check if gpu is available, else return an exception
        if not torch.cuda.is_available():
            logger.warning("GPU not available - using CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            logger.info(
                f"--- Device set to model {self.name}: " + torch.cuda.get_device_name(0))

    def load_config(self, loss: str, epochs: int, optimizer: str, **kwargs: dict) -> None:
        """
        Load config function for the model.
        :param loss: loss function
        :param epochs: number of epochs
        :param optimizer: optimizer
        :param kwargs: other parameters
        """

        # Get default_config
        default_config = self.get_default_config()

        # Copy kwargs
        config = copy.deepcopy(kwargs)

        # Add parameters
        config["batch_size"] = config.get(
            "batch_size", default_config["batch_size"])
        config["lr"] = config.get("lr", default_config["lr"])

        # Add loss, epochs and optimizer to config
        config["loss"] = Loss.get_loss(loss)
        config["optimizer"] = Optimizer.get_optimizer(
            optimizer, config["lr"], model=self.model)
        config["epochs"] = epochs
        config["trained_epochs"] = epochs

        self.config = config

    def get_default_config(self) -> dict[str, int | str]:
        """
        Get default config function for the model.
        :return: default config
        """
        return {"batch_size": 32, "lr": 0.000035, "mask_unlabeled": False, "early_stopping": 10}

    def load_transformer_config(self, config: dict[str, int | float | str]) -> dict[str, int | float | str]:
        """
        Load transformer config function for the model.
        :param config: configuration to set up the transformer architecture
        :return: transformer config
        """
        # Check if all necessary parameters are in the config.
        default_config = self.get_default_transformer_config()
        new_config = default_config.copy()
        for key in default_config:
            if key in config:
                new_config[key] = config[key]

        return new_config

    def get_default_transformer_config(self) -> dict[str, int | float | str]:
        """
        Get default config function for the model.
        :return: default config
        """
        return {
            'heads': 12,
            'emb_dim': 512,
            'forward_dim': 2048,
            'dropout': 0.1,
            'n_layers': 12,
            "tokenizer": "patch",
            'tokenizer_args': {},
            'seq_len': 17280,
            'num_class': 2,
            'pooling': 'none'
        }

    def train(self, X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array) -> None:
        """
        Train function for the model.
        :param X_train: the training data
        :param X_test: the test data
        :param y_train: the training labels
        :param y_test: the test labels
        """
        logger.critical(
            "--- Train of base class called. Did you forget to override it?")
        raise ModelException(
            "Train of base class called. Did you forget to override it?")

    def train_full(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model on the full dataset.
        :param X_train: the training data
        :param y_train: the training labels
        """
        logger.critical(
            "--- Train full of base class called. Did you forget to override it?")
        raise ModelException(
            "Train full of base class called. Did you forget to override it?")

    def pred(self, data: np.ndarray[Any, dtype[Any]], with_cpu: bool = False) -> ndarray[Any, dtype[Any]]:
        """
        Prediction function for the model.
        :param data: unlabelled data
        :param with_cpu: whether to use cpu
        :return: predictions of the model (windows, labels)
        """
        logger.critical(
            "--- Pred of base transformer class called. Did you forget to override it?")
        raise ModelException(
            "Pred of base transformer class called. Did you forget to override it?")

    def _pred_one_batch(self, data: torch.utils.data.DataLoader, preds: List[float], model: nn.Module) -> List[float]:
        """
        Predict one batch and return the predictions.
        :param data: data to predict on
        :param preds: predictions to append to
        :param model: model to predict with
        :return: predictions
        """

        # Make predictions without gradient
        with torch.no_grad():
            data[0] = data[0].float()
            output = model(data[0].to(self.device))
            preds = np.concatenate((preds, output.cpu().numpy()), axis=0)
        return preds

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
        logger.info("Model saved to: " + path)

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
        self.transformer_config['seq_len'] = self.config['seq_len']
        self.model = TransformerPool(tokenizer_args=self.transformer_config["tokenizer_args"],
                                     **((self.transformer_config, self.transformer_config.pop("tokenizer_args"))[0]))
        if not only_hyperparameters:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.reset_optimizer()

    def reset_optimizer(self) -> None:
        """
        Reset the optimizer to the initial state. Useful for retraining the model.
        """
        self.config['optimizer'] = type(self.config['optimizer'])(
            self.model.parameters(), lr=self.config['optimizer'].param_groups[0]['lr'])


# Custom adam optimizer
# Create custom adam optimizer
        # # save layer names
        # layer_names = []
        # for idx, (name, param) in enumerate(self.model.named_parameters()):
        #     layer_names.append(name)

        # # placeholder
        # parameters = []

        # # store params & learning rates
        # for idx, name in enumerate(layer_names):

        #     # Learning rate
        #     lr = self.config['lr']

        #     # parameter group name
        #     cur_group_name = name.split('.')[0]

        #     # update learning rate
        #     if cur_group_name == 'tokenizer':
        #         lr = self.config['lr_tokenizer']

        #     # display info
        #     logger.debug(f'{idx}: lr = {lr:.6f}, {name}')

        #     # append layer parameters
        #     parameters += [{'params': [p for n, p in self.model.named_parameters() if n == name and p.requires_grad],
        #                     'lr': lr}]

        # self.config['optimizer'] = type(self.config['optimizer'])(parameters)
