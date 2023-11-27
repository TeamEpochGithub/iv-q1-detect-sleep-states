import numpy as np
import torch
import wandb
from src.models.trainers.event_state_trainer import EventStateTrainer

from .architectures.spectrogram_encoder_decoder import SpectrogramEncoderDecoder
from .event_model import EventModel
from .trainers.event_trainer import EventTrainer

from .. import data_info
from ..logger.logger import logger


class EventSegmentation2DCNN(EventModel):
    """
    This model is an event segmentation model based on the Unet 1D CNN. It uses the architecture from the SegSimple1DCNN class.
    """

    def __init__(self, config: dict, name: str) -> None:
        """
        Init function of the example model
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
            logger.info(
                f"--- Device set to model {self.name}: " + torch.cuda.get_device_name(0))

        self.model_type = "Spectrogram_2D_Cnn"
        data_info.window_size /= self.config.get('hop_length', 1)
        # We load the model architecture here. 2 Out channels, one for onset, one for offset event state prediction
        if self.config.get("use_auxiliary_awake", False):
            self.model = SpectrogramEncoderDecoder(
                in_channels=len(data_info.X_columns), out_channels=5, model_type=self.model_type, config=self.config)
        else:
            self.model = SpectrogramEncoderDecoder(
                in_channels=len(data_info.X_columns), out_channels=2, model_type=self.model_type, config=self.config)

        # Load config
        self.load_config(config)
        # Print model summary
        if wandb.run is not None:
            if data_info.plot_summary:
                from torchsummary import summary
                summary(self.model.cuda(), input_size=(
                    len(data_info.X_columns), data_info.window_size))
        # the downsample rate for the spectrogram models is the hop length
        # change the data info global downsample rate to be the hop length
        data_info.downsampling_factor = self.config.get('hop_length', 1)

    def get_default_config(self) -> dict:
        """
        Get default config function for the model.
        :return: default config
        """
        return {
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 10,
            "hidden_layers": 32,
            "kernel_size": 7,
            "depth": 2,
            "early_stopping": -1,
            "threshold": 0.5,
            "weight_decay": 0.0,
            "mask_unlabeled": False,
            "use_auxiliary_awake": False,
            "activation_delay": 0,
            "lr_schedule": None
        }

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
        use_auxiliary_awake = self.config["use_auxiliary_awake"]
        if early_stopping > 0:
            logger.info(
                f"--- Early stopping enabled with patience of {early_stopping} epochs.")

        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)

        # Get only the 2 event state features
        labels_list = [data_info.y_columns["state-onset"],
                       data_info.y_columns["state-wakeup"]]
        if mask_unlabeled:
            # Add awake label to front of the list
            labels_list.insert(0, data_info.y_columns["awake"])
        if use_auxiliary_awake:
            # Add awake label to end of the list
            labels_list.append(data_info.y_columns["awake"])
        labels_list = np.array(labels_list)

        y_train = torch.from_numpy(y_train[:, :, labels_list])
        y_test = torch.from_numpy(y_test[:, :, labels_list])

        # downsampling is not done in pretrain for spectrogram so we need to downsample y here
        y_train_downsampled = []
        for i in range(y_train.shape[0]):
            downsampled_channels = []
            for j in range(y_train.shape[2]):
                downsampled_channels.append(np.median(y_train[i, :, j].reshape(-1, self.config.get('hop_length', 1)), axis=1))
            y_train_downsampled.append(np.array(downsampled_channels))
        y_train = torch.from_numpy(np.array(y_train_downsampled)).permute(0, 2, 1)
        del y_train_downsampled

        # same loop as above to downsample the test data
        y_test_downsampled = []
        for i in range(y_test.shape[0]):
            downsampled_channels = []
            for j in range(y_test.shape[2]):
                downsampled_channels.append(np.median(y_test[i, :, j].reshape(-1, self.config.get('hop_length', 1)), axis=1))
            y_test_downsampled.append(np.array(downsampled_channels))
        y_test = torch.from_numpy(np.array(y_test_downsampled)).permute(0, 2, 1)
        del y_test_downsampled
        # Turn last column into one hot encoding of awake so that it can be used as auxiliary awake
        if use_auxiliary_awake:
            # Change all 3's for last column to 2's
            y_train[:, :, -1] = torch.where(
                y_train[:, :, -1] == 3, torch.tensor(2), y_train[:, :, -1])
            y_test[:, :, -1] = torch.where(
                y_test[:, :, -1] == 3, torch.tensor(2), y_test[:, :, -1])

            awake = y_train[:, :, -1]
            awake = torch.nn.functional.one_hot(awake.to(torch.int64))
            y_train = torch.cat((y_train[:, :, :-1], awake.float()), dim=2)

            awake = y_test[:, :, -1]
            awake = torch.nn.functional.one_hot(awake.to(torch.int64))
            y_test = torch.cat((y_test[:, :, :-1], awake.float()), dim=2)

        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

        # Print the shapes and types of train and test
        logger.info(
            f"--- X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(
            f"--- X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logger.info(
            f"--- X_train type: {X_train.dtype}, y_train type: {y_train.dtype}")
        logger.info(
            f"--- X_test type: {X_test.dtype}, y_test type: {y_test.dtype}")

        # Create a dataloader from the dataset
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size)

        if use_auxiliary_awake:
            trainer = EventStateTrainer(
                epochs, criterion, early_stopping=early_stopping, mask_unlabeled=mask_unlabeled)
        else:
            trainer = EventTrainer(
                epochs, criterion, early_stopping=early_stopping, mask_unlabeled=mask_unlabeled)
        avg_losses, avg_val_losses, total_epochs = trainer.fit(
            trainloader=train_dataloader, testloader=test_dataloader, model=self.model, optimizer=optimizer, name=self.name, scheduler=scheduler,
            activation_delay=activation_delay)

        if wandb.run is not None:
            self.log_train_test(
                avg_losses[:total_epochs], avg_val_losses[:total_epochs], total_epochs)

        logger.info("--- Training of model complete!")
        self.config["total_epochs"] = total_epochs

    # TODO refactor to overwrite event trainer
    def train_full(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
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
        use_auxiliary_awake = self.config.get("use_auxiliary_awake", False)
        if early_stopping > 0:
            logger.info(
                f"--- Early stopping enabled with patience of {early_stopping} epochs.")

        x_train = torch.from_numpy(x_train)

        # Get only the 2 event state features
        labels_list = [data_info.y_columns["state-onset"],
                       data_info.y_columns["state-wakeup"]]
        if mask_unlabeled:
            # Add awake label to front of the list
            labels_list.insert(0, data_info.y_columns["awake"])
        if use_auxiliary_awake:
            # Add awake label to end of the list
            labels_list.append(data_info.y_columns["awake"])
        labels_list = np.array(labels_list)

        y_train = torch.from_numpy(y_train[:, :, labels_list])

        # downsample the y data
        y_train_downsampled = []
        for i in range(y_train.shape[0]):
            downsampled_channels = []
            for j in range(y_train.shape[2]):
                downsampled_channels.append(np.median(y_train[i, :, j].reshape(-1, self.config.get('hop_length', 1)), axis=1))
            y_train_downsampled.append(np.array(downsampled_channels))
        y_train = torch.from_numpy(np.array(y_train_downsampled))
        del y_train_downsampled

        # Create a dataset from X and y
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

        # Print the shapes and types of train and test
        logger.info(
            f"--- X_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        logger.info(
            f"--- X_train type: {x_train.dtype}, y_train type: {y_train.dtype}")

        # Create a dataloader from the dataset
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size)

        if use_auxiliary_awake:
            trainer = EventStateTrainer(
                epochs, criterion, early_stopping=early_stopping, mask_unlabeled=mask_unlabeled)
        else:
            trainer = EventTrainer(
                epochs, criterion, early_stopping=early_stopping, mask_unlabeled=mask_unlabeled)
        trainer.fit(
            trainloader=train_dataloader, testloader=None, model=self.model, optimizer=optimizer, name=self.name, scheduler=scheduler,
            activation_delay=activation_delay)
        logger.info("Full train complete!")

    def reset_weights(self) -> None:
        """
        Reset the weights of the model. Useful for retraining the model.
        """
        torch.manual_seed(42)
        if self.config.get("use_auxiliary_awake", False):
            self.model = SpectrogramEncoderDecoder(
                in_channels=len(data_info.X_columns), out_channels=5, model_type=self.model_type, config=self.config)
        else:
            self.model = SpectrogramEncoderDecoder(
                in_channels=len(data_info.X_columns), out_channels=2, model_type=self.model_type, config=self.config)
