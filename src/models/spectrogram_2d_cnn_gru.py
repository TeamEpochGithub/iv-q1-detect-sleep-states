import torch
import wandb

from .architectures.spectrogram_cnn_gru import MultiResidualBiGRUwSpectrogramCNN
from .event_model import EventModel
from .. import data_info
from ..logger.logger import logger


class EventSegmentation2DCNNGRU(EventModel):
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
        # Features we want in the spectrgoram
        if self.config.get("use_spec_features", False):
            spec_features = ['f_enmo', 'f_anglez_diff_abs']
            # add the downsampling methods to these features
            spec_features_downsampled = []
            downsampling_methods = ["mean", "median", "max", "min", "std", "var", "range"]
            for feature in spec_features:
                for method in downsampling_methods:
                    # exclude max range and var from anglezdiffabs
                    if feature == "f_anglez_diff_abs" and method in ["max", "range", "var"]:
                        continue
                    spec_features_downsampled.append(feature + "_" + method)
            # Read the indices of the features we want to pass along the spectrogram from datainfo
            spec_features_indices = [data_info.X_columns[feature] for feature in spec_features_downsampled]
        else:
            spec_features_indices = list(range(len(data_info.X_columns.values())))
        # We load the model architecture here. 2 Out channels, one for onset, one for offset event state prediction
        if self.config.get("use_auxiliary_awake", False):
            self.model = MultiResidualBiGRUwSpectrogramCNN(in_channels=len(data_info.X_columns),
                                                           out_channels=5, model_type=self.model_type,
                                                           config=self.config,
                                                           spec_features_indices=spec_features_indices)
        else:
            self.model = MultiResidualBiGRUwSpectrogramCNN(in_channels=len(data_info.X_columns),
                                                           out_channels=2, model_type=self.model_type,
                                                           config=self.config,
                                                           spec_features_indices=spec_features_indices)
        data_info.window_size = 17280 // (data_info.downsampling_factor * config.get('hop_length', 1))
        self.inference_batch_size = 1
        # Load config
        self.load_config(config)
        # Print model summary
        if wandb.run is not None:
            if data_info.plot_summary:
                from torchsummary import summary
                summary(self.model.cuda(), input_size=(
                    len(data_info.X_columns), data_info.window_size))

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

    def reset_weights(self) -> None:
        """
        Reset the weights of the model. Useful for retraining the model.
        """
        torch.manual_seed(42)
        if self.config.get("use_spec_features", False):
            spec_features = ['f_enmo', 'f_anglez_diff_abs']
            # add the downsampling methods to these features
            spec_features_downsampled = []
            downsampling_methods = ["mean", "median", "max", "min", "std", "var", "range"]
            for feature in spec_features:
                for method in downsampling_methods:
                    # exclude max range and var from anglezdiffabs
                    if feature == "f_anglez_diff_abs" and method in ["max", "range", "var"]:
                        continue
                    spec_features_downsampled.append(feature + "_" + method)
            # Read the indices of the features we want to pass along the spectrogram from datainfo
            spec_features_indices = [data_info.X_columns[feature] for feature in spec_features_downsampled]
        else:
            spec_features_indices = list(range(len(data_info.X_columns.values())))
        if self.config.get("use_auxiliary_awake", False):
            self.model = MultiResidualBiGRUwSpectrogramCNN(in_channels=len(data_info.X_columns),
                                                           out_channels=5, model_type=self.model_type,
                                                           config=self.config,
                                                           spec_features_indices=spec_features_indices)
        else:
            self.model = MultiResidualBiGRUwSpectrogramCNN(in_channels=len(data_info.X_columns),
                                                           out_channels=2, model_type=self.model_type,
                                                           config=self.config,
                                                           spec_features_indices=spec_features_indices)
