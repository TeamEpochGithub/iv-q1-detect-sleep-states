import torch

from src.models.event_model import EventModel

from .architectures.seg_unet_1d_cnn import SegUnet1D
from .. import data_info
from ..logger.logger import logger


class EventSegmentationUnet1DCNN(EventModel):
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

        # Set the model type
        self.model_type = "event-segmentation"

        self.config["network_params"]["in_channels"] = len(data_info.X_columns)
        self.config["network_params"]["window_size"] = data_info.window_size
        self.config["network_params"]["out_channels"] = 2

        # We load the model architecture here. 2 Out channels, one for onset, one for offset event state prediction
        self.model = SegUnet1D(**self.config["network_params"])

        # Load config
        self.load_config(config)

    def get_default_config(self) -> dict:
        """
        Get default config function for the model.
        :return: default config
        """
        return {
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 10,
            "early_stopping": -1,
            "threshold": 0.0,
            "weight_decay": 0.0,
            "mask_unlabeled": False,
            "use_auxiliary_awake": False,
            "lr_schedule": {
                "t_initial": 100,
                "warmup_t": 5,
                "warmup_lr_init": 0.000001,
                "lr_min": 2e-8
            },
            "activation_delay": 0,
            "network_params": {
                "activation": "relu",
                "hidden_layers": 8,
                "kernel_size": 7,
                "depth": 2,
            }
        }

    def reset_weights(self) -> None:
        """
        Reset the weights of the model.
        """
        self.model = SegUnet1D(**self.config['network_params'])