import torch
from src.models.event_model import EventModel
from .architectures.multi_res_bi_GRU import MultiResidualBiGRU
from .. import data_info
from ..logger.logger import logger


class EventResGRU(EventModel):
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
            logger.info(
                f"--- Device set to model {self.name}: " + torch.cuda.get_device_name(0))

        self.model_type = "segmentation"
        self.num_features = len(data_info.X_columns)

        # Create model
        self.model = MultiResidualBiGRU(
            self.num_features, **config['network_params'])

        self.load_config(config)

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
                "hidden_layers": 8,
                "flatten": False,
                "bidir": True,

            }
        }

    def reset_weights(self) -> None:
        """
        Reset the weights of the model. Useful for retraining the model.
        """
        torch.manual_seed(42)
        self.model = MultiResidualBiGRU(self.num_features, **self.config['network_params'])
