import torch

from src.logger.logger import logger
from src.models.event_model import EventModel
from .architecture.transformer_pool import TransformerPool
from ... import data_info


class EventSegmentationTransformer(EventModel):
    """
    This is the model file for the event segmentation transformer model.
    """

    def __init__(self, config: dict, name: str) -> None:
        """
        Init function of the event segmentation transformer model
        :param config: configuration to set up the model
        :param data_shape: shape of the data (channels, sequence_size)
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

        # Init model
        self.model_type = "event-segmentation-transformer"

        # Load transformer config and model
        config["network_params"]["t_type"] = "event"
        if config.get("use_auxiliary_awake", False):
            config["network_params"]["num_class"] = 3
        else:
            config["network_params"]["num_class"] = 2
        config['network_params']["seq_len"] = data_info.window_size
        config['network_params']["tokenizer_args"]["channels"] = len(
            data_info.X_columns)
        self.model = TransformerPool(**config['network_params'])

        # Load model class config
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
            "use_auxiliary_awake": False,
            "lr_schedule": {
                "t_initial": 100,
                "warmup_t": 5,
                "warmup_lr_init": 0.000001,
                "lr_min": 2e-8
            },
            "network_params": {
                "heads": 8,
                "emb_dim": 48,
                "forward_dim": 96,
                "n_layers": 6,
                "pooling": "none",
                "tokenizer": "patch",
                "tokenizer_args": {
                    "patch_size": 12
                },
                "pe": "other",
                "dropout": 0.0
            }
        }

    def reset_weights(self) -> None:
        """
        Reset the weights of the model. Useful for retraining the model.
        """
        torch.manual_seed(42)
        self.model = TransformerPool(**self.config['network_params'])
