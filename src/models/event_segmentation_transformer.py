import torch

from src.logger.logger import logger
from src.models.architectures.transformer.residual_transformer import ResidualTransformer
from src.models.event_model import EventModel
from .. import data_info


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
        if self.config.get("use_auxiliary_awake", False):
            self.config["network_params"]["num_class"] = 5
        else:
            self.config["network_params"]["num_class"] = 2
        self.config['network_params']["seq_len"] = data_info.window_size
        self.config['network_params']["input_size"] = len(
            data_info.X_columns)
        if 'residual_model' in self.config['network_params']:
            self.config['network_params']['residual_model']['input_size'] = self.config['network_params']['input_size']
            self.config['network_params']['residual_model']['out_size'] = self.config["network_params"]["num_class"]
        self.model = ResidualTransformer(**self.config['network_params'])

        # Load model class config
        self.load_config(self.config)

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
            "attention": {
                "type": "normal"
            },
            "lr_schedule": {
                "t_initial": 100,
                "warmup_t": 5,
                "warmup_lr_init": 0.000001,
                "lr_min": 2e-8
            },
            "network_params": {
                "heads": 6,
                "emb_dim": 256,
                "expansion": 4,
                "n_layers": 6,
                "residual_model": {
                    "hidden_size": 20,
                    "n_layers": 8,
                    "activation": "gelu",
                    "dropout": 0.4,
                    "bidir": True
                },
                "patch_size": 8,
                "pe": "fixed",
                "dropout": 0.6229411429626215,
                "attention": {
                    "type": "sparse",
                    "block_size": 60,
                    "local_attn_ctx": 20,
                    "attn_mode": "all"
                }
            }
        }

    def reset_weights(self) -> None:
        """
        Reset the weights of the model. Useful for retraining the model.
        """
        torch.manual_seed(42)
        self.model = ResidualTransformer(**self.config['network_params'])
