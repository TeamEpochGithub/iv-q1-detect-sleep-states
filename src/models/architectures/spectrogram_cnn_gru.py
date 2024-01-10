import torch
import torchaudio.transforms as T
from src.external.segmentation_models_pytorch import Unet
from torch import nn

from src.models.architectures import multi_res_bi_GRU
from src.models.architectures.Unet_decoder import UNet1DDecoder


class MultiResidualBiGRUwSpectrogramCNN(nn.Module):
    def __init__(self, in_channels, out_channels, model_type, config, spec_features_indices):
        super(MultiResidualBiGRUwSpectrogramCNN, self).__init__()
        self.config = config
        self.gru_params = config.get('gru_params', {})
        self.spec_features_indices = spec_features_indices
        self.encoder = Unet(
            encoder_name=config.get('encoder_name', 'resnet34'),
            encoder_weights=config.get('encoder_weights', 'imagenet'),
            # The channels used by the encoder are for now only anglez and enmo
            # so this is hardcoded to 2 for now
            in_channels=len(spec_features_indices),
            classes=1,
            encoder_depth=config.get('encoder_depth', 5),
        )
        self.spectrogram = nn.Sequential(
            T.Spectrogram(n_fft=config.get('n_fft', 127), hop_length=config.get('hop_length', 1)),
            T.AmplitudeToDB(top_db=80),
            SpecNormalize()
        )
        self.GRU = multi_res_bi_GRU.MultiResidualBiGRU(input_size=in_channels,
                                                       hidden_size=self.gru_params.get("hidden_size", 64),
                                                       out_size=out_channels,
                                                       n_layers=self.gru_params.get("n_layers", 5),
                                                       bidir=True, activation=self.gru_params.get("activation", "relu"),
                                                       flatten=False, dropout=0,
                                                       internal_layers=1, model_name='')
        # will shape the encoder outputs to the same shape as the original inputs
        self.liner = nn.Linear(in_features=(config.get('n_fft', 127) + 1) // 2, out_features=in_channels)

        self.decoder = UNet1DDecoder(
            n_channels=(config.get('n_fft', 127) + 1) // 2,
            n_classes=out_channels,
            bilinear=config.get('bilinear', False),
            scale_factor=config.get('scale_factor', 2),
            duration=17280 // (12 * config.get('hop_length', 1)),
        )

    def forward(self, x, use_activation=True):
        # Pass only enmo and anglez to the spectrogram
        x = x.permute(0, 2, 1)
        x_spec = self.spectrogram(x[:, self.spec_features_indices, :])
        x_encoded = self.encoder(x_spec).squeeze(1)
        # The rest of the features are subsampled and passed to the decoder
        # as residual features
        if self.config.get('use_decoder', False):
            x_decoded = self.decoder(x_encoded)
        else:
            x_decoded = torch.zeros_like(x_encoded)
        x_encoded = x_encoded.permute(0, 2, 1)
        x_encoded_linear = self.liner(x_encoded)

        # downsample the input features to the same shape as the encoded features
        x = x[:, self.spec_features_indices, ::self.config.get('hop_length')]
        # now sum the residual features x and the encoded features x
        x_encoded_linear[:, ::self.config.get('hop_length'), self.spec_features_indices] += x.permute(0, 2, 1)

        y, _ = self.GRU(x_encoded_linear, use_activation=use_activation)
        return y + x_decoded


class SpecNormalize(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (batch, channel, freq, time)
        min_ = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_ = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        return (x - min_) / (max_ - min_ + self.eps)
