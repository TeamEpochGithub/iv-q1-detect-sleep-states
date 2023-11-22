from src.models.architectures import multi_res_bi_GRU
from torch import nn
from src.external.segmentation_models_pytorch import Unet
import torchaudio.transforms as T


class MultiResidualBiGRUwSpectrogramCNN(nn.Module):
    def __init__(self, in_channels, out_channels, model_type, config):
        super(MultiResidualBiGRUwSpectrogramCNN, self).__init__()
        self.config = config
        self.gru_params = config.get('gru_params', {})

        # TODO exclude some of the features from the spectrogram
        self.encoder = Unet(
            encoder_name=config.get('encoder_name', 'resnet34'),
            encoder_weights=config.get('encoder_weights', 'imagenet'),
            # The channels used by the encoder are for now only anglez and enmo
            # so this is hardcoded to 2 for now
            in_channels=in_channels,
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
                                                       out_size=out_channels, n_layers=self.gru_params.get("n_layers", 5),
                                                       bidir=True, activation=self.gru_params.get("activation", "relu"),
                                                       flatten=False, dropout=0,
                                                       internal_layers=1, model_name='')
        # will shape the encoder outputs to the same shape as the original inputs
        self.liner = nn.Linear(in_features=(config.get('n_fft', 127)+1)//2, out_features=in_channels)

    def forward(self, x, use_activation=True):
        # Pass only enmo and anglez to the spectrogram
        x = x.permute(0, 2, 1)
        x_spec = self.spectrogram(x)
        x_encoded = self.encoder(x_spec).squeeze(1)
        # The rest of the features are subsampled and passed to the decoder
        # as residual features
        x_encoded = x_encoded.permute(0, 2, 1)
        x_encoded = self.liner(x_encoded)

        # TODO if some features are excluded from the spectrgoram chnage this
        # downsample the input features to the same shape as the encoded features
        x = x[:, :, ::self.config.get('hop_length')]
        # now sum the residual features x and the encoded features x
        x_encoded += x.permute(0, 2, 1)

        y = self.GRU(x_encoded, use_activation=use_activation)
        return y


class SpecNormalize(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (batch, channel, freq, time)
        min_ = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_ = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        return (x - min_) / (max_ - min_ + self.eps)
