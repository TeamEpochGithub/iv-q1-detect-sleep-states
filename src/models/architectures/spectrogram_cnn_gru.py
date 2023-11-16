from src.models.architectures import multi_res_bi_GRU
from torch import nn
from segmentation_models_pytorch import Unet
import torchaudio.transforms as T

class SpectrogramCNNGRU(nn.Module):
    def __init__(self, in_channels, out_channels, model_type, config):
        super(SpectrogramCNNGRU, self).__init__()
        self.config = config
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
        self.GRU = multi_res_bi_GRU.MultiResidualBiGRU(input_size=(config.get('n_fft', 127)+1)//2, 
                                                       hidden_size=64, out_size=out_channels, n_layers=5, bidir=True, activation='relu', 
                                                       flatten=False, dropout=0,
                                                       internal_layers=1, model_name='')

    def forward(self, x):
        # Pass only enmo and anglez to the spectrogram
        x_spec = self.spectrogram(x)
        x_encoded = self.encoder(x_spec).squeeze(1)
        # The rest of the features are subsampled and passed to the decoder
        # as residual features
        y = self.GRU(x_encoded)
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
