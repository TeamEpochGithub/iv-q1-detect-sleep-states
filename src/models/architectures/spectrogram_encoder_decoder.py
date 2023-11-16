from torch import nn
from segmentation_models_pytorch import Unet
from src.models.architectures.Unet_decoder import UNet1DDecoder
import torchaudio.transforms as T
from torch import cat

# TODO if there are any features that work well with a spectrogram
# Change the hard coded 2 from this code


class SpectrogramEncoderDecoder(nn.Module):
    """
    Pytorch implementation of a really simple baseline model.
    """

    def __init__(self, in_channels, out_channels, model_type, config):
        super(SpectrogramEncoderDecoder, self).__init__()
        self.config = config
        self.num_res_features = in_channels - 3
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
        self.dropout = nn.Dropout(config.get('dropout_prob', 0.05))
        self.decoder = UNet1DDecoder(
            n_channels=(config.get('n_fft', 127)+1)//2 + self.num_res_features,
            n_classes=out_channels,
            bilinear=config.get('bilinear', False),
            scale_factor=config.get('scale_factor', 2),
            duration=17280//config.get('hop_length', 1),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        # Pass only enmo and anglez to the spectrogram
        x_spec = self.spectrogram(x[:, 0:3, :])
        x_encoded = self.encoder(x_spec).squeeze(1)
        # The rest of the features are subsampled and passed to the decoder
        # as residual features
        y = self.decoder(cat((x_encoded, x[:, 3:, ::self.config.get('hop_length')]), dim=1))
        y = self.decoder(x_encoded)
        if self.config.get('use_activation', False):
            y = self.activation(y)
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
