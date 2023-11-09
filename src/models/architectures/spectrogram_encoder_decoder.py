from torch import nn
import torchaudio.transforms as T
from segmentation_models_pytorch import Unet
from src.models.architectures.Unet_decoder import UNet1DDecoder


class SpectrogramEncoderDecoder(nn.Module):
    """
    Pytorch implementation of a really simple baseline model.
    """

    def __init__(self, in_channels, out_channels, model_type, config):
        super(SpectrogramEncoderDecoder, self).__init__()
        self.spectrogram = T.Spectrogram(n_fft=config.get('n_fft', 127), hop_length=config.get('hop_length', 1))
        self.encoder = Unet(
            encoder_name=config.get('encoder_name', 'resnet34'),
            encoder_weights=config.get('encoder_weights', 'imagenet'),
            in_channels=in_channels,
            classes=1,
        )
        self.decoder = UNet1DDecoder(
            n_channels=1,
            n_classes=out_channels,
            bilinear=config.get('bilinear', True),
            scale_factor=config.get('scale_factor', 1),
            # TODO do not hardcode this
            duration=17280
        )

    def forward(self, x):
        x_spec = self.spectrogram(x)
        x_encoded = self.encoder(x_spec).squeeze(1)
        y = self.decoder(x_encoded)
        return y
