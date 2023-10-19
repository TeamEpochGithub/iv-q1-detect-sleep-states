# Class for kurtosis feature
from .feature_engineering import FE
from ..logger.logger import logger


class Rotation(FE):

    def __init__(self, config):
        super().__init__(config)
        self.window_sizes = self.config.get('window_sizes', [100])

    def feature_engineering(self, data):
        for window_size in self.window_sizes:
            logger.debug(f"Calculating rotation smoothed with window size {window_size}")
            rotation = (data.groupby('series_id')['anglez']
                        .diff()
                        .abs()
                        .bfill()
                        .clip(upper=10)
                        .rolling(window=window_size, center=True)
                        .median()
                        .ffill()
                        .bfill()
                        .reset_index(0, drop=True))
            data[f'f_rotation_{window_size}'] = rotation.astype('float32')
        return data
