# Class for kurtosis feature
from .feature_engineering import FE
from ..logger.logger import logger


class Rotation(FE):

    def __init__(self, config):
        super().__init__(config)
        self.window_sizes = self.config.get('window_sizes', [100])

    def feature_engineering(self, data):
        abs_diff = (data.groupby('series_id')['anglez']
                    .diff()
                    .abs()
                    .clip(upper=10))
        for window_size in self.window_sizes:
            logger.debug(f"Calculating rotation smoothed with window size {window_size}")
            change = (abs_diff
                      .rolling(window=window_size, center=True)
                      .median()
                      .ffill()
                      .bfill()
                      .reset_index(0, drop=True))
            data[f'f_rotation_{window_size}'] = change.astype('float32')
        return data
