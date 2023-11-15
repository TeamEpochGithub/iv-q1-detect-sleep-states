from dataclasses import dataclass, field

import pandas as pd

from .feature_engineering import FE
from ..logger.logger import logger


@dataclass
class Rotation(FE):
    # TODO Add docstrings for the class and feature_engineering function
    # TODO Add tests

    window_sizes: list[int] = field(default_factory=lambda: [100])

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
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
