import pandas as pd

from .feature_engineering import FE
from ..logger.logger import logger


class Rotation(FE):
    # TODO Add docstrings for the class and feature_engineering function
    # TODO Add tests

    def __init__(self, window_sizes: list[int] | None = None, **kwargs: dict) -> None:
        """Initialize the Rotation class

        :param window_sizes: the window sizes to use for the rolling window
        """
        super().__init__(**kwargs | {"kind": "rotation"})

        if window_sizes is None:
            self.window_sizes = [100]
        else:
            self.window_sizes = window_sizes

    def __repr__(self) -> str:
        """Return a string representation of a Rotation object"""
        return f"{self.__class__.__name__}(window_sizes={self.window_sizes})"

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
