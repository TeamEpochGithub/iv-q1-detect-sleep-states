from dataclasses import dataclass

import pandas as pd

from .rolling_window import RollingWindow
from ..logger.logger import logger


@dataclass
class Mean(RollingWindow):
    # TODO Add docstrings for the class, feature_engineering and mean functions
    # TODO Add tests

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.debug("------ All features: " + str(self.features))
        # Loop through window sizes
        for feature in self.features:
            for window_size in self.window_sizes:
                # Create rolling window features for mean
                data = self.mean(data, window_size, feature)
            logger.debug("--------- Feature done: " + str(feature))
        logger.debug("------ All features done")
        return data

    # Create rolling window features for mean
    def mean(self, data: pd.DataFrame, window_size: int, feature: str) -> pd.DataFrame:
        # Create a rolling window for mean per series_id
        data["f_mean_" + feature + "_" + str(window_size)] = data.groupby("series_id")[feature].rolling(
            window_size).mean().reset_index(0, drop=True)

        # Make sure there are no NaN values turn them into 0
        data["f_mean_" + feature + "_" + str(window_size)] = data["f_mean_" + feature + "_" + str(window_size)].fillna(
            0.0)
        return data
