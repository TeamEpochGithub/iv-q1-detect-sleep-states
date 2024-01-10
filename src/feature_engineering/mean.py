from dataclasses import dataclass

import gc
from tqdm import tqdm

from .rolling_window import RollingWindow
from ..logger.logger import logger


@dataclass
class Mean(RollingWindow):

    def feature_engineering(self, data: dict) -> dict:
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
    def mean(self, data: dict, window_size: int, feature: str) -> dict:
        # Create a rolling window for mean per series_id
        for sid in tqdm(data.keys()):
            data[sid]["f_mean_" + feature + "_" + str(window_size)] = data[sid][feature].rolling(
                window_size).mean().reset_index(0, drop=True)

            # Make sure there are no NaN values turn them into 0
            data[sid]["f_mean_" + feature + "_" + str(window_size)] = data[sid]["f_mean_" + feature + "_" + str(window_size)].fillna(
                0.0)
            gc.collect()
        return data
