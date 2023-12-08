from dataclasses import dataclass

import gc
from tqdm import tqdm

from .rolling_window import RollingWindow
from ..logger.logger import logger


@dataclass
class Kurtosis(RollingWindow):
    # TODO Add docstrings for the class, feature_engineering and kurtosis functions
    # TODO Add tests

    def feature_engineering(self, data: dict) -> dict:
        # Loop through window sizes
        logger.debug("------ All features: " + str(self.features))
        for feature in self.features:
            for window_size in self.window_sizes:
                # Create rolling window features for kurtosis
                data = self.kurtosis(data, window_size, feature)
            logger.debug("--------- Feature done: " + str(feature))
        logger.debug("------ All features done")
        return data

    # Create rolling window features for kurtosis
    def kurtosis(self, data: dict, window_size: int, feature: str) -> dict:
        # Create a rolling window for kurtosis per series_id
        for sid in tqdm(data.keys()):
            data[sid]["f_kurtosis_" + feature + "_" + str(window_size)] = data[sid][feature].rolling(
                window_size).kurt().reset_index(0, drop=True)

            # Make sure there are no NaN values turn them into 0
            data[sid]["f_kurtosis_" + feature + "_" + str(window_size)] = data[sid][
                "f_kurtosis_" + feature + "_" + str(window_size)].fillna(0.0)

            # Clip kurtosis
            data[sid]["f_kurtosis_" + feature + "_" + str(window_size)] = data[sid][
                "f_kurtosis_" + feature + "_" + str(window_size)].clip(
                lower=data[sid]["f_kurtosis_" + feature + "_" + str(window_size)].mean() - 5 * data[sid][
                    "f_kurtosis_" + feature + "_" + str(window_size)].std(),
                upper=data[sid]["f_kurtosis_" + feature + "_" + str(window_size)].mean() + 5 * data[sid][
                    "f_kurtosis_" + feature + "_" + str(window_size)].std())
            gc.collect()
        return data
