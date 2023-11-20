from dataclasses import dataclass

from tqdm import tqdm

from .rolling_window import RollingWindow
from ..logger.logger import logger


@dataclass
class Skewness(RollingWindow):
    # TODO Add docstrings for the class, feature_engineering and mean functions
    # TODO Add tests

    def feature_engineering(self, data: dict) -> dict:
        # Loop through window sizes
        logger.debug("------ All features: " + str(self.features))
        for feature in self.features:
            for window_size in self.window_sizes:
                # Create rolling window features for skewness
                data = self.skewness(data, window_size, feature)
            logger.debug("--------- Feature done: " + str(feature))
        logger.debug("------ All features done")
        return data

    # Create rolling window features for skewness
    def skewness(self, data: dict, window_size: int, feature: str) -> dict:
        # Create a rolling window for skewness per series_id
        for sid in tqdm(data.keys()):
            data[sid]["f_skewness_" + feature + "_" + str(window_size)] = data[sid][feature].rolling(
                window_size).skew().reset_index(0, drop=True)

            # Make sure there are no NaN values turn them into 0
            data[sid]["f_skewness_" + feature + "_" + str(window_size)] = data[sid][
                "f_skewness_" + feature + "_" + str(window_size)].fillna(0.0)
        return data
