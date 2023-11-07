import pandas as pd

from .rolling_window import RollingWindow
from ..logger.logger import logger


class Kurtosis(RollingWindow):
    # TODO Add docstrings for the class, feature_engineering and kurtosis functions
    # TODO Add tests

    def __init__(self, **kwargs: dict) -> None:
        """Initialize the Kurtosis class"""
        super().__init__(**kwargs | {"kind": "kurtosis"})

    def __repr__(self) -> str:
        """Return a string representation of a Kurtosis object"""
        return f"{self.__class__.__name__}(window_sizes={self.window_sizes}, features={self.features})"

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
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
    def kurtosis(self, data: pd.DataFrame, window_size: int, feature: str) -> pd.DataFrame:
        # Create a rolling window for kurtosis per series_id
        data["f_kurtosis_" + feature + "_" + str(window_size)] = data.groupby("series_id")[feature].rolling(
            window_size).kurt().reset_index(0, drop=True)

        # Make sure there are no NaN values turn them into 0
        data["f_kurtosis_" + feature + "_" + str(window_size)] = data[
            "f_kurtosis_" + feature + "_" + str(window_size)].fillna(0.0)

        # Clip kurtosis
        data["f_kurtosis_" + feature + "_" + str(window_size)] = data[
            "f_kurtosis_" + feature + "_" + str(window_size)].clip(
            lower=data["f_kurtosis_" + feature + "_" + str(window_size)].mean() - 5 * data[
                "f_kurtosis_" + feature + "_" + str(window_size)].std(),
            upper=data["f_kurtosis_" + feature + "_" + str(window_size)].mean() + 5 * data[
                "f_kurtosis_" + feature + "_" + str(window_size)].std())

        return data
