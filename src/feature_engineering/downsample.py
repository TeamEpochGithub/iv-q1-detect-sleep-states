# Class for kurtosis feature
from .feature_engineering import FE
from ..logger.logger import logger


class Downsample(FE):

    def __init__(self, factor: int = 1, features: list[str] = None, methods: list[str] = None, standard: str = 'mean', **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.features = features
        self.methods = methods
        self.standard = standard

    def feature_engineering(self, data):
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
    def downsample(self, data):

        # Create a rolling window for kurtosis per series_id
        data["f_kurtosis_" + feature + "_" + str(window_size)] = data.groupby("series_id")[feature].rolling(
            window_size).kurt().reset_index(0, drop=True)

        # Make sure there are no NaN values turn them into 0
        data["f_kurtosis_" + feature + "_" + str(window_size)] = data["f_kurtosis_" + feature + "_" + str(window_size)].fillna(0.0)
        return data
