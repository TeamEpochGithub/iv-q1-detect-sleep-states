# Class for kurtosis feature
from ..feature_engineering.rolling_window import RollingWindow


class Kurtosis(RollingWindow):

    def __init__(self, config):
        super().__init__(config)

    def fe(self, data):
        # Loop through window sizes
        for feature in self.features:
            for window_size in self.window_sizes:
                # Create rolling window features for kurtosis
                data = self.kurtosis(data, window_size, feature)
        print(data.head(10))
        return data

    # Create rolling window features for kurtosis
    def kurtosis(self, data, window_size, feature):
        # Create a rolling window for kurtosis per series_id
        data["kurtosis_" + feature + "_" + str(window_size)] = data.groupby("series_id")[feature].rolling(
            window_size).kurt().reset_index(0, drop=True)

        # Make sure there are no NaN values turn them into 0
        data["kurtosis_" + feature + "_" + str(window_size)] = data["kurtosis_" + feature + "_" + str(window_size)].fillna(0.0)
        return data
