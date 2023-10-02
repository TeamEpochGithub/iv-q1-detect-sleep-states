from src.feature_engineering.feature_engineering import FE


class RollingWindow(FE):
    def __init__(self, config):
        super().__init__(config)
        self.window_sizes = (self.config["window_sizes"])
        self.window_sizes.sort()
        self.features = self.config["features"]
        self.features.sort()

    def feature_engineering(self, data):
        # Group the data
        return data
