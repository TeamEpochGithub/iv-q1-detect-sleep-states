

class FE:
    def __init__(self, config):
        # Init function
        self.config = config

    def feature_engineering(self, data):
        # Feature engineering function
        raise NotImplementedError

    def run(self, data):

        return self.feature_engineering(data)
