# Base class for preprocessing


class PP:
    def __init__(self, config):
        self.config = config

    def preprocess(self, data):
        raise NotImplementedError
