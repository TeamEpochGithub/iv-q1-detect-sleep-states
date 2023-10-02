# This is the hyperparameter optimization class file
from ..logger.logger import logger


class HPO:
    def __init__(self, model, data, params):
        self.model = model
        self.data = data
        self.params = params

    def optimize(self):
        logger.info("Hyperparameter optimization not implemented yet")
        return 1
