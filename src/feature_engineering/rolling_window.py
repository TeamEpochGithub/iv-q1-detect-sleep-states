import pandas as pd

from .feature_engineering import FE, FEException
from ..logger.logger import logger


class RollingWindow(FE):
    """Base class for features that need a rolling window.

    # TODO Describe this class
    """

    def __init__(self, window_sizes: list[int], features: list[str], **kwargs: dict) -> None:
        """Initialize the Rolling Window class

        # TODO Describe window_sizes and features
        :param window_sizes: ???
        :param features: ???
        """
        super().__init__(**kwargs)

        self.window_sizes = window_sizes
        self.window_sizes.sort()
        self.features = features
        self.features.sort()

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the data. This method should be overridden by the child class.

        :param data: the data to process
        :return: the processed data
        """
        logger.critical("Rolling Window is a base class. Did you forget to override it?")
        raise FEException("Rolling Window is a base class. Did you forget to override it?")
