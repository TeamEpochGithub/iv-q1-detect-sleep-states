from abc import ABC, abstractmethod
from dataclasses import dataclass


from .feature_engineering import FE, FEException
from ..logger.logger import logger


@dataclass
class RollingWindow(FE, ABC):
    """Base class for features that need a rolling window.

    """

    window_sizes: list[int]
    features: list[str]

    def __post_init__(self) -> None:
        """Sort the window sizes and features"""
        self.window_sizes.sort()
        self.features.sort()

    @abstractmethod
    def feature_engineering(self, data: dict) -> dict:
        """Process the data. This method should be overridden by the child class.

        :param data: the data to process
        :return: the processed data
        """
        logger.critical("Rolling Window is a base class. Did you forget to override it?")
        raise FEException("Rolling Window is a base class. Did you forget to override it?")
