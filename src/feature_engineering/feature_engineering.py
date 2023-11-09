from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


from ..logger.logger import logger


class FE(ABC):
    """
    Base class for feature engineering (FE) steps.
    All child classes should implement the feature engineering method.
    """

    def __init__(self, kind: str, **kwargs: dict) -> None:
        """Initialize the feature engineering step.

        :param kind: the kind of feature engineering step
        :param kwargs: the parameters for the feature engineering step
        """
        self.kind = kind

    def __hash__(self) -> int:
        """Return the hash of the feature engineering step

        Just hash the string representation of the feature engineering step.
        """
        return hash(repr(self))

    def __eq__(self, other: object) -> bool:
        """Check if the feature engineering steps are equal. Necessary for hashing."""
        if not isinstance(other, FE):
            return NotImplemented
        return repr(self) == repr(other)

    @staticmethod
    def from_config_single(config: dict) -> FE:
        """Parse the config of a single feature engineering step and return the feature engineering step.

        :param config: the config of a single feature engineering step
        :return: the specific feature engineering step
        """
        from .kurtosis import Kurtosis
        from .mean import Mean
        from .skewness import Skewness
        from .time import Time
        from .rotation import Rotation
        from .sun import Sun

        _FEATURE_ENGINEERING_STEPS: dict[str, type[FE]] = {
            "kurtosis": Kurtosis,
            "mean": Mean,
            "skewness": Skewness,
            "time": Time,
            "rotation": Rotation,
            "sun": Sun
        }

        kind: str = config["kind"]

        try:
            return _FEATURE_ENGINEERING_STEPS[kind](**config)
        except KeyError:
            logger.critical(f"Unknown feature engineering step: {kind}")
            raise FEException(f"Unknown feature engineering step: {kind}")

    @staticmethod
    def from_config(config_list: list[dict]) -> list[FE]:
        """Parse the config list of feature engineering steps and return the feature engineering steps.

        It also filters out the steps that are only for training.

        :param config_list: the list of feature engineering steps
        :return: the list of feature engineering steps
        """
        return [FE.from_config_single(fe_step) for fe_step in config_list]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the feature engineering step.

        :param data: the data to process
        :return: the processed data
        """
        return self.feature_engineering(data)

    @abstractmethod
    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the data. This method should be overridden by the child class.

        :param data: the data to process
        :return: the processed data
        """
        logger.critical("Feature engineering method not implemented. Did you forget to override it?")
        raise FEException("Feature engineering method not implemented. Did you forget to override it?")


class FEException(Exception):
    """
    Exception class for feature engineering steps.
    """
    pass
