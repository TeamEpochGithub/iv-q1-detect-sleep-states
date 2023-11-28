from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Final, TypeAlias

import pandas as pd

JSON: TypeAlias = None | int | str | bool | list['JSON'] | dict[str, 'JSON']


class FE(ABC):
    """
    Base class for feature engineering (FE) steps.
    All child classes should implement the feature engineering method.
    """

    @staticmethod
    def from_config_single(config: dict[str, JSON]) -> FE:
        """Parse the config of a single feature engineering step and return the feature engineering step.

        :param config: the config of a single feature engineering step
        :return: the specific feature engineering step
        """
        config_copy = copy.deepcopy(config)
        kind: str = config_copy.pop("kind", None)

        assert kind is not None, "No kind specified for feature engineering step"

        from .kurtosis import Kurtosis
        from .mean import Mean
        from .skewness import Skewness
        from .time import Time
        from .rotation import Rotation
        from .sun import Sun
        from .sin_hour import SinHour
        from .add_holidays import AddHolidays
        from .parser import Parser
        from .add_school_hours import AddSchoolHours

        _FEATURE_ENGINEERING_KINDS: Final[dict[str, type[FE]]] = {
            'kurtosis': Kurtosis,
            'mean': Mean,
            'skewness': Skewness,
            'time': Time,
            'rotation': Rotation,
            'sun': Sun,
            'sin_hour': SinHour,
            'add_holidays': AddHolidays,
            'parser': Parser,
            'add_school_hours': AddSchoolHours
        }

        assert kind in _FEATURE_ENGINEERING_KINDS, f"Unknown feature engineering step: {kind=}"
        return _FEATURE_ENGINEERING_KINDS[kind](**config_copy)

    @staticmethod
    def from_config(config_list: list[dict[str, JSON]]) -> list[FE]:
        """Parse the config list of feature engineering steps and return the feature engineering steps.

        It also filters out the steps that are only for training.

        :param config_list: the list of feature engineering steps
        :return: the list of feature engineering steps
        """
        return [FE.from_config_single(fe_step) for fe_step in config_list]

    def run(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Run the feature engineering step.

        :param data: the data to process
        :return: the processed data
        """
        return self.feature_engineering(data)

    @abstractmethod
    def feature_engineering(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Process the data. This method should be overridden by the child class.

        :param data: the data to process
        :return: the processed data
        """
        pass


class FEException(Exception):
    """
    Exception class for feature engineering steps.
    """
    pass
