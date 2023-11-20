from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import pandas as pd

from .feature_engineering import FE, FEException
from ..logger.logger import logger

_TIME_FEATURES: Final[set[str]] = {"year", "month", "week", "weekday", "day", "hour", "minute", "second", "microsecond"}


@dataclass
class Time(FE):
    """Add time features to the data

    The following time-related features can be added: "year", "month", "week", "weekday", "day", "hour", "minute", "second", "microsecond".

    :param time_features: the time features to add
    """
    time_features: Iterable[str]

    def __post_init__(self) -> None:
        """Check if the time features are supported"""
        if any(time_feature not in _TIME_FEATURES for time_feature in self.time_features):
            logger.critical(f"Unknown time features: {self.time_features}")
            raise FEException(f"Unknown time features: {self.time_features}")

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add the selected time columns.

        :param data: the original data
        :return: the data with the selected time data added
        """
        for time_feature in self.time_features:
            match time_feature:
                case "week":
                    data["f_week"] = data["timestamp"].dt.isocalendar().week.astype("uint8")
                case "weekday":
                    data["f_weekday"] = data["timestamp"].dt.weekday.astype("uint8")
                case _:
                    data[f"f_{time_feature}"] = data["timestamp"].dt.__getattribute__(time_feature)

        return data
