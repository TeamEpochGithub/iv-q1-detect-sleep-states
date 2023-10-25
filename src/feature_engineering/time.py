import pandas as pd

from .feature_engineering import FE, FEException

_TIME_FEATURES: list[str] = ["year", "month", "day", "hour", "minute", "second", "microsecond"]


class Time(FE):
    """Add time features to the data

    The following time-related features can be added: "year", "month", "day", "hour", "minute", "second", "microsecond".

    # TODO Add "weekday" and "week"
    """

    def __init__(self, time_features: str | list[str], **kwargs: dict) -> None:
        """Initialize the Time class

        :param time_features: the time features to add
        """
        super().__init__(**kwargs | {"kind": "time"})

        if isinstance(time_features, list):
            self.time_features = time_features
        else:
            self.time_features = [time_features]

        if any(time_feature not in _TIME_FEATURES for time_feature in self.time_features):
            raise FEException(f"Unknown time features: {time_features}")

    def __repr__(self) -> str:
        """Return a string representation of a Time object"""
        return f"{self.__class__.__name__}(time_features={self.time_features})"

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add the selected time columns.

        :param data: the original data
        :return: the data with the selected time data added
        """
        for time_feature in self.time_features:
            data[f"f_{time_feature}"] = data["timestamp"].dt.__getattribute__(time_feature)

        return data
