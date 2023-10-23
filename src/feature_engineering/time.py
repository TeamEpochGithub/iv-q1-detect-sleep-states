import pandas as pd

from .feature_engineering import FE


class Time(FE):
    """Add time features to the data

    The following time-related features can be added: day, hour, minute, and second.

    # TODO Refactor to add parameter `time_features: str | list[str]` for weekday, week, month, year, etc.
    """

    def __init__(self, day: bool, hour: bool, minute: bool, second: bool, **kwargs: dict) -> None:
        """Initialize the Time class"""
        super().__init__(**kwargs)
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second

    def __str__(self) -> str:
        """Return the name of the class as a string"""
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        """Return a string representation of a Time object"""
        return f"{self.__class__.__name__}(day={self.day}, hour={self.hour}, minute={self.minute}, second={self.second})"

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add the selected time columns.

        :param data: the original data
        :return: the data with the selected time data added
        """
        if self.day:
            data["f_day"] = data["timestamp"].dt.day
        if self.hour:
            data["f_hour"] = data["timestamp"].dt.hour
        if self.minute:
            data["f_minute"] = data["timestamp"].dt.minute
        if self.second:
            data["f_second"] = data["timestamp"].dt.second

        return data
