from dataclasses import dataclass
from typing import Final

import pandas as pd

from src.feature_engineering.feature_engineering import FE

_MONDAY: Final[int] = 0
_TUESDAY: Final[int] = 1
_WEDNESDAY: Final[int] = 2
_THURSDAY: Final[int] = 3
_FRIDAY: Final[int] = 4

_SCHOOL_START_TIME: Final[tuple[int, int]] = 8, 0
_SCHOOL_END_TIME: Final[tuple[int, int]] = 15, 45
_HIGH_SCHOOL_END_TIME_MON_TUE: Final[tuple[int, int]] = 16, 20


@dataclass
class AddSchoolHours(FE):
    """Add school hours to the data.

    School hours in New York are 8:00-15:45 for middle school.
    For high schools, it's 8:00-15:45 on Wednesday, Thursday, and Friday's, and 8:00-16:20 on Monday and Tuesday.

    [Source for the school times](https://www.uft.org/your-rights/know-your-rights/length-school-day)
    """

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add the school hour column.

        The values are 0 for no school, 1 for school, 2 for high school only.

        :param data: the original data with at least the weekday, hour, and minute column
        :return: the data with the school hour column added
        """
        # Check if the f_weekday, f_hour, and f_minute column exists
        assert {"f_weekday", "f_hour", "f_minute"}.issubset(set(data.columns)), \
            "Missing necessary Time columns (weekday, hour, minute) for AddSchoolHours"

        _num_columns_before: int = len(data.columns)  # For assertion

        # Create a column for the school hour
        data["f_school_hour"] = 0

        # Check if it's (middle) school hours
        school_hours = (
                data["f_weekday"].isin(range(_MONDAY, _FRIDAY))
                & ((data["f_hour"] > _SCHOOL_START_TIME[0])  # After 8
                   | ((data["f_hour"] == _SCHOOL_START_TIME[0])
                      & (data["f_minute"] >= _SCHOOL_START_TIME[1])))  # After 08:00
                & ((data["f_hour"] < _SCHOOL_END_TIME[0])  # Before 15
                   | ((data["f_hour"] == _SCHOOL_END_TIME[0])
                      & (data["f_minute"] <= _SCHOOL_END_TIME[1])))  # Before 15:45
        )
        data.loc[school_hours, "f_school_hour"] = 1

        # Check if it's high school hours on Monday and Tuesday
        high_school_hours_mon_tue = (
                data["f_weekday"].isin([_MONDAY, _TUESDAY])
                & ((data["f_hour"] > _SCHOOL_END_TIME[0])  # After 15
                   | ((data["f_hour"] == _SCHOOL_END_TIME[0])
                      & (data["f_minute"] > _SCHOOL_END_TIME[1])))  # After 15:45
                & ((data["f_hour"] < _HIGH_SCHOOL_END_TIME_MON_TUE[0])  # Before 16
                   | ((data["f_hour"] == _HIGH_SCHOOL_END_TIME_MON_TUE[0])
                      & (data["f_minute"] <= _HIGH_SCHOOL_END_TIME_MON_TUE[1])))  # Before 16:20
        )
        data.loc[high_school_hours_mon_tue, "f_school_hour"] = 2

        # Convert the school hour column to uint8
        data["f_school_hour"] = data["f_school_hour"].astype("uint8")

        assert len(data.columns) == _num_columns_before + 1, "Column f_school_hour was not added!"
        return data
