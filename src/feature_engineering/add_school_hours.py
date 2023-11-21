from dataclasses import dataclass
from typing import Final

import pandas as pd

from src.feature_engineering.feature_engineering import FE, FEException
from src.logger.logger import logger

_MONDAY: Final[int] = 0
_TUESDAY: Final[int] = 1
_WEDNESDAY: Final[int] = 2
_THURSDAY: Final[int] = 3
_FRIDAY: Final[int] = 4
_SATURDAY: Final[int] = 5
_SUNDAY: Final[int] = 6

_SCHOOL_START_TIME: Final[str] = "08:00"
_SCHOOL_END_TIME: Final[str] = "15:45"
_HIGH_SCHOOL_END_TIME_MON_TUE: Final[str] = "14:20"


@dataclass
class AddSchoolHours(FE):
    """Add school hours to the data.

    School hours in New York are 8:00-15:45 for middle school.
    For high schools, it's 8:00-15:45 on Wednesday, Thursday, and Friday's, and 8:00-14:20 on Monday and Tuesday.

    [Source for the school times](https://www.uft.org/your-rights/know-your-rights/length-school-day)
    """

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add the school hour column.

        The values are 0 for no school, 1 for school, 2 for middle school only.

        :param data: the original data
        :return: the data with the school hour column added
        """
        # Check if the f_weekday column exists
        if "f_weekday" not in data.columns:
            logger.critical("The f_weekday column is required for AddSchoolHours")
            raise FEException("The f_weekday column is required for AddSchoolHours")

        # Create a column for the school hour
        data["school_hour"] = 0

        # Set the school hour to 2 for all timestamps between 8:00-15:45 on work days
        middle_school_hours = (data["f_weekday"].isin(range(_MONDAY, _FRIDAY)) & (
                data["timestamp"].dt.strftime("%H:%M") >= _SCHOOL_START_TIME) & (
                                       data["timestamp"].dt.strftime("%H:%M") <= _SCHOOL_END_TIME))
        data.loc[middle_school_hours, "school_hour"] = 2

        # Set the school hour to 1 for all timestamps between 8:00-15:45 on Wednesday, Thursday, and Friday's and 8:00-14:20 on Monday and Tuesday
        high_school_hours_mon_tue = (data["f_weekday"].isin([_MONDAY, _TUESDAY])) & (
                    data["timestamp"].dt.strftime("%H:%M") >= _SCHOOL_START_TIME) & (
                                                data["timestamp"].dt.strftime("%H:%M") <= _HIGH_SCHOOL_END_TIME_MON_TUE)
        high_school_hours_wed_thu_fri = (
                data["f_weekday"].isin(range(_WEDNESDAY, _FRIDAY)) & (
                data["timestamp"].dt.strftime("%H:%M") >= _SCHOOL_START_TIME) & (
                        data["timestamp"].dt.strftime("%H:%M") <= _SCHOOL_END_TIME))
        data.loc[high_school_hours_mon_tue | high_school_hours_wed_thu_fri, "school_hour"] = 1

        # Convert the school hour column to uint8
        data["school_hour"] = data["school_hour"].astype("uint8")
        return data
