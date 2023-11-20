from dataclasses import dataclass

import pandas as pd

from src.feature_engineering.feature_engineering import FE

HOLIDAYS_DATES: list[pd.Timestamp] = [
    *pd.date_range("2017-6-28", "2017-9-5"),  # Summer holidays 2017
    *pd.date_range("2017-9-21", "2017-9-22"),  # Rosh Hashanah 2017
    pd.Timestamp("2017-10-9"),  # Columbus Day 2017
    # *pd.date_range("2017-9-29", "2017-9-30"),  # Yom Kippur 2017
    *pd.date_range("2017-11-23", "2017-11-24"),  # Thanksgiving 2017
    *pd.date_range("2017-12-25", "2018-1-1"),  # Christmas 2017
    pd.Timestamp("2018-1-15"),  # Martin Luther King Day 2018
    pd.Timestamp("2018-2-12"),  # Lincoln's Birthday 2018
    *pd.date_range("2018-2-16", "2018-2-23"),  # Lunar New Year & Midwinter Recess 2018
    *pd.date_range("2018-3-30", "2018-4-6"),  # Spring Recess 2018
    pd.Timestamp("2018-5-28"),  # Memorial Day 2018
    pd.Timestamp("2018-6-7"),  # Anniversary Day 2018
    pd.Timestamp("2018-6-15"),  # Eid al-Fitr 2018
    *pd.date_range("2018-6-28", "2018-9-5"),  # Summer holidays 2018
]

MIDDLE_SCHOOL_HOLIDAYS_DATES: list[pd.Timestamp] = [
    pd.Timestamp("2018-6-11"),  # Clerical Day 2018
]

HIGH_SCHOOL_HOLIDAYS_DATES: list[pd.Timestamp] = [
    pd.Timestamp("2018-1-26"),  # Regents Scoring Day 2018
    pd.Timestamp("2018-1-29"),  # Chancellor’s Conference Day for High Schools 2018
]


@dataclass
class AddHolidays(FE):
    """Add which days are public holidays in New York.

    The public holidays are added to the "holiday" column.
    For simplicity, we only add the holidays from the start of the school year in 2017.
    """

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add the holiday column.

        The values are 0 for no holiday, 1 for holiday, 2 for middle school holiday, and 3 for high school holiday.

        :param data: the original data with the timestamp column
        :return: the data with the holiday column added
        """
        # Add holidays
        data["holiday"] = 0
        data.loc[data["timestamp"].isin(HOLIDAYS_DATES), "holiday"] = 1
        data.loc[data["timestamp"].isin(MIDDLE_SCHOOL_HOLIDAYS_DATES), "holiday"] = 2
        data.loc[data["timestamp"].isin(HIGH_SCHOOL_HOLIDAYS_DATES), "holiday"] = 3

        return data
