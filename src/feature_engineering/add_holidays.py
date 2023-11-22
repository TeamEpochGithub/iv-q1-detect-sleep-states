from dataclasses import dataclass
from typing import Final
from tqdm import tqdm

import pandas as pd

from src.feature_engineering.feature_engineering import FE

HOLIDAYS_DATES: Final[set[pd.Timestamp]] = {
    *pd.date_range("2017-6-28", "2017-9-5"),  # Summer holidays 2017
    *pd.date_range("2017-9-21", "2017-9-22"),  # Rosh Hashanah 2017
    pd.Timestamp("2017-10-9"),  # Columbus Day 2017
    # *pd.date_range("2017-9-29", "2017-9-30"),  # Yom Kippur 2017  # TODO Check if Yom Kippur was a school holiday in 2017
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
    *pd.date_range("2018-9-10", "2018-9-11"),  # Rosh Hashanah 2018
    pd.Timestamp("2018-9-19"),  # Yom Kippur 2018
    pd.Timestamp("2018-10-8"),  # Columbus Day 2018
    pd.Timestamp("2018-11-6"),  # Election day 2018
    pd.Timestamp("2018-11-12"),  # Veterans Day 2018
    *pd.date_range("2018-11-22", "2018-11-23"),  # Thanksgiving 2018
    *pd.date_range("2018-12-24", "2019-1-1"),  # Christmas 2018
    pd.Timestamp("2019-1-21"),  # Martin Luther King Day 2019
    pd.Timestamp("2019-2-12"),  # Lincoln's Birthday 2019
    pd.Timestamp("2019-2-5"),  # Lunar New Year 2019
    *pd.date_range("2019-2-18", "2019-2-22"),  # Midwinter Recess 2019
    *pd.date_range("2019-4-19", "2019-4-26"),  # Spring Recess 2019
    pd.Timestamp("2019-5-27"),  # Memorial Day 2019
    pd.Timestamp("2019-6-4"),  # Eid al-Fitr 2019
    pd.Timestamp("2019-6-6"),  # Anniversary Day 2019
    *pd.date_range("2019-6-27", "2019-9-4"),  # Summer holidays 2019
    *pd.date_range("2019-9-30", "2019-10-1"),  # Rosh Hashanah 2019
    pd.Timestamp("2019-10-9"),  # Yom Kippur 2019
    pd.Timestamp("2019-10-14"),  # Columbus Day 2019
    pd.Timestamp("2019-11-5"),  # Election day 2019
    pd.Timestamp("2019-11-11"),  # Veterans Day 2019
    *pd.date_range("2019-11-28", "2019-11-29"),  # Thanksgiving 2019
    *pd.date_range("2019-12-23", "2020-1-1"),  # Christmas 2019
    pd.Timestamp("2020-1-20"),  # Martin Luther King Day 2020
    pd.Timestamp("2020-2-12"),  # Lincoln's Birthday 2020
    pd.Timestamp("2020-2-24"),  # Lunar New Year 2020
    *pd.date_range("2020-2-17", "2020-2-21"),  # Midwinter Recess 2020
    *pd.date_range("2020-4-9", "2020-4-17"),  # Spring Recess 2020
    pd.Timestamp("2020-5-25"),  # Memorial Day 2020
    pd.Timestamp("2020-6-4"),  # Eid al-Fitr 2020
    pd.Timestamp("2020-6-5"),  # Anniversary Day 2020
    *pd.date_range("2020-6-29", "2020-9-4"),  # Summer holidays 2020
    *pd.date_range("2020-9-18", "2020-9-19"),  # Rosh Hashanah 2020
}

MIDDLE_SCHOOL_HOLIDAYS_DATES: Final[set[pd.Timestamp]] = {
    pd.Timestamp("2018-6-11"),  # Clerical Day 2018
    pd.Timestamp("2019-6-11"),  # Clerical Day 2019
}

HIGH_SCHOOL_HOLIDAYS_DATES: Final[set[pd.Timestamp]] = {
    pd.Timestamp("2018-1-26"),  # Regents Scoring Day 2018
    pd.Timestamp("2018-1-29"),  # Chancellor’s Conference Day for High Schools 2018
    pd.Timestamp("2019-1-28"),  # Chancellor’s Conference Day for High Schools 2019
    pd.Timestamp("2020-1-27"),  # January Clerical Day 2020
}


@dataclass
class AddHolidays(FE):
    """Add which days are public holidays in New York.

    The public holidays are added to the "holiday" column.
    For simplicity, we only add the holidays from the start of the school year in 2017 to
    the end of the school year in 2020.
    """

    def feature_engineering(self, data: dict) -> dict:
        """Add the holiday column.

        The values are 0 for no holiday, 1 for holiday, 2 for middle school holiday, and 3 for high school holiday.

        :param data: the original data with the timestamp column
        :return: the data with the holiday column added
        """
        # Add holidays
        assert "timestamp" in data[0].columns, "The timestamp column is missing"
        for sid in tqdm(data.keys()):
            data[sid]["f_holiday"] = 0
            data[sid].loc[data[sid]["timestamp"].dt.floor("D").isin(HOLIDAYS_DATES), "f_holiday"] = 1
            data[sid].loc[data[sid]["timestamp"].dt.floor("D").isin(MIDDLE_SCHOOL_HOLIDAYS_DATES), "f_holiday"] = 2
            data[sid].loc[data[sid]["timestamp"].dt.floor("D").isin(HIGH_SCHOOL_HOLIDAYS_DATES), "f_holiday"] = 3

            # Convert to uint8
            data[sid]["f_holiday"] = data[sid]["f_holiday"].astype("uint8")
        return data
