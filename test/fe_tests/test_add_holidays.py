from unittest import TestCase

import pandas as pd

from src.feature_engineering.add_holidays import AddHolidays
from src.feature_engineering.feature_engineering import FE


class Test(TestCase):
    add_holidays: FE = AddHolidays()

    def test_add_holidays(self) -> None:
        data_before: pd.DataFrame = pd.DataFrame({
            "timestamp": pd.date_range("2017-6-28", "2017-10-9")
        })

        data_after = self.add_holidays.feature_engineering(data_before)

        self.assertEqual(data_after["holiday"].iloc[0], 1)
        self.assertEqual(data_after["holiday"].iloc[69], 1)
        self.assertEqual(data_after["holiday"].iloc[70], 0)
        self.assertEqual(data_after["holiday"].iloc[71], 0)
        self.assertEqual(data_after["holiday"].iloc[102], 0)
        self.assertEqual(data_after["holiday"].iloc[103], 1)

    def test_add_holidays_middle_school(self) -> None:
        data_before: pd.DataFrame = pd.DataFrame({
            "timestamp": pd.date_range("2018-6-11", "2018-6-12")
        })

        data_after = self.add_holidays.feature_engineering(data_before)

        self.assertEqual(data_after["holiday"].iloc[0], 2)
        self.assertEqual(data_after["holiday"].iloc[1], 0)

    def test_add_holidays_high_school(self) -> None:
        data_before: pd.DataFrame = pd.DataFrame({
            "timestamp": pd.date_range("2018-1-26", "2018-1-27")
        })

        data_after = self.add_holidays.feature_engineering(data_before)

        self.assertEqual(data_after["holiday"].iloc[0], 3)
        self.assertEqual(data_after["holiday"].iloc[1], 0)