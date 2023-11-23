from unittest import TestCase

import pandas as pd

from src.feature_engineering.add_school_hours import AddSchoolHours


class TestAddSchoolHours(TestCase):
    def test_feature_engineering(self):
        fe: AddSchoolHours = AddSchoolHours()

        data: dict[str, pd.DataFrame] = {
            "test": pd.DataFrame({
                "f_weekday": [0, 0, 0, 0, 0, 0, 3, 5, 6, 6],
                "f_hour": [7, 8, 15, 16, 16, 16, 15, 15, 9, 0],
                "f_minute": [59, 45, 46, 20, 21, 45, 20, 0, 59, 0],
            })
        }

        data = fe.feature_engineering(data)

        self.assertEqual(0, data["test"]["f_school_hour"].iloc[0])
        self.assertEqual(1, data["test"]["f_school_hour"].iloc[1])
        self.assertEqual(2, data["test"]["f_school_hour"].iloc[2])
        self.assertEqual(2, data["test"]["f_school_hour"].iloc[3])
        self.assertEqual(0, data["test"]["f_school_hour"].iloc[4])
        self.assertEqual(0, data["test"]["f_school_hour"].iloc[5])
        self.assertEqual(1, data["test"]["f_school_hour"].iloc[6])
        self.assertEqual(0, data["test"]["f_school_hour"].iloc[7])
        self.assertEqual(0, data["test"]["f_school_hour"].iloc[8])
        self.assertEqual(0, data["test"]["f_school_hour"].iloc[9])
