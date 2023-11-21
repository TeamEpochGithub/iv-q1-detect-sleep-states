from unittest import TestCase

import pandas as pd

from src.feature_engineering.time import Time


class TestTime(TestCase):
    def test_time_init(self):
        time = Time(["day"])
        self.assertEqual("Time(time_features=['day'])", str(time))

        time = Time(["day", "hour"])
        self.assertEqual("Time(time_features=['day', 'hour'])", str(time))

    def test_repr(self):
        self.assertEqual("Time(time_features=['day'])", repr(Time(["day"])))

    # def test_time_feature_engineering_single(self):
    #     data = pd.DataFrame({
    #         "timestamp": pd.to_datetime(["2021-01-04 12:15:18", "2022-02-05 13:16:19", "2023-03-06 14:17:20"])
    #     })

    #     time = Time(["day"])
    #     data = time.feature_engineering(data)

    #     self.assertEqual(2, len(data.columns))
    #     self.assertEqual("timestamp", data.columns[0])
    #     self.assertEqual("f_day", data.columns[1])

    #     self.assertEqual(4, data["f_day"][0])
    #     self.assertEqual(5, data["f_day"][1])
    #     self.assertEqual(6, data["f_day"][2])

    # def test_time_feature_engineering_multiple(self):
    #     data = pd.DataFrame({
    #         "timestamp": pd.to_datetime(["2021-01-04 12:15:18", "2022-02-05 13:16:19", "2023-03-06 14:17:20"])
    #     })

    #     time = Time(["year", "month", "day", "hour", "minute", "second"])
    #     data = time.feature_engineering(data)

    #     self.assertEqual(7, len(data.columns))
    #     self.assertEqual("f_year", data.columns[1])
    #     self.assertEqual("f_month", data.columns[2])
    #     self.assertEqual("f_day", data.columns[3])
    #     self.assertEqual("f_hour", data.columns[4])
    #     self.assertEqual("f_minute", data.columns[5])
    #     self.assertEqual("f_second", data.columns[6])

    #     self.assertEqual(2021, data["f_year"][0])
    #     self.assertEqual(2022, data["f_year"][1])
    #     self.assertEqual(2023, data["f_year"][2])

    #     self.assertEqual(1, data["f_month"][0])
    #     self.assertEqual(2, data["f_month"][1])
    #     self.assertEqual(3, data["f_month"][2])

    #     self.assertEqual(4, data["f_day"][0])
    #     self.assertEqual(5, data["f_day"][1])
    #     self.assertEqual(6, data["f_day"][2])

    #     self.assertEqual(12, data["f_hour"][0])
    #     self.assertEqual(13, data["f_hour"][1])
    #     self.assertEqual(14, data["f_hour"][2])

    #     self.assertEqual(15, data["f_minute"][0])
    #     self.assertEqual(16, data["f_minute"][1])
    #     self.assertEqual(17, data["f_minute"][2])

    #     self.assertEqual(18, data["f_second"][0])
    #     self.assertEqual(19, data["f_second"][1])
    #     self.assertEqual(20, data["f_second"][2])
