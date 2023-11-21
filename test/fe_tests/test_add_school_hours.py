# from unittest import TestCase
#
# import pandas as pd
#
# from src.feature_engineering.add_school_hours import AddSchoolHours
#
#
# class TestAddSchoolHours(TestCase):
#     def test_feature_engineering(self):
#         fe: AddSchoolHours = AddSchoolHours()
#
#         data = pd.DataFrame({
#             "timestamp": pd.date_range("2020-01-01", "2020-01-04", freq="H"),
#         })
#
#         data = fe.feature_engineering(data)
#
#         self.assertEqual(0, data["school_hour"].iloc[0])
#         self.assertEqual(1, data["school_hour"].iloc[8])
#         self.assertEqual(1, data["school_hour"].iloc[10])
#         self.assertEqual(0, data["school_hour"].iloc[18])
