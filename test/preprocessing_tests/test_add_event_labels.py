from unittest import TestCase

import pandas as pd

from src.preprocessing.add_event_labels import AddEventLabels


class TestAddEventLabels(TestCase):
    def test_preprocess_event_labels_normal(self):
        """
        This test should test the most common case (going from awake to sleep to awake) of the AddEventLabels preprocessing step.
        """
        df_dict = {"timestamp": pd.date_range(start="2021-01-01", periods=10, freq="5S"),
                   "enmo": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "anglez": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "step": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "series_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   "awake": [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                   "window": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

        # Create a dataframe from the dict
        df = pd.DataFrame(df_dict)

        # Run the preprocessing step
        pp = AddEventLabels()
        df = pp.preprocess(df)

        # Check if the preprocessing step worked for onset (sleep onset occur should happen at step 3, so the onset column should be 3)
        self.assertEqual(df["onset"].to_list(), [3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

        # Check if the preprocessing step worked for wakeup (sleep awake occur should happen at step 8, so the onset column should be 8)
        self.assertEqual(df["wakeup"].to_list(), [8, 8, 8, 8, 8, 8, 8, 8, 8, 8])

        self.assertEqual(df["onset-NaN"].to_list(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(df["wakeup-NaN"].to_list(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        print(df["onset-NaN"].dtype)
        # Check if the NaN columns are of type int8
        self.assertEqual("int8", df["onset-NaN"].dtype)
        self.assertEqual("int8", df["wakeup-NaN"].dtype)

        # Check if the onset and wakeup are of type int16
        self.assertEqual("int16", df["onset"].dtype)
        self.assertEqual("int16", df["wakeup"].dtype)

    def test_preprocess_event_labels_only_sleep(self):
        """
        This test should test a case where there is no sleep awakening and only an onset.
        """
        df_dict = {"timestamp": pd.date_range(start="2021-01-01", periods=10, freq="5S"),
                   "enmo": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "anglez": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "step": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "series_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   "awake": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   "window": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

        # Create a dataframe from the dict
        df = pd.DataFrame(df_dict)

        # Run the preprocessing step
        pp = AddEventLabels()
        df = pp.preprocess(df)

        print(df.head())

        # Check if the preprocessing step worked for onset (sleep onset occur should happen at step 3, so the onset column should be 3)
        self.assertEqual(df["onset"].to_list(), [3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

        # Check if the preprocessing step worked for wakeup (sleep awake occur should happen at step 8, so the onset column should be 8)
        self.assertEqual(df["wakeup"].to_list(), [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

        self.assertEqual(df["onset-NaN"].to_list(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(df["wakeup-NaN"].to_list(), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        print(df["onset-NaN"].dtype)
        # Check if the NaN columns are of type int8
        self.assertEqual("int8", df["onset-NaN"].dtype)
        self.assertEqual("int8", df["wakeup-NaN"].dtype)

        # Check if the onset and wakeup are of type int16
        self.assertEqual("int16", df["onset"].dtype)
        self.assertEqual("int16", df["wakeup"].dtype)

    def test_preprocess_event_labels_no_valid_sleep(self):
        """
        This test should test a case where there is no valid onset / wakeup.
        """
        df_dict = {"timestamp": pd.date_range(start="2021-01-01", periods=10, freq="5S"),
                   "enmo": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "anglez": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "step": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "series_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   "awake": [1, 2, 2, 2, 2, 2, 2, 1, 1, 1],
                   "window": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

        # Create a dataframe from the dict
        df = pd.DataFrame(df_dict)

        # Run the preprocessing step
        pp = AddEventLabels()
        df = pp.preprocess(df)

        print(df.head())

        # Check if the preprocessing step worked for onset (sleep onset occur should happen at step 3, so the onset column should be 3)
        self.assertEqual(df["onset"].to_list(), [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

        # Check if the preprocessing step worked for wakeup (sleep awake occur should happen at step 8, so the onset column should be 8)
        self.assertEqual(df["wakeup"].to_list(), [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

        self.assertEqual(df["onset-NaN"].to_list(), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(df["wakeup-NaN"].to_list(), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        print(df["onset-NaN"].dtype)
        # Check if the NaN columns are of type int8
        self.assertEqual("int8", df["onset-NaN"].dtype)
        self.assertEqual("int8", df["wakeup-NaN"].dtype)

        # Check if the onset and wakeup are of type int16
        self.assertEqual("int16", df["onset"].dtype)
        self.assertEqual("int16", df["wakeup"].dtype)
