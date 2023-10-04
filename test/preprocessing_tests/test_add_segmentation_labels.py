from unittest import TestCase

import pandas as pd

from src.preprocessing.add_segmentation_labels import AddSegmentationLabels


class TestAddSegmentationLabels(TestCase):
    def test_preprocess_segmentation_labels_normal(self):
        """
        This test should test the onehot encoding of the awake column
        """
        df_dict = {"timestamp": pd.date_range(start="2021-01-01", periods=10, freq="5S"),
                   "enmo": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "anglez": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "step": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "series_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   "awake": [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
                   "window": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

        # Create a dataframe from the dict
        df = pd.DataFrame(df_dict)

        # Run the preprocessing step
        pp = AddSegmentationLabels()
        df = pp.preprocess(df)

        print(df.head())
        print(df.info())
        self.assertEqual(df["hot-asleep"].to_list(), [0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(df["hot-awake"].to_list(), [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(df["hot-NaN"].to_list(), [0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

        # Check if the columns are of type int8
        self.assertEqual("int8", df["hot-asleep"].dtype)
        self.assertEqual("int8", df["hot-awake"].dtype)
        self.assertEqual("int8", df["hot-NaN"].dtype)
        self.assertEqual("int8", df["awake"].dtype)

