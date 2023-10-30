from unittest import TestCase

import pandas as pd

from src.preprocessing.add_regression_labels import AddRegressionLabels
from src.preprocessing.pp import PPException


class TestAddEventLabels(TestCase):
    pp = AddRegressionLabels('./dummy_event_path', './dummy_id_encoding_path')

    def test_add_event_labels_crash(self):
        df_dict = {"timestamp": pd.date_range(start="2021-01-01", periods=10, freq="5S"),
                   "enmo": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "anglez": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "step": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "series_id": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                   "awake": [1, 1, 0, 0, 0, 0, 0, 2, 2, 2]}

        # Create a dataframe from the dict
        df = pd.DataFrame(df_dict)

        with self.assertRaises(PPException) as context:
            self.pp.run(df)
        self.assertEqual(
            "No window column. Did you run SplitWindows before?", str(context.exception))
