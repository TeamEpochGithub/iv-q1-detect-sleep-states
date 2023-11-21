from unittest import TestCase

import numpy as np
import pandas as pd

from src.preprocessing.add_event_labels import AddEventLabels
from src.preprocessing.pp import PPException


class TestAddEventSegmentationLabels(TestCase):
    pp = AddEventLabels('./dummy_event_path', './dummy_id_encoding_path', smoothing=2)

    def test_custom_score_array(self):
        arr = [0] * 361 + [1] + [0] * 361
        arr = np.array(arr)
        scoring = self.pp.custom_score_array(arr)
        self.assertEqual(scoring[361], 1)
        self.assertEqual(scoring[348], 0.9)
        self.assertEqual(scoring[0], 0)
        self.assertEqual(scoring[-1], 0)

    def test_repr(self):
        self.assertEqual("AddEventLabels(events_path='./dummy_event_path', id_encoding_path='./dummy_id_encoding_path', smoothing=2, steepness=1)", self.pp.__repr__())

    # def test_add_event_labels_crash(self):
    #     df_dict = {"timestamp": pd.date_range(start="2021-01-01", periods=10, freq="5S"),
    #                "enmo": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                "anglez": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                "step": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                "series_id": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    #                "awake": [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    #                "window": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    #     # Create a dataframe from the dict
    #     df = pd.DataFrame(df_dict)

    #     with self.assertRaises(PPException) as context:
    #         self.pp.run(df)
    #     self.assertEqual("Window column is present, this preprocessing step should be run before SplitWindows", str(context.exception))
