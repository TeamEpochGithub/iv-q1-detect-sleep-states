from datetime import datetime
from unittest import TestCase

import numpy as np
import pandas as pd

from src.preprocessing.pp import PPException
from src.preprocessing.remove_unlabeled import RemoveUnlabeled


class TestRemoveUnlabeled(TestCase):
    """
    Tests the RemoveUnlabeled class.
    """

    remove_unlabeled = RemoveUnlabeled()

    def test_no_state(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "series_id": np.repeat("test", 20),
            "step": range(20),
            "timestamp": pd.date_range(datetime.today(), periods=20, freq="5S"),
            "anglez": np.random.uniform(2, 5, 20),
            "enmo": np.random.uniform(0, 1, 20),
        })

        with self.assertRaises(PPException) as context:
            self.remove_unlabeled.preprocess(df)
        self.assertEqual("No awake column. Did you run AddStateLabels before?", str(context.exception))

    def test_no_window(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "series_id": np.repeat("test", 20),
            "step": range(20),
            "timestamp": pd.date_range(datetime.today(), periods=20, freq="5S"),
            "anglez": np.random.uniform(2, 5, 20),
            "enmo": np.random.uniform(0, 1, 20),
            "awake": np.concatenate(
                (np.repeat(1, 3), np.repeat(0, 2), np.repeat(2, 5), np.repeat(0, 6), np.repeat(2, 4))),
        })

        df_test: pd.DataFrame = self.remove_unlabeled.preprocess(df)
        self.assertEqual(11, df_test.shape[0])

    def test_window(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "series_id": np.repeat("test", 20),
            "step": range(20),
            "timestamp": pd.date_range(datetime.today(), periods=20, freq="5S"),
            "anglez": np.random.uniform(2, 5, 20),
            "enmo": np.random.uniform(0, 1, 20),
            "awake": np.concatenate(
                (np.repeat(1, 2), np.repeat(0, 5), np.repeat(2, 8), np.repeat(0, 2), np.repeat(2, 3))),
            "window": np.concatenate(
                (np.repeat(1, 3), np.repeat(2, 5), np.repeat(3, 5), np.repeat(4, 5), np.repeat(5, 2))),
        })

        df_test: pd.DataFrame = self.remove_unlabeled.preprocess(df)
        self.assertEqual(13, df_test.shape[0])

    def test_window_unchanged(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "series_id": np.repeat("test", 20),
            "step": range(20),
            "timestamp": pd.date_range(datetime.today(), periods=20, freq="5S"),
            "anglez": np.random.uniform(2, 5, 20),
            "enmo": np.random.uniform(0, 1, 20),
            "awake": np.concatenate(
                (np.repeat(1, 2), np.repeat(0, 5), np.repeat(1, 8), np.repeat(0, 2), np.repeat(1, 3))),
            "window": np.concatenate(
                (np.repeat(1, 3), np.repeat(2, 5), np.repeat(3, 5), np.repeat(4, 5), np.repeat(5, 2))),
        })

        df_test: pd.DataFrame = self.remove_unlabeled.preprocess(df)
        self.assertEqual(20, df_test.shape[0])
