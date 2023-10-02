"""Unit tests for the Truncate preprocessing step"""

from datetime import datetime
from unittest import TestCase

import numpy as np
import pandas as pd

from src.preprocessing.pp import PPException
from src.preprocessing.truncate import Truncate


class TestTruncate(TestCase):
    """
    Tests the Truncate class.
    """

    truncate = Truncate()

    def test_no_state(self) -> None:
        df: pd.DataFrame = pd.DataFrame({
            "series_id": np.repeat("test", 20),
            "step": range(20),
            "timestamp": pd.date_range(datetime.today(), periods=20, freq="5S"),
            "anglez": np.random.uniform(2, 5, 20),
            "enmo": np.random.uniform(0, 1, 20),
        })

        with self.assertRaises(PPException) as context:
            self.truncate.preprocess(df)
        self.assertEqual("No awake column. Did you run AddStateLabels before?", str(context.exception))

    def test_truncate(self) -> None:
        df = pd.DataFrame({
            "series_id": np.repeat("test", 20),
            "step": range(20),
            "timestamp": pd.date_range(datetime.today(), periods=20, freq="5S"),
            "anglez": np.random.uniform(2, 5, 20),
            "enmo": np.random.uniform(0, 1, 20),
            "awake": np.concatenate(
                (np.repeat(1, 3), np.repeat(0, 2), np.repeat(2, 5), np.repeat(0, 6), np.repeat(2, 4))),
        })

        df_test: pd.DataFrame = self.truncate.preprocess(df)
        self.assertEqual(16, df_test.shape[0])

    def test_truncate_no_change(self) -> None:
        df = pd.DataFrame({
            "series_id": np.repeat("test", 20),
            "step": range(20),
            "timestamp": pd.date_range(datetime.today(), periods=20, freq="5S"),
            "anglez": np.random.uniform(2, 5, 20),
            "enmo": np.random.uniform(0, 1, 20),
            "awake": np.concatenate(
                (np.repeat(1, 2), np.repeat(0, 5), np.repeat(1, 8), np.repeat(0, 2), np.repeat(1, 3))),
        })

        df_test: pd.DataFrame = self.truncate.preprocess(df)
        self.assertEqual(20, df_test.shape[0])
