"""Unit tests for the Truncate preprocessing step"""

from datetime import datetime
from unittest import TestCase

import numpy as np
import pandas as pd
from src.preprocessing.truncate import Truncate


class TestTruncate(TestCase):
    truncate = Truncate()

    def test_truncate_success(self) -> None:
        df = pd.DataFrame({
            "series_id": np.repeat("test", 20),
            "step": range(20),
            "timestamp": pd.date_range(datetime.today(), periods=20, freq="5S"),
            "anglez": np.random.uniform(2, 5, 20),
            "enmo": np.random.uniform(0, 1, 20),
            "event": np.concatenate(
                (np.repeat(None, 4), ["onset"], np.repeat(None, 10), ["wakeup"], np.repeat(None, 4))),
        })

        result = self.truncate.preprocess(df)
        assert (df.shape[0] > result.shape[0])

    def test_truncate_no_change(self) -> None:
        df = pd.DataFrame({
            "series_id": np.repeat("test", 20),
            "step": range(20),
            "timestamp": pd.date_range(datetime.today(), periods=20, freq="5S"),
            "anglez": np.random.uniform(2, 5, 20),
            "enmo": np.random.uniform(0, 1, 20),
            "event": np.concatenate(
                (np.repeat(None, 4), ["onset"], np.repeat(None, 14), ["wakeup"])),
        })

        result = self.truncate.preprocess(df)
        assert (df.shape[0] == result.shape[0])
