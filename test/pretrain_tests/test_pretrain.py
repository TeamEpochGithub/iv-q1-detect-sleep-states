from unittest import TestCase

import numpy as np
import pandas as pd

from src import data_info
from src.pretrain.pretrain import Pretrain


class TestPretrain(TestCase):
    def test_from_config(self):
        pretrain: Pretrain = Pretrain.from_config({"test_size": 0.5, "scaler": {
            "kind": "standard-scaler",
            "copy": True
        }})
        data_info.window_size = 17280
        self.assertEqual(pretrain.test_size, 0.5)
        self.assertEqual(pretrain.scaler.kind, "standard-scaler")
        self.assertTrue(pretrain.scaler.scaler.copy)

    def test_pretrain(self):
        data_info.window_size = 17280
        pretrain: Pretrain = Pretrain.from_config({"test_size": 0.25, "scaler": {"kind": "standard-scaler"}})

        df: pd.DataFrame = pd.DataFrame({"series_id": np.concatenate(
            (np.repeat(0, 34560), np.repeat(1, 34560), np.repeat(2, 34560), np.repeat(3, 34560))),
            "enmo": np.random.rand(138240) * 2 + 1,
            "window": np.random.rand(138240) * 2 + 1,
            "anglez": np.random.rand(138240) * 2 + 1,
            "awake": np.random.rand(138240) * 2 + 1,
            "f_test": np.random.rand(138240) * 2 + 1})

        data = {
            0: df[df["series_id"] == 0].drop("series_id", axis=1),
            1: df[df["series_id"] == 1].drop("series_id", axis=1),
            2: df[df["series_id"] == 2].drop("series_id", axis=1),
            3: df[df["series_id"] == 3].drop("series_id", axis=1),
        }

        X_train, X_test, y_train, y_test, train_idx, test_idx, groups = pretrain.pretrain_split(data)

        self.assertEqual(X_train.shape, (6, 17280, 3))
        self.assertEqual(X_test.shape, (2, 17280, 3))
        self.assertEqual(y_train.shape, (6, 17280, 1))
        self.assertEqual(y_test.shape, (2, 17280, 1))

        # Assert that train data is perfectly normal
        for feature in range(3):
            flat = X_train[:, :, feature].flatten()
            self.assertAlmostEqual(0, flat.mean(), delta=1e-5)
            self.assertAlmostEqual(1, flat.std(), delta=1e-5)

        # Assert that test data is sort of normal
        for feature in range(3):
            flat = X_test[:, :, feature].flatten()
            self.assertAlmostEqual(0, flat.mean(), delta=0.1)
            self.assertAlmostEqual(1, flat.std(), delta=0.1)

    def test_preprocess(self):
        data_info.window_size = 17280
        pretrain: Pretrain = Pretrain.from_config({"test_size": 0.5, "scaler": {"kind": "none"}})

        df: pd.DataFrame = pd.DataFrame({"series_id": np.repeat(0, 34560),
                                         "enmo": np.repeat(0, 34560),
                                         "anglez": np.repeat(0, 34560),
                                         "f_test": np.repeat(0, 34560)})
        data = {0: df.drop("series_id", axis=1)}

        pretrain.scaler.fit(df)

        x_data = pretrain.preprocess(data)
        self.assertEqual(x_data.shape, (2, 17280, 3))

    def test_train_test_split(self):
        data_info.window_size = 17280
        data = {
            "42": pd.DataFrame({"enmo": [0, 1], "anglez": [1, 2]}),
            "420": pd.DataFrame({"enmo": [0, 1], "anglez": [1, 2]}),
        }

        train_series, test_series = Pretrain.train_test_split(data, test_size=0.5)
        self.assertEqual(np.array(["42"]), train_series)
        self.assertEqual(np.array(["420"]), test_series)

    def test_get_features(self):
        data_info.window_size = 17280
        df: pd.DataFrame = pd.DataFrame({"series_id": [0, 1],
                                         "enmo": [0, 1],
                                         "anglez": [1, 2],
                                         "awake": [0, 1],
                                         "f_test": [0, 1]})
        self.assertListEqual(list(Pretrain.get_features(df).columns), ["f_enmo", "f_anglez", "f_test"])

    def test_split_on_labels(self):
        data_info.window_size = 17280
        df: pd.DataFrame = pd.DataFrame({"series_id": [0, 1],
                                         "enmo": [0, 1],
                                         "anglez": [1, 2],
                                         "awake": [0, 1],
                                         "onset": [0, 1],
                                         "wakeup": [0, 1],
                                         "onset-NaN": [0, 1],
                                         "wakeup-NaN": [0, 1]})
        X, y = Pretrain.split_on_labels(df)
        self.assertListEqual(list(X.columns), ["f_enmo", "f_anglez"])
        self.assertListEqual(list(y.columns), ["awake", "onset", "wakeup", "onset-NaN", "wakeup-NaN", "series_id"])

    def test_to_window_numpy(self):
        data_info.window_size = 17280
        df: pd.DataFrame = pd.DataFrame({"series_id": np.repeat(0, 34560),
                                         "enmo": np.repeat(0, 34560),
                                         "anglez": np.repeat(0, 34560)})
        arr = df.to_numpy(dtype=np.float32)
        self.assertEqual(Pretrain.to_windows(arr).shape, (2, 17280, 3))
