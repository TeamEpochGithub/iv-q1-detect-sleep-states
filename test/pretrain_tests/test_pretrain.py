from unittest import TestCase

import numpy as np
import pandas as pd

from src.pretrain.pretrain import Pretrain


class TestPretrain(TestCase):
    def test_from_config(self):
        pretrain: Pretrain = Pretrain.from_config({"test_size": 0.5, "scaler": {
            "kind": "standard-scaler",
            "copy": True
        }})
        self.assertEqual(pretrain.test_size, 0.5)
        self.assertEqual(pretrain.scaler.kind, "standard-scaler")
        self.assertTrue(pretrain.scaler.scaler.copy)

    def test_pretrain(self):
        pretrain: Pretrain = Pretrain.from_config({"test_size": 0.25, "scaler": {"kind": "standard-scaler"}})

        df: pd.DataFrame = pd.DataFrame({"series_id": np.concatenate(
            (np.repeat(0, 34560), np.repeat(1, 34560), np.repeat(2, 34560), np.repeat(3, 34560))),
            "enmo": np.random.rand(138240) * 2 + 1,
            "anglez": np.random.rand(138240) * 2 + 1,
            "awake": np.random.rand(138240) * 2 + 1,
            "f_test": np.random.rand(138240) * 2 + 1})

        X_train, X_test, y_train, y_test, train_idx, test_idx = pretrain.pretrain_split(df)

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
            flat = X_train[:, :, feature].flatten()
            self.assertAlmostEqual(0, flat.mean(), delta=0.1)
            self.assertAlmostEqual(1, flat.std(), delta=0.1)

    def test_preprocess(self):
        pretrain: Pretrain = Pretrain.from_config({"test_size": 0.5, "scaler": {"kind": "none"}})

        df: pd.DataFrame = pd.DataFrame({"series_id": np.repeat(0, 34560),
                                         "enmo": np.repeat(0, 34560),
                                         "anglez": np.repeat(0, 34560),
                                         "f_test": np.repeat(0, 34560)})

        pretrain.scaler.fit(df)

        x_data = pretrain.preprocess(df)
        self.assertEqual(x_data.shape, (2, 17280, 3))

    def test_train_test_split(self):
        df: pd.DataFrame = pd.DataFrame({"series_id": [0, 1],
                                         "enmo": [0, 1],
                                         "anglez": [1, 2]})
        train_data, test_data, train_idx, test_idx = Pretrain.train_test_split(df, test_size=0.5)
        self.assertEqual(train_data.shape, (1, 3))
        self.assertEqual(test_data.shape, (1, 3))
        self.assertEqual(train_idx.shape, (1,))
        self.assertEqual(test_idx.shape, (1,))

    def test_get_features(self):
        df: pd.DataFrame = pd.DataFrame({"series_id": [0, 1],
                                         "enmo": [0, 1],
                                         "anglez": [1, 2],
                                         "awake": [0, 1],
                                         "f_test": [0, 1]})
        self.assertListEqual(list(Pretrain.get_features(df).columns), ["enmo", "anglez", "f_test"])

    def test_split_on_labels(self):
        df: pd.DataFrame = pd.DataFrame({"series_id": [0, 1],
                                         "enmo": [0, 1],
                                         "anglez": [1, 2],
                                         "awake": [0, 1],
                                         "onset": [0, 1],
                                         "wakeup": [0, 1],
                                         "onset-NaN": [0, 1],
                                         "wakeup-NaN": [0, 1]})
        X, y = Pretrain.split_on_labels(df)
        self.assertListEqual(list(X.columns), ["enmo", "anglez"])
        self.assertListEqual(list(y.columns), ["awake", "onset", "onset-NaN", "wakeup", "wakeup-NaN"])

    def test_to_window_numpy(self):
        df: pd.DataFrame = pd.DataFrame({"series_id": np.repeat(0, 34560),
                                         "enmo": np.repeat(0, 34560),
                                         "anglez": np.repeat(0, 34560)})
        arr = df.to_numpy(dtype=np.float32)
        self.assertEqual(Pretrain.to_windows(arr).shape, (2, 17280, 3))
