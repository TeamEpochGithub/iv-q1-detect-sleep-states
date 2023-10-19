from unittest import TestCase

import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer, \
    QuantileTransformer, Normalizer

from src.scaler.scaler import Scaler, ScalerException


class TestScaler(TestCase):
    def test_get_scaler(self):
        self.assertIsInstance(Scaler("standard-scaler").scaler, StandardScaler)
        self.assertIsInstance(Scaler("minmax-scaler").scaler, MinMaxScaler)
        self.assertIsInstance(Scaler("robust-scaler").scaler, RobustScaler)
        self.assertIsInstance(Scaler("maxabs-scaler").scaler, MaxAbsScaler)
        self.assertIsInstance(Scaler("power-transformer").scaler, PowerTransformer)
        self.assertIsInstance(Scaler("quantile-transformer").scaler, QuantileTransformer)
        self.assertIsInstance(Scaler("normalizer").scaler, Normalizer)
        self.assertIsNone(Scaler("none").scaler)
        self.assertRaises(ScalerException, Scaler, "unknown")

    def test_scale_standard(self):
        scaler: Scaler = Scaler("standard-scaler")

        df = pd.DataFrame({"enmo": [0, 1],
                           "anglez": [0, 2],
                           "trash": [123, 321]})

        scaler.fit(df)
        scaler.transform(df)

        self.assertEqual(scaler.scaler.mean_[0], 0.5)
        self.assertEqual(scaler.scaler.mean_[1], 1)
        self.assertEqual(scaler.scaler.var_[0], 0.25)
        self.assertEqual(scaler.scaler.var_[1], 1)
