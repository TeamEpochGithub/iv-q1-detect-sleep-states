from unittest import TestCase

import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer, \
    QuantileTransformer, Normalizer

from src.scaler.scaler import Scaler, ScalerException


class TestScaler(TestCase):
    def test_get_scaler(self):
        self.assertIsInstance(Scaler("standard").scaler, StandardScaler)
        self.assertIsInstance(Scaler("minmax").scaler, MinMaxScaler)
        self.assertIsInstance(Scaler("robust").scaler, RobustScaler)
        self.assertIsInstance(Scaler("maxabs").scaler, MaxAbsScaler)
        self.assertIsInstance(Scaler("powertransformer").scaler, PowerTransformer)
        self.assertIsInstance(Scaler("quantiletransformer").scaler, QuantileTransformer)
        self.assertIsInstance(Scaler("normalizer").scaler, Normalizer)
        self.assertIsNone(Scaler("none").scaler)
        self.assertRaises(ScalerException, Scaler, "unknown")

    def test_scale_standard(self):
        scaler: Scaler = Scaler("standard")

        df = pd.DataFrame({"enmo": [0, 1],
                           "anglez": [0, 2],
                           "trash": [123, 321]})

        scaler.fit(df)
        scaler.transform(df)

        self.assertEqual(scaler.scaler.mean_[0], 0.5)
        self.assertEqual(scaler.scaler.mean_[1], 1)
        self.assertEqual(scaler.scaler.var_[0], 0.25)
        self.assertEqual(scaler.scaler.var_[1], 1)
