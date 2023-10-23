from unittest import TestCase

from src.feature_engineering.feature_engineering import FE, FEException
from src.feature_engineering.kurtosis import Kurtosis
from src.feature_engineering.mean import Mean
from src.feature_engineering.rotation import Rotation
from src.feature_engineering.skewness import Skewness
from src.feature_engineering.time import Time


class TestFE(TestCase):
    def test_from_config_single(self):
        self.assertIsInstance(
            FE.from_config_single({"kind": "kurtosis", "window_sizes": [5, 10], "features": ["anglez", "enmo"]}),
            Kurtosis)
        self.assertIsInstance(
            FE.from_config_single({"kind": "mean", "window_sizes": [5, 10], "features": ["anglez", "enmo"]}), Mean)
        self.assertIsInstance(
            FE.from_config_single({"kind": "skewness", "window_sizes": [5, 10], "features": ["anglez", "enmo"]}),
            Skewness)
        self.assertIsInstance(
            FE.from_config_single({"kind": "time", "day": True, "hour": True, "minute": True, "second": True}), Time)
        self.assertIsInstance(FE.from_config_single({"kind": "rotation"}), Rotation)
        self.assertRaises(FEException, FE.from_config_single, {"kind": "e"})

    def test_from_config(self):
        config: dict = {
            "feature_engineering": [
                {
                    "kind": "kurtosis",
                    "window_sizes": [5, 10],
                    "features": ["anglez", "enmo"]
                },
                {
                    "kind": "mean",
                    "window_sizes": [5, 10],
                    "features": ["anglez", "enmo"]
                },
                {
                    "kind": "skewness",
                    "window_sizes": [5, 10],
                    "features": ["anglez", "enmo"]
                },
                {
                    "kind": "time",
                    "day": True,
                    "hour": True,
                    "minute": True,
                    "second": True
                },
                {
                    "kind": "rotation",
                    "window_sizes": [100]
                }
            ]
        }

        # Test parsing
        fe_steps = FE.from_config(config["feature_engineering"])

        self.assertIsInstance(fe_steps[0], Kurtosis)
        self.assertListEqual(fe_steps[0].window_sizes, [5, 10])
        self.assertListEqual(fe_steps[0].features, ["anglez", "enmo"])
        self.assertIsInstance(fe_steps[1], Mean)
        self.assertListEqual(fe_steps[1].window_sizes, [5, 10])
        self.assertListEqual(fe_steps[1].features, ["anglez", "enmo"])
        self.assertIsInstance(fe_steps[2], Skewness)
        self.assertListEqual(fe_steps[2].window_sizes, [5, 10])
        self.assertListEqual(fe_steps[2].features, ["anglez", "enmo"])
        self.assertIsInstance(fe_steps[3], Time)
        self.assertTrue(fe_steps[3].day)
        self.assertTrue(fe_steps[3].hour)
        self.assertTrue(fe_steps[3].minute)
        self.assertTrue(fe_steps[3].second)
        self.assertIsInstance(fe_steps[4], Rotation)
        self.assertListEqual(fe_steps[4].window_sizes, [100])
