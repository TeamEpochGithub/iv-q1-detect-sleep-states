from unittest import TestCase

from src.feature_engineering.kurtosis import Kurtosis


class TestKurtosis(TestCase):

    def test_repr(self):
        self.assertEqual("Kurtosis(window_sizes=[5, 10], features=['anglez', 'enmo'])",
                         repr(Kurtosis(window_sizes=[5, 10], features=["anglez", "enmo"])))
