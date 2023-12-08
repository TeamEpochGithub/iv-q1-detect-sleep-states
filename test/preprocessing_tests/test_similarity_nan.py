from unittest import TestCase
import numpy as np
import pandas as pd

from src import data_info
from src.preprocessing.similarity_nan import SimilarityNan


class SimilarityTest(TestCase):
    def test_repr(self):
        self.assertEqual("SimilarityNan(as_feature=False)", SimilarityNan(as_feature=False).__repr__())

    def test_last_window_diff(self):
        # create a series where the second and last window are the same,
        # but are not exactly a multiple of STEPS_PER_DAY
        data_info.window_size = 17280
        df = pd.DataFrame({
            'anglez': np.concatenate([
                np.arange(17280),
                np.arange(17280) + 1,
                np.arange(17280) + 2,
                np.arange(17280 - 1000)])
        })

        # run the function
        sn = SimilarityNan(as_feature=True)
        df = sn.similarity_nan(df)

        expected = np.concatenate([
            np.zeros(17280 - 1000), np.ones(1000),
            np.ones(17280),
            np.ones(17280),
            np.zeros(17280 - 1000)])

        self.assertTrue((expected == df['f_similarity_nan']).all())
