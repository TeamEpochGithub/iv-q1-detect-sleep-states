from unittest import TestCase
import numpy as np
import pandas as pd

from src.preprocessing.similarity_nan import similarity_nan

STEPS_PER_DAY = (24 * 60 * 60) // 5


class SimilarityTest(TestCase):
    def test_last_window_diff(self):
        # create a series where the second and last window are the same,
        # but are not exactly a multiple of STEPS_PER_DAY
        df = pd.DataFrame({
            'anglez': np.concatenate([
                np.arange(STEPS_PER_DAY),
                np.arange(STEPS_PER_DAY) + 1,
                np.arange(STEPS_PER_DAY) + 2,
                np.arange(STEPS_PER_DAY - 1000)])
        })

        # run the function
        df = similarity_nan(df)

        expected = np.concatenate([
            np.zeros(STEPS_PER_DAY - 1000), np.ones(1000),
            np.ones(STEPS_PER_DAY),
            np.ones(STEPS_PER_DAY),
            np.zeros(STEPS_PER_DAY - 1000)])

        self.assertTrue((expected == df['f_similarity_nan']).all())

