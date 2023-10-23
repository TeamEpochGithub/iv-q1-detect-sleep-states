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
            'anglez': np.concat([
                np.arange(STEPS_PER_DAY),
                np.arange(STEPS_PER_DAY)+1,
                np.arange(STEPS_PER_DAY)+2,
                np.arange(STEPS_PER_DAY - 1000)])
        })

        # run the function
        df = similarity_nan(df)

        self.assertTrue(df['similarity_nan']=)
