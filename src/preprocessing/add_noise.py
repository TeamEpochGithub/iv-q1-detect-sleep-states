from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..preprocessing.pp import PP


@dataclass
class AddNoise(PP):
    """Adds noise to the data

    Adds random Gaussian distributed noise to the "anglez" column.
    """

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by adding noise to the data.

        It creates a new column with the cumulative sum of the anglez column

        :param data: the data without noise
        :return: the data with noise added to the "anglez" column
        """
        data['anglez'] = data['anglez'] + np.random.normal(0, 0.1, len(data['anglez']))
        return data
