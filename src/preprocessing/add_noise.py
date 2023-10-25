import numpy as np
import pandas as pd

from ..preprocessing.pp import PP


class AddNoise(PP):
    """Adds noise to the data

    Adds random Gaussian distributed noise to the "anglez" column.
    """

    def __init__(self, **kwargs: dict) -> None:
        """Initialize the AddNoise class"""
        super().__init__(**kwargs | {"kind": "add_noise"})

    def __repr__(self) -> str:
        """Return a string representation of a AddNoise object"""
        return f"{self.__class__.__name__}()"

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by adding noise to the data.

        It creates a new column with the cumulative sum of the anglez column

        :param data: the data without noise
        :return: the data with noise added to the "anglez" column
        """
        data['anglez'] = data['anglez'] + np.random.normal(0, 0.1, len(data['anglez']))
        return data
