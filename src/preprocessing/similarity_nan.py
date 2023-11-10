from src import data_info
from src.preprocessing.pp import PP
import numpy as np
from tqdm import tqdm
import pandas as pd
from src.logger.logger import logger


class SimilarityNan(PP):
    # TODO Add docstrings for the class and preprocess function and complete the docstring for the similarity_nan function
    # TODO Add type hints to the similarity_nan function

    def __init__(self, as_feature: bool = False, **kwargs: dict) -> None:
        super().__init__(**kwargs | {"kind": "similarity_nan"})
        self.as_feature = as_feature

    def __repr__(self) -> str:
        """Return a string representation of a SimilarityNan object"""
        return f"{self.__class__.__name__}(as_feature={self.as_feature})"

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas()
        data = (data.groupby('series_id')
                .progress_apply(self.similarity_nan)
                .reset_index(0, drop=True))
        return data

    def similarity_nan(self, series):
        """Computes the similarity of each point to that at the same time in the last 24h hours"""
        col_name = 'f_similarity_nan' if self.as_feature else 'similarity_nan'

        if len(series) < data_info.window_size:
            logger.warning(f"Series {series.iloc[0]['series_id']} is shorter than a day,"
                           f" setting similarity to 1. Should never happen...")
            series['f_similarity_nan'] = 1
            return series

        # pad the series to a multiple of steps per day
        feature_np = series['anglez'].to_numpy()
        padded = np.pad(feature_np, (0, data_info.window_size - (len(series) % data_info.window_size)), 'constant', constant_values=0)

        # compute the absolute difference between each day
        days = len(padded) // data_info.window_size
        comparison = np.empty((days, len(padded)))
        for day in range(days):
            day_data = padded[day * data_info.window_size: (day + 1) * data_info.window_size]
            tiled = np.tile(day_data, days)
            comparison[day] = np.abs(tiled - padded)

        # set the self comparisons to inf
        for day in range(days):
            comparison[day, day * data_info.window_size: (day + 1) * data_info.window_size] = np.inf

        # get the minimum
        diff = np.min(comparison, axis=0)

        # add the diff to the series as a column of float32
        series[col_name] = diff[:len(feature_np)].astype(np.float32)
        return series
