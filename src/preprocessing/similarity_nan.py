from collections.abc import MutableMapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import data_info
from src.logger.logger import logger
from src.preprocessing.pp import PP


@dataclass
class SimilarityNan(PP):
    as_feature: bool = False

    def preprocess(self, data: MutableMapping[str, pd.DataFrame]) -> MutableMapping[str, pd.DataFrame]:
        tqdm.pandas()
        for sid in data.keys():
            data[sid] = self.similarity_nan(data[sid])
        return data

    def similarity_nan(self, series: pd.DataFrame) -> pd.DataFrame:
        """Computes the similarity of each point to that at the same time in the last 24h hours"""
        col_name = 'f_similarity_nan' if self.as_feature else 'similarity_nan'

        if len(series) < data_info.window_size:
            logger.warning("Series is shorter than a day, setting similarity to 1. Should never happen...")
            series['f_similarity_nan'] = 1
            return series

        # pad the series to a multiple of steps per day
        feature_np = series['anglez'].to_numpy()
        padded = np.pad(feature_np, (0, data_info.window_size - (len(series) % data_info.window_size)), 'constant',
                        constant_values=0)

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
