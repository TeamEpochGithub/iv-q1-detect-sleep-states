from src.preprocessing.pp import PP
import numpy as np
from tqdm import tqdm
import pandas as pd


class SimilarityNan(PP):
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas()

        data = data.groupby('series_id').progress_apply(similarity_nan).reset_index(drop=True)
        return data


def similarity_nan(series):
    """Computes the similarity of each point to that at the same time in the last 24h hours"""
    STEPS_PER_DAY = (24 * 60 * 60) // 5

    if len(series) < STEPS_PER_DAY:
        return np.zeros(len(series))

    # get an array of the anglez padded with NaNs to make it a multiple of STEPS_PER_DAY
    feature_np = series['anglez'].to_numpy()
    padded = np.pad(feature_np, (0, STEPS_PER_DAY - (len(series) % STEPS_PER_DAY)), 'constant', constant_values=0)

    # make the comparison matrix
    num_days = len(padded) // STEPS_PER_DAY
    comparison = np.empty((num_days, len(padded)))
    for day in range(num_days):
        day_data = padded[day * STEPS_PER_DAY:(day + 1) * STEPS_PER_DAY]
        comparison[day, :] = np.tile(day_data, num_days)
        comparison[day, :] = np.abs(comparison[day, :] - padded)

    # set the self-comparison to inf
    for day in range(num_days):
        comparison[day, day * STEPS_PER_DAY:(day + 1) * STEPS_PER_DAY] = np.inf

    min_comparison = np.min(comparison, axis=0)

    diff = min_comparison[:len(series)]

    series['f_similarity_nan'] = diff
    return series
