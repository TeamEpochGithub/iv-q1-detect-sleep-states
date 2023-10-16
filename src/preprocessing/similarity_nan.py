from src.preprocessing.pp import PP
import numpy as np
from tqdm import tqdm
import pandas as pd


class SimilarityNan(PP):
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas()

        data = data.groupby('series_id').progress_apply(last_window_diff)
        return data


def last_window_diff(series):
    STEPS_PER_DAY = (24 * 60 * 60) // 5

    if len(series) < STEPS_PER_DAY:
        return np.zeros(len(series))

    last_24h = series['anglez'].iloc[-STEPS_PER_DAY:]

    # create a comparison series, consisting of the last 24 repeated to the size of the series, backwards from the end
    comparison = np.tile(last_24h.to_numpy(), len(series) // STEPS_PER_DAY + 1)[-len(series):]

    # compute the absolute difference
    diff = np.abs(series['anglez'].to_numpy() - comparison)

    # pad left with the average diff and reshape
    diff_padded_left = np.pad(diff, (0, STEPS_PER_DAY - len(diff) % STEPS_PER_DAY), constant_values=np.mean(diff))
    reshaped = diff_padded_left.reshape(-1, STEPS_PER_DAY)

    # replace the diff of the last day with that of the most similar day
    # (else it would always be 0 by comparing to itself)
    avg_similarity = np.mean(reshaped, axis=1)
    best = np.argmin(avg_similarity[:-1])
    diff[-STEPS_PER_DAY:] = reshaped[best]

    series['similarity_nan'] = diff
    return series
