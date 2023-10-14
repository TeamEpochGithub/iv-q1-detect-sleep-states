import json

import numpy as np
import pandas as pd
from tqdm import tqdm

STEPS_PER_MINUTE = 12
STEPS_PER_HOUR = 12 * 60
STEPS_PER_DAY = 12 * 60 * 24

dataset = 'train'


def fill_series_labels(series_id, series) -> None:
    series['awake'] = 2

    awake_col = series.columns.get_loc('awake')
    current_events = events[events["series_id"] == series_id]

    if len(current_events) == 0:
        series['awake'] = 2
        return

    # iterate over event labels and fill in the awake column segment by segment
    prev_step = 0
    prev_was_nan = False
    for _, row in current_events.iterrows():
        step = row['step']
        if np.isnan(step):
            prev_was_nan = True
            continue

        step = int(step)
        if prev_was_nan:
            series.iloc[prev_step:step, awake_col] = 2
        elif row['event'] == 'onset':
            series.iloc[prev_step:step, awake_col] = 1
        elif row['event'] == 'wakeup':
            series.iloc[prev_step:step, awake_col] = 0
        else:
            raise Exception(f"Unknown event type: {row['event']}")

        prev_step = step
        prev_was_nan = False

    # set the tail as unlabeled/NaN
    series.iloc[prev_step:, awake_col] = 3


first_timestamps = json.load(open(f'./data/processed/{dataset}/first_timestamps.json'))
events = pd.read_csv(f'./data/raw/{dataset}_events.csv')

pbar = tqdm(first_timestamps)
for series_id in pbar:
    series = pd.read_parquet(f'./data/processed/{dataset}/windowed/{series_id}.parquet')
    fill_series_labels(series_id, series)
    series.to_parquet(f'./data/processed/{dataset}/labeled/{series_id}.parquet')
