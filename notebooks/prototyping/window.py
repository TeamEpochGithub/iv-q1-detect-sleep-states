import json

import pandas as pd
from tqdm import tqdm

STEPS_PER_MINUTE = 12
STEPS_PER_HOUR = 12 * 60
STEPS_PER_DAY = 12 * 60 * 24

dataset = 'train'

first_timestamps = json.load(open(f'./data/processed/{dataset}/first_timestamps.json'))

pbar = tqdm(first_timestamps)
for series_id in pbar:
    series = pd.read_parquet(f'./data/processed/{dataset}/compressed/{series_id}.parquet')
    first_timestamp = first_timestamps[series_id]

    # find when this series starts
    first_time = pd.to_datetime(first_timestamp[:19], format="%Y-%m-%dT%H:%M:%S")
    first_time_steps = (first_time.hour * 3600 + first_time.minute * 60 + first_time.second) // 5

    # calculate offset, so that the window 1 will start at 15:00
    three_pm_steps = 15 * STEPS_PER_HOUR
    offset = three_pm_steps - first_time_steps
    if offset < 0:
        offset += STEPS_PER_DAY

    series['window'] = ((series.index.astype(int) - offset) // STEPS_PER_DAY)+1
    series.to_parquet(f'./data/processed/{dataset}/windowed/{series_id}.parquet')
