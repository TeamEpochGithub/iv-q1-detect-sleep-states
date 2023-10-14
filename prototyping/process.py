import json
import os

import pandas as pd
import polars as pl
from tqdm import tqdm

STEPS_PER_MINUTE = 12
STEPS_PER_HOUR = 12 * 60
STEPS_PER_DAY = 12 * 60 * 24

dataset = 'train'

print(os.getcwd())
data = pd.read_parquet(f'./data/raw/{dataset}_series.parquet')

print('loaded parquet')
print(data.dtypes)
print(data.head())

first_timestamps = dict()

pbar = tqdm(data.groupby(['series_id']))
for series_id, group in pbar:

    # drop series
    sid = group['series_id'].iloc[0]
    group.drop(['series_id'], axis=1, inplace=True)

    # save first timestamp
    first_timestamp = group.iloc[0]['timestamp']
    first_timestamps[sid] = first_timestamp

    # convert timestamp to datetime
    pbar.set_description(
        f"Processing series {sid}, of {len(group) / STEPS_PER_DAY:.1f} days, converting timestamp to datetime")
    timestamp_pl = pl.from_pandas(pd.Series(group.timestamp, copy=False))
    timestamp_pl = timestamp_pl.str.slice(0, 19)
    timestamp_pl = timestamp_pl.str.to_datetime(format="%Y-%m-%dT%H:%M:%S", time_unit='ms')
    group['timestamp'] = timestamp_pl

    # convert timestamp to hours and minutes
    pbar.set_description(
        f"Processing series {sid}, of {len(group) / STEPS_PER_DAY:.1f} days, "
        f"converting timestamp to hours and minutes")

    group['hour'] = group['timestamp'].dt.hour
    group['minute'] = group['timestamp'].dt.minute
    group.drop(['timestamp'], axis=1, inplace=True)

    # use step as index
    group.set_index('step', inplace=True)

    # save the series
    group.to_parquet(f'./data/processed/{dataset}/compressed/{sid}.parquet')

print(group.head())

# write first timestamps to file
with open(f'./data/processed/{dataset}/first_timestamps.json', 'w') as f:
    json.dump(first_timestamps, f)
