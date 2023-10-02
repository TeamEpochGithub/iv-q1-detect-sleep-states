import pandas as pd
from src.preprocessing.pp import PP
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import gc


class SplitWindows(PP):

    def __init__(self, start_hour: float = 15):
        super().__init__()
        self.start_hour = start_hour
        self.window_size = 24 * 60 * 12
        self.steps_before = (start_hour * 60 * 12)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:

        # Pad the series with 0s
        # Loop through the series_ids
        tqdm.pandas()

        df = df.groupby('series_id').progress_apply(
            self.pad_series).reset_index(drop=True)
        print("Padding done")
        # Split the data into 24 hour windows per series_id
        df = df.groupby('series_id').progress_apply(
            self.preprocess_series).reset_index(0, drop=True)

        return df

    def preprocess_series(self, df: pd.DataFrame) -> pd.DataFrame:
        df['window'] = df.index // self.window_size
        return df

    def pad_series(self, group: pd.DataFrame) -> pd.DataFrame:

        # Garbage collect
        gc.collect()

        # Pad types
        pad_type = {'step': np.uint32, 'series_id': np.uint16, 'awake': np.uint8, 'enmo': np.float32, 'anglez': np.float32, 'timestamp': 'datetime64[ns]'}

        # Get current series_id
        curr_series_id = group['series_id'].iloc[0]

        # Find the timestamp of the first row for each series_id
        first_time = group['timestamp'].iloc[0]

        # Get initial seconds
        initial_steps = first_time.hour * 60 * 12 + first_time.minute * 12 + int(first_time.second / 5)

        # If index is 0, then the first row is at 15:00:00 so do nothing
        # If index is negative, time is before 15:00:00 and after 00:00:00 so add initial seconds and 9 hours
        amount_of_padding_start = 0
        if initial_steps < self.steps_before:
            amount_of_padding_start += (self.window_size - self.steps_before) + initial_steps
        else:
            amount_of_padding_start += initial_steps - self.steps_before


        start_pad_df = pd.DataFrame({'timestamp': [first_time - pd.Timedelta(seconds=(amount_of_padding_start - i + 1) * 5) for i in reversed(range(amount_of_padding_start))],
                                     'enmo': [0] * amount_of_padding_start,
                                     'anglez': [0] * amount_of_padding_start,
                                     'step': [-i for i in reversed(range(1, amount_of_padding_start + 1))],
                                     'series_id': [curr_series_id] * amount_of_padding_start,
                                     'awake': [2] * amount_of_padding_start
                                     })

        # Find the timestamp of the last row for each series_id
        last_time = group['timestamp'].iloc[-1]

        # Get last seconds
        last_steps = last_time.hour * 60 * 12 + last_time.minute * 12 + int(last_time.second / 5)

        # Pad the end with enmo = 0 and anglez = 0 and steps being relative to the last step until timestamp is 15:00:00
        amount_of_padding_end = 0
        if last_steps > self.steps_before:
            amount_of_padding_end += (self.steps_before - 1) + (self.window_size - last_steps)
        else:
            amount_of_padding_end += (self.steps_before - 1) - last_steps

        last_step = group['step'].iloc[-1]
        end_pad_df = pd.DataFrame({'timestamp': [last_time + pd.Timedelta(seconds=i * 5) for i in range(1, amount_of_padding_end + 1)],
                                   'enmo': [0] * amount_of_padding_end,
                                   'anglez': [0] * amount_of_padding_end,
                                   'step': [last_step + i for i in range(1, amount_of_padding_end + 1)],
                                   'series_id': [curr_series_id] * amount_of_padding_end,
                                   'awake': [2] * amount_of_padding_end
                                   })

        # Concatenate the dfs
        dfs_to_concat = [start_pad_df, group, end_pad_df]
        group = pd.concat(dfs_to_concat, ignore_index=True)
        group.astype(pad_type)

        return group
