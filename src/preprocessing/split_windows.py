import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..preprocessing.pp import PP


class SplitWindows(PP):
    """Splits the data into 24h windows

    A new column named window will be added that contains the window number for each row.
    """

    def __init__(self, start_hour: float = 15, **kwargs) -> None:
        """Initialize the SplitWindows class.

        :param start_hour: the hour of the day to start the window at. Default is 15.
        """
        super().__init__(**kwargs)

        self.start_hour = start_hour
        self.window_size = 24 * 60 * 12
        self.steps_before = (start_hour * 60 * 12)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by splitting it into 24h windows.

        :param df: the data without windows
        :return: the preprocessed data with window numbers
        """

        # Pad the series with 0s
        # Loop through the series_ids
        tqdm.pandas()

        df = df.groupby('series_id').progress_apply(
            self.pad_series).reset_index(drop=True)

        # Split the data into 24 hour windows per series_id
        df = df.groupby('series_id').progress_apply(
            self.preprocess_series).reset_index(0, drop=True)

        return df

    def preprocess_series(self, series: pd.DataFrame) -> pd.DataFrame:
        series['window'] = series.reset_index(
            0, drop=True).index // self.window_size
        series.astype({'window': np.uint8})
        return series

    def pad_series(self, group: pd.DataFrame) -> pd.DataFrame:

        # Garbage collect
        gc.collect()

        # Pad types
        pad_type = {'step': np.uint32, 'series_id': np.uint16,
                    'enmo': np.float32, 'anglez': np.float32, 'timestamp': 'datetime64[ns]'}
        if 'awake' in group.columns:
            pad_type['awake'] = np.uint8

        # Get current series_id
        curr_series_id = group['series_id'].iloc[0]

        # Find the timestamp of the first row for each series_id
        first_time = group['timestamp'].iloc[0]

        # Get initial seconds
        initial_steps = first_time.hour * 60 * 12 + \
            first_time.minute * 12 + int(first_time.second / 5)

        # If index is 0, then the first row is at 15:00:00 so do nothing
        # If index is negative, time is before 15:00:00 and after 00:00:00 so add initial seconds and 9 hours
        amount_of_padding_start = 0
        if initial_steps < self.steps_before:
            amount_of_padding_start += (self.window_size -
                                        self.steps_before) + initial_steps
        else:
            amount_of_padding_start += initial_steps - self.steps_before

        # Create numpy arrays of zeros for enmo, anglez, and awake
        enmo = np.zeros(amount_of_padding_start)
        anglez = np.zeros(amount_of_padding_start)

        # Create a numpy array of step values
        step = (-np.arange(1, amount_of_padding_start + 1))[::-1]

        # Create a numpy array of timestamps using date range
        timestamps = (first_time - np.arange(1, amount_of_padding_start + 1)
                      * pd.Timedelta(seconds=5))[::-1]

        # Create a numpy array of series ids
        series_id = np.full(amount_of_padding_start, curr_series_id)

        # Create the start padding dataframe
        start_pad_df = pd.DataFrame({'timestamp': timestamps,
                                    'enmo': enmo,
                                     'anglez': anglez,
                                     'step': step,
                                     'series_id': series_id})

        # only pad the awake column if it exists
        if 'awake' in group.columns:
            start_pad_df['awake'] = np.full(amount_of_padding_start, 2)

        # Find the timestamp of the last row for each series_id
        last_time = group['timestamp'].iloc[-1]

        # Pad the end with enmo = 0 and anglez = 0 and steps being relative to the last step until timestamp is 15:00:00
        amount_of_padding_end = 0
        amount_of_padding_end = 17280 - \
            ((len(start_pad_df) + len(group)) % 17280)

        last_step = group['step'].iloc[-1]

        # Create numpy arrays of zeros for enmo, anglez, and awake
        enmo = np.zeros(amount_of_padding_end)
        anglez = np.zeros(amount_of_padding_end)

        # Create a numpy array of step values
        step = np.arange(last_step + 1, last_step + amount_of_padding_end + 1)

        # Create a numpy array of timestamps using date range
        timestamps = last_time + \
            np.arange(1, amount_of_padding_end + 1) * pd.Timedelta(seconds=5)

        # Create a numpy array of series ids
        series_id = np.full(amount_of_padding_end, curr_series_id)

        # Create the end padding dataframe
        end_pad_df = pd.DataFrame({'timestamp': timestamps,
                                   'enmo': enmo,
                                   'anglez': anglez,
                                   'step': step,
                                   'series_id': series_id})

        # only pad the awake column if it exists
        if 'awake' in group.columns:
            end_pad_df['awake'] = np.full(amount_of_padding_end, 2)

        # Concatenate the dfs
        dfs_to_concat = [start_pad_df, group, end_pad_df]
        group = pd.concat(dfs_to_concat, ignore_index=True)
        group.astype(pad_type)

        return group
