import gc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import data_info
from ..preprocessing.pp import PP


@dataclass
class SplitWindows(PP):
    """Splits the data into 24h windows

    A new column named window will be added that contains the window number for each row.
    :param start_hour: the hour of the day to start the window at. Default is 15:00.
    """
    start_hour: float = 15

    _steps_before: float = field(init=False, default=0, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize the SplitWindows class."""
        self._steps_before = (self.start_hour * 60 * 12)

    def __repr__(self) -> str:
        """Return a string representation of a SplitWindows object"""
        return f"{self.__class__.__name__}(start_hour={self.start_hour})"

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by splitting it into 24h windows.

        :param df: the data without windows
        :return: the preprocessed data with window numbers
        """

        # Pad the series with 0s
        # Loop through the series_ids

        for sid in tqdm(data.keys()):
            data[sid] = self.pad_series(data[sid]).reset_index(0, drop=True)
            data[sid] = self.preprocess_series(data[sid]).reset_index(0, drop=True)

        # df = self.clip_enmo_df(df)
        return data

    def preprocess_series(self, series: pd.DataFrame) -> pd.DataFrame:
        series['window'] = series.reset_index(
            0, drop=True).index // data_info.window_size
        series.astype({'window': np.uint8})
        return series

    # Adding clipping makes the score worse after some testing, but we can keep it here for now
    # def clip_enmo_df(self, df: pd.DataFrame) -> pd.DataFrame:
    #     df['enmo'] = df['enmo'].clip(upper=df['enmo'].mean() + 5 * df['enmo'].std())
    #     return df

    def pad_series(self, group: pd.DataFrame) -> pd.DataFrame:

        # Garbage collect
        gc.collect()

        # Pad types
        pad_type = {'step': np.int32,
                    'enmo': np.float32, 'anglez': np.float32, 'timestamp': 'datetime64[ns]', 'utc': np.uint16}

        optionals = {  # (data_type, default_val for padding)
            'awake': (np.uint8, 3),
            'state-onset': (np.float32, 0),
            'state-wakeup': (np.float32, 0),
            'f_similarity_nan': (np.float32, 0),
            'similarity_nan': (np.float32, 0),
            'utc': (np.uint16, 0),
        }

        # set the data types for the optional columns
        for col, (data_type, default_val) in optionals.items():
            if col in group.columns:
                pad_type[col] = data_type

        # Find the timestamp of the first row for each series_id
        first_time = group['timestamp'].iloc[0]

        # Get initial seconds
        initial_steps = first_time.hour * 60 * 12 + first_time.minute * 12 + int(first_time.second / 5)

        # If index is 0, then the first row is at 15:00:00 so do nothing
        # If index is negative, time is before 15:00:00 and after 00:00:00 so add initial seconds and 9 hours
        amount_of_padding_start = 0
        if initial_steps < self._steps_before:
            amount_of_padding_start += (data_info.window_size -
                                        self._steps_before) + initial_steps
        else:
            amount_of_padding_start += initial_steps - self._steps_before

        # Create a numpy array of step values
        step = (-np.arange(1, amount_of_padding_start + 1))[::-1]

        # Create a numpy array of timestamps using date range
        timestamps = (first_time - np.arange(1, amount_of_padding_start + 1)
                      * pd.Timedelta(seconds=5))[::-1]

        # Create the start padding dataframe
        start_pad_df = pd.DataFrame({'timestamp': timestamps,
                                     'enmo': np.zeros(amount_of_padding_start),
                                     'anglez': np.zeros(amount_of_padding_start),
                                     'utc': np.zeros(amount_of_padding_start),
                                     'step': step})

        # only pad the optional columns if they exist
        for col, (data_type, default_val) in optionals.items():
            if col in group.columns:
                start_pad_df[col] = np.full(amount_of_padding_start, default_val)

        # Find the timestamp of the last row for each series_id
        last_time = group['timestamp'].iloc[-1]

        # Pad the end with enmo = 0 and anglez = 0 and steps being relative to the last step until timestamp is 15:00:00
        amount_of_padding_end = data_info.window_size - ((len(start_pad_df) + len(group)) % data_info.window_size)

        last_step = group['step'].iloc[-1]

        # Create a numpy array of step values
        step = np.arange(last_step + 1, last_step + amount_of_padding_end + 1)

        # Create a numpy array of timestamps using date range
        timestamps = last_time + np.arange(1, amount_of_padding_end + 1) * pd.Timedelta(seconds=5)

        # Create the end padding dataframe
        end_pad_df = pd.DataFrame({'timestamp': timestamps,
                                   'enmo': np.zeros(amount_of_padding_end),
                                   'anglez': np.zeros(amount_of_padding_end),
                                   'utc': np.zeros(amount_of_padding_end),
                                   'step': step})

        # only pad the optional columns if they exist
        for col, (data_type, default_val) in optionals.items():
            if col in group.columns:
                end_pad_df[col] = np.full(amount_of_padding_end, default_val)

        # Concatenate the dfs
        dfs_to_concat = [start_pad_df, group, end_pad_df]
        group = pd.concat(dfs_to_concat, ignore_index=True)

        return group.astype(pad_type)
