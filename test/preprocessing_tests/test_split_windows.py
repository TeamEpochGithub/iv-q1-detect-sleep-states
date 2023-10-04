import unittest

import pandas as pd
import numpy as np
from src.preprocessing.split_windows import SplitWindows


class MyTestCase(unittest.TestCase):
    def test_split_windows(self):
        split_windows = SplitWindows(15)

        # Create a two-day long dataframe, with 5 seconds per step
        # start 1 hour too early
        start_time = pd.Timestamp(year=2023, month=1, day=1, hour=16)
        # Add 1.5 days to the start time
        end_time = start_time + pd.Timedelta(days=2)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='5S')
        df = pd.DataFrame({'timestamp': timestamps, 'series_id': np.zeros(len(timestamps)), 'step': np.zeros(
            len(timestamps), dtype=np.uint32), 'awake': np.zeros(len(timestamps), dtype=np.uint8)})

        # define expected window boundaries
        window_1_start = int(24 * 60 * 60 / 5)
        window_2_start = window_1_start + int(24 * 60 * 60 / 5)

        df = split_windows.preprocess(df)

        # verify the two windows
        window0 = df.loc[0:window_1_start-1, 'window']
        window1 = df.loc[window_1_start:window_2_start-1, 'window']
        self.assertTrue((window0 == 0).all())
        self.assertTrue((window1 == 1).all())
