import pandas as pd
from src.preprocessing.pp import PP


class SplitWindows(PP):

    def __init__(self, start_hour: float = 15):
        super().__init__()
        self.start_hour = start_hour

    def preprocess(self, df):
        # Find the timestamp of the first row
        first_time = pd.to_datetime(df['timestamp'].iloc[0]).time()

        initial_seconds = first_time.hour * 3600 + first_time.minute * 60 + first_time.second

        # Find the index of the first row that has a timestamp of 15:00:00
        index_start = int((self.start_hour * 60 * 60 - initial_seconds) / 5)

        # Calculate the number of rows per 24-hour window
        rows_per_window = int(24 * 60 * 60 / 5)

        # Create an empty column
        df['window'] = pd.Series(dtype='int')

        # Split the DataFrame into 24-hour windows
        df.loc[:index_start-1, 'window'] = 0  # assign window 0
        for w, row in enumerate(range(index_start, len(df), rows_per_window)):
            df.loc[row:row + rows_per_window-1, 'window'] = w + 1  # assign window 1 and further
        return df
