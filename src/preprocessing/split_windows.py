import pandas as pd
from src.preprocessing.pp import PP


class SplitWindows(PP):

    def __init__(self, start_hour: float = 15):
        super().__init__()
        self.start_hour = start_hour

    def preprocess(self, df):
        # Find the timestamp of the first row for each series_id
        df['first_time'] = df.groupby('series_id')['timestamp'].transform('first')
        
        # Turn timestamp into datetime
        df['first_time'] = df['first_time'].apply(lambda x: pd.to_datetime(x).time())
        
        # Get initial seconds
        df['initial_seconds'] = df['first_time'].apply(lambda x: x.hour * 60 * 60 + x.minute * 60 + x.second)

        # Find the index of the first row that has a timestamp of 15:00:00
        df['index_start'] = ((self.start_hour * 60 * 60 - df['initial_seconds']) / 5).astype(int)

        # If the index is negative, add 24 hours
        df.loc[df['index_start'] < 0, 'index_start'] = 1

        # Calculate the number of rows per 24-hour window
        df['rows_per_window'] = int(24 * 60 * 12)

        # Create an empty column
        df['window'] = pd.Series(dtype='int')

        # Split the DataFrame into 24-hour windows
        df.loc[:df['index_start']-1, 'window'] = 0
        for w, row in enumerate(range(df['index_start'], len(df), df['rows_per_window'])):
            df.loc[row:row + df['rows_per_window']-1, 'window'] = w + 1


        print("Train data:")
        for series_id, group in df.groupby("series_id"):
            print(f"Series ID: {series_id}")
            for window in group["window"].unique():
                print(f"\tWindow: {window}")
        return df
