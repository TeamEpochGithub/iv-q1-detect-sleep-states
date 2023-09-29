import pandas as pd
from src.preprocessing.pp import PP
import matplotlib.pyplot as plt

class SplitWindows(PP):

    def __init__(self, start_hour: float = 15):
        super().__init__()
        self.start_hour = start_hour

    def preprocess(self, df):
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Pad the series with 0s
        df = df.groupby('series_id').apply(self.pad_series).reset_index(drop=True)

        # Print length of series
        print(f"Length of series: {len(df)}")

        # Split the data into 24 hour windows per series_id
        df = df.groupby('series_id').apply(self.preprocess_series).reset_index(0, drop=True)


        # Plot the data
        for series_id, group in df.groupby("series_id"):
            # Plot enmo
            plt.plot(group["timestamp"], group["enmo"])
            # Plot window
            plt.plot(group["timestamp"], group["window"])
            plt.title(f"Series ID: {series_id}")
            plt.xlabel("Timestamp")
            plt.ylabel("ENMO")
            plt.show()
                


        print("Train data:")
        for series_id, group in df.groupby("series_id"):
            print(f"Series ID: {series_id}")
            for window in group["window"].unique():
                print(f"\tWindow: {window}")
        return df

    def preprocess_series(self, df):
        # Find the timestamp of the first row
        first_time = pd.to_datetime(df['timestamp'].iloc[0]).time()

        initial_seconds = first_time.hour * 3600 + first_time.minute * 60 + first_time.second

        # Calculate the number of rows per 24-hour window
        rows_per_window = int(24 * 60 * 60 / 5)

        # Find the index of the next row that has a timestamp of 15:00:00
        index_start = int((self.start_hour * 60 * 60 - initial_seconds) / 5)
        if index_start <= 0:
            index_start = rows_per_window - index_start

        # Create an empty column
        df['window'] = pd.Series(dtype='int')

        # Split the DataFrame into 24-hour windows
        df.iloc[:index_start, df.columns.get_loc('window')] = 0  # assign window 0
        for w, row in enumerate(range(index_start, len(df), rows_per_window)):
            df.iloc[row:row + rows_per_window-1, df.columns.get_loc('window')] = w + 1  # assign window 1 and further
        return df


    def pad_series(self, group):

        # Find the timestamp of the first row for each series_id
        first_time = group['timestamp'].iloc[0]

        # Get initial seconds
        initial_seconds = first_time.hour * 60 * 60 + \
            first_time.minute * 60 + first_time.second

        # Find the index of the first row that has a timestamp of 15:00:00
        index_start = int((15 * 60 * 60 - initial_seconds) / 5)

        # If the index is negative, pad the front with enmo = 0 and anglez = 0 and steps being relative to the first step
        if index_start < 0:
            index_start -= 1
            pad_df = pd.DataFrame({'timestamp': [first_time - pd.Timedelta(seconds=i * 5) for i in range(1, -index_start)],
                                'enmo': [0] * (-index_start - 1),
                                'anglez': [0] * (-index_start - 1),
                                'step': [group['step'].iloc[0] - i for i in range(1, -index_start)],
                                'series_id': [group['series_id'].iloc[0]] * (-index_start - 1)
                                })
            # Sort dataframe by step
            pad_df = pad_df.sort_values('step')
            group = pd.concat([pad_df, group], ignore_index=True)

        # Find the timestamp of the last row for each series_id
        last_time = group['timestamp'].iloc[-1]

        # Get last seconds
        last_seconds = last_time.hour * 60 * 60 + \
            last_time.minute * 60 + last_time.second

        # Find the difference between the last row and 15:00:00
        index_end = int((last_seconds - 15 * 60 * 60) / 5)

        # Pad the end with enmo = 0 and anglez = 0 and steps being relative to the last step until timestamp is 15:00:00
        if index_end > 0:
            # Time has to add next day as well
            index_end -= (24 * 60 * 12)
            pad_df = pd.DataFrame({'timestamp': [last_time + pd.Timedelta(seconds=i * 5) for i in range(1, -index_end)],
                                'enmo': [0] * (-index_end - 1),
                                'anglez': [0] * (-index_end - 1),
                                'step': [group['step'].iloc[-1] + i for i in range(1, -index_end)],
                                'series_id': [group['series_id'].iloc[-1]] * (-index_end - 1)
                                })
            # Sort dataframe by step
            pad_df = pad_df.sort_values('step')
            group = pd.concat([group, pad_df], ignore_index=True)
        elif index_end < 0:
            # Time has to add up to 15:00:00
            pad_df = pd.DataFrame({'timestamp': [last_time + pd.Timedelta(seconds=i * 5) for i in range(1, -index_end)],
                                'enmo': [0] * (-index_end - 1),
                                'anglez': [0] * (-index_end - 1),
                                'step': [group['step'].iloc[-1] + i for i in range(1, -index_end)],
                                'series_id': [group['series_id'].iloc[-1]] * (-index_end - 1)
                                })
            # Sort dataframe by step
            pad_df = pad_df.sort_values('step')
            group = pd.concat([group, pad_df], ignore_index=True)
        return group