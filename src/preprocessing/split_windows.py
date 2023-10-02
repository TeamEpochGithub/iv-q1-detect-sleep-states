import pandas as pd
from src.preprocessing.pp import PP
from matplotlib import pyplot as plt


class SplitWindows(PP):

    def __init__(self, start_hour: float = 15):
        super().__init__()
        self.start_hour = start_hour

    def preprocess(self, df):

        # Pad the series with 0s
        df = df.groupby('series_id').apply(
            self.pad_series).reset_index(drop=True)
        print("Padding done")
        # Split the data into 24 hour windows per series_id
        df = df.groupby('series_id').apply(
            self.preprocess_series).reset_index(0, drop=True)

        return df

    def preprocess_series(self, df):
        # Find the timestamp of the first row
        first_time = df['timestamp'].iloc[0]

        initial_seconds = first_time.hour * 3600 + \
            first_time.minute * 60 + first_time.second

        # Calculate the number of rows per 24-hour window
        rows_per_window = int(24 * 60 * 60 / 5)

        # Find the index of the next row that has a timestamp of 15:00:00
        index_start = int((self.start_hour * 60 * 60 - initial_seconds) / 5)
        if index_start <= 0:
            index_start = rows_per_window - index_start

        # Create an empty column
        df['window'] = pd.Series(dtype='int')

        # Split the DataFrame into 24-hour windows
        df.iloc[:index_start, df.columns.get_loc(
            'window')] = 0  # assign window 0
        for w, row in enumerate(range(index_start, len(df), rows_per_window)):
            # assign window 1 and further
            df.iloc[row:row + rows_per_window,
                    df.columns.get_loc('window')] = w + 1
        return df

    def pad_series(self, group):

        # Find the timestamp of the first row for each series_id
        first_time = group['timestamp'].iloc[0]

        # Get initial seconds
        initial_steps = first_time.hour * 60 * 12 + \
            first_time.minute * 12 + int(first_time.second / 5)

        # If index is 0, then the first row is at 15:00:00 so do nothing
        # If index is negative, time is before 15:00:00 and after 00:00:00 so add initial seconds and 9 hours
        amount_of_padding_start = 0
        if initial_steps < (15 * 60 * 12):
            amount_of_padding_start += (9 * 60 * 12) + initial_steps
        else:
            amount_of_padding_start += initial_steps - (15 * 60 * 12)
        start_pad_df = pd.DataFrame({'timestamp': [first_time - pd.Timedelta(seconds=i * 5) for i in range(1, amount_of_padding_start + 1)],
                                     'enmo': [0] * amount_of_padding_start,
                                     'anglez': [0] * amount_of_padding_start,
                                     'step': [group['step'].iloc[0] - i for i in range(1, amount_of_padding_start + 1)],
                                     'series_id': [group['series_id'].iloc[0]] * amount_of_padding_start,
                                     'awake': [2] * amount_of_padding_start
                                     })
        start_pad_df = start_pad_df.sort_values('step')
        group = pd.concat([start_pad_df, group], ignore_index=True)

        # Find the timestamp of the last row for each series_id
        last_time = group['timestamp'].iloc[-1]

        # Get last seconds
        last_steps = last_time.hour * 60 * 12 + \
            last_time.minute * 12 + int(last_time.second / 5)

        # Pad the end with enmo = 0 and anglez = 0 and steps being relative to the last step until timestamp is 15:00:00
        amount_of_padding_end = 0
        if last_steps > (15 * 60 * 12):
            amount_of_padding_end += (15 * 60 * 12 - 1) + (24 * 60 * 12 - last_steps)
        else:
            amount_of_padding_end += (15 * 60 * 12 - 1) - last_steps
        end_pad_df = pd.DataFrame({'timestamp': [last_time + pd.Timedelta(seconds=i * 5) for i in range(1, amount_of_padding_end + 1)],
                                     'enmo': [0] * amount_of_padding_end,
                                     'anglez': [0] * amount_of_padding_end,
                                     'step': [group['step'].iloc[-1] + i for i in range(1, amount_of_padding_end + 1)],
                                     'series_id': [group['series_id'].iloc[-1]] * amount_of_padding_end,
                                     'awake': [2] * amount_of_padding_end
                                     })
        end_pad_df = end_pad_df.sort_values('step')
        group = pd.concat([group, end_pad_df], ignore_index=True)

        # Create plot to check if padding is correct
        plt.plot(group['timestamp'], group['enmo'])
        plt.show()
        return group
