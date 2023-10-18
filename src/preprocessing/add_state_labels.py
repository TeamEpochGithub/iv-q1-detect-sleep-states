import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..preprocessing.pp import PP


class AddStateLabels(PP):
    """Adds state labels to each row of the data

    The state labels are added to the "awake" column based on the events csv file.
    The values are 0 for asleep, 1 for awake, and 2 for unlabeled.
    """

    def __init__(self, events_path: str, id_encoding_path: str, **kwargs: dict) -> None:
        """Initialize the AddStateLabels class.

        :param events_path: the path to the events csv file
        :param id_encoding_path: the path to the encoding file of the series id
        """
        super().__init__(**kwargs)

        self.events_path: str = events_path
        self.events: pd.DataFrame = pd.DataFrame()

        self.id_encoding_path: str = id_encoding_path
        self.id_encoding: dict = {}

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the preprocessing step.

        :param data: the data to preprocess
        :return: the preprocessed data
        :raises FileNotFoundError: If the events csv or id_encoding json file is not found
        """
        self.events = pd.read_csv(self.events_path)
        self.id_encoding = json.load(open(self.id_encoding_path))
        res = self.preprocess(data)
        del self.events
        return res

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by adding state labels to each row of the data.

        :param data: the data without state labels
        :return: the data with state labels added to the "awake" column
        """

        # Initialize the awake column as 42, to catch errors later (-1 not possible in uint8)
        data['awake'] = 42
        data['awake'] = data['awake'].astype('uint8')

        # apply encoding to events
        self.events['series_id'] = self.events['series_id'].map(self.id_encoding)

        # Hand-picked weird cases, with unlabeled, non-nan tails
        # the ids are hard-coded as full id strings, require encoding fist
        weird_series = ["0cfc06c129cc", "31011ade7c0a", "55a47ff9dc8a", "a596ad0b82aa", "a9a2f7fac455"]
        weird_series_encoded = []
        for id in weird_series:
            if id in self.id_encoding:
                weird_series_encoded.append(self.id_encoding[id])

        # iterate over the series and set the awake column
        tqdm.pandas()
        data = (data
                .groupby('series_id')
                .progress_apply(lambda x: self.fill_series_labels(x, weird_series_encoded))
                .reset_index(drop=True))

        return data

    def fill_series_labels(self, series, weird_series_encoded):
        awake_col = series.columns.get_loc('awake')
        series_id = series['series_id'].iloc[0]
        current_events = self.events[self.events["series_id"] == series_id]
        if len(current_events) == 0:
            series['awake'] = 2
            return series

        # iterate over event labels and fill in the awake column segment by segment
        prev_step = 0
        prev_was_nan = False
        for _, row in current_events.iterrows():
            step = row['step']
            if np.isnan(step):
                prev_was_nan = True
                continue

            step = int(step)
            if prev_was_nan:
                series.iloc[prev_step:step, awake_col] = 2
            elif row['event'] == 'onset':
                series.iloc[prev_step:step, awake_col] = 1
            elif row['event'] == 'wakeup':
                series.iloc[prev_step:step, awake_col] = 0
            else:
                raise Exception(f"Unknown event type: {row['event']}")

            prev_step = step
            prev_was_nan = False

        # set the tail based on the last event, unless it's a weird series, which has a NaN tail
        last_event = current_events['event'].tail(1).values[0]
        if prev_was_nan:
            series.iloc[prev_step:, awake_col] = 2
        elif series_id in weird_series_encoded:
            series.iloc[prev_step:, awake_col] = 2
        elif last_event == 'wakeup':
            series.iloc[prev_step:, awake_col] = 1
        elif last_event == 'onset':
            series.iloc[prev_step:, awake_col] = 0

        # TODO: shift?
        return series
