import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..logger.logger import logger
from ..preprocessing.pp import PP

from line_profiler_pycharm import profile

class AddStateLabels(PP):
    """Adds state labels to each row of the data

    The state labels are added to the "awake" column based on the events csv file.
    The values are 0 for asleep, 1 for awake, and 2 for unlabeled.
    """

    def __init__(self, events_path: str, **kwargs) -> None:
        """Initialize the AddStateLabels class.

        :param events_path: the path to the events csv file
        :raises PPException: If no path to the events csv file is given
        """
        super().__init__(**kwargs)

        self.events_path: str = events_path
        self.events: pd.DataFrame = pd.DataFrame()

        # TODO Don't hardcode the file name, add it as a parameter in the config.json #99
        self.id_encoding_path: str = "series_id_encoding.json"
        self.id_encoding: dict = {}

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the preprocessing step.

        :param data: the data to preprocess
        :return: the preprocessed data
        :raises FileNotFoundError: If the events csv or id_encoding json file is not found
        """
        self.events = pd.read_csv(self.events_path)
        self.id_encoding = json.load(open(self.id_encoding_path))
        return self.preprocess(data)

    @profile
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
        awake_col = data.columns.get_loc('awake')
        for id, series in tqdm(data.groupby('series_id')):
            current_events = self.events[self.events["series_id"] == id]
            if len(current_events) == 0:
                series['awake'] = 2

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
            if id in weird_series_encoded:
                series.iloc[prev_step:, awake_col] = 2
            if current_events['event'].iloc[-1] == 'wakeup':
                series.iloc[prev_step:, awake_col] = 1
            elif current_events['event'].iloc[-1] == 'onset':
                series.iloc[prev_step:, awake_col] = 0

            # TODO: shift?

        return data
