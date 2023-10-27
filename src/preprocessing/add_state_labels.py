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

    def __init__(self, events_path: str, id_encoding_path: str,
                 use_similarity_nan: bool, fill_limit: int, **kwargs: dict) -> None:
        """Initialize the AddStateLabels class.

        :param events_path: the path to the events csv file
        :param id_encoding_path: the path to the encoding file of the series id
        """
        super().__init__(**kwargs | {"kind": "add_state_labels"})

        self.events_path: str = events_path
        self.events: pd.DataFrame = pd.DataFrame()

        self.id_encoding_path: str = id_encoding_path
        self.id_encoding: dict = {}

        self.use_similarity_nan: bool = use_similarity_nan
        self.fill_limit = fill_limit

    def __repr__(self) -> str:
        """Return a string representation of a AddStateLabels object"""
        return f"{self.__class__.__name__}(events_path={self.events_path}, id_encoding_path={self.id_encoding_path})"

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
        weird_series_encoded = [self.id_encoding[sid] for sid in weird_series if sid in self.id_encoding]

        # iterate over the series and set the awake column
        tqdm.pandas()
        if self.use_similarity_nan:
            similarity_cols = [col for col in data.columns if col.endswith('similarity_nan')]
            if len(similarity_cols) == 0:
                raise Exception("No (f_)similarity_nan column found, but use_similarity_nan is set to True")
            data = (data
                    .groupby('series_id')
                    .progress_apply(lambda x: self.set_awake_with_similarity(x, ))
                    .reset_index(drop=True))
        else:
            data = (data
                    .groupby('series_id')
                    .progress_apply(lambda x: self.set_awake(x, weird_series_encoded))
                    .reset_index(drop=True))

        return data

    def set_awake(self, series, weird_series_encoded):
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

    def set_awake_with_similarity(self, series, similarity_col_name):
        """Set awake using nan_similarity, adds labels of 2 (nan) or 3 (unlabeled)"""
        awake_col = series.columns.get_loc('awake')
        series_id = series['series_id'].iloc[0]
        current_events = self.events[self.events["series_id"] == series_id]
        if len(current_events) == 0:
            series['awake'] = 2
            return series

        # initialize as unlabeled, and set nan based on similarity_nan
        series['awake'] = 3
        series['awake'][series[similarity_col_name] == 0] = 2

        # iterate over event labels and fill in the awake column segment by segment
        prev_step = 0
        prev_event = None

        fill_value_before = {
            "onset": 1,
            "wakeup": 0,
        }
        fill_value_after = {
            "onset": 0,
            "wakeup": 1,
        }

        for _, row in current_events.iterrows():
            step = row['step']

            if np.isnan(step):
                if prev_event != 'nan' and prev_event is not None:
                    # transition from non-nan to nan
                    self.fill_forward(awake_col, fill_value_after[prev_event], prev_step, series)
                prev_event = 'nan'
            else:
                event = row['event']
                if prev_event == 'nan':
                    # transition from nan to non-nan
                    self.fill_backward(awake_col, fill_value_before[event], prev_step, series, step)
                else:
                    # non-nan to non-nan segment
                    series.iloc[prev_step:step, awake_col] = fill_value_before[event]

                prev_step = step
                prev_event = event

        # fill in the tail of the series after the last event
        self.fill_forward(awake_col, prev_event, prev_step, series)
        return series

    def fill_backward(self, awake_col, fill_value, prev_step, series, step):
        """Fill in the awake column backwards from step to the last non-nan similar value, up to a limit"""
        search_slice = series.iloc[prev_step:step, awake_col]
        slice_similar_mask = (search_slice == 2)

        # weird trick, argmax returns the index of the first occurrence of the max value,
        # so we reverse it twice to get the last index where the mask is 1 (the max value)
        last_similar = slice_similar_mask[::-1].argmax()
        start_of_fill = step - last_similar

        start_of_fill = max(start_of_fill, step - self.fill_limit)
        series.iloc[start_of_fill:step, awake_col] = fill_value

    def fill_forward(self, awake_col, fill_value, prev_step, series):
        """Fill in the awake column forward from prev_step to the first non-nan similar value, up to a limit"""
        search_slice = series.iloc[prev_step:prev_step + self.fill_limit, awake_col]
        slice_similar_mask = (search_slice == 2)
        first_similar = slice_similar_mask.argmax()
        end_of_fill = prev_step + first_similar
        end_of_fill = min(end_of_fill, prev_step + self.fill_limit)
        series.iloc[prev_step:end_of_fill, awake_col] = fill_value
