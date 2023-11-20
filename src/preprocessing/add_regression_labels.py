from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import data_info
from ..logger.logger import logger
from ..preprocessing.pp import PP, PPException

import json


@dataclass
class AddRegressionLabels(PP):
    """Preprocessing step that adds the event labels to the data

    This will add the event labels to the data by using the event data.

    :param events_path: the path to the events csv file
    :param id_encoding_path: the path to the encoding file of the series id
    """
    events_path: str
    id_encoding_path: str

    _events: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False, compare=False)
    _id_encoding: dict = field(init=False, default_factory=dict, repr=False, compare=False)

    def run(self, data: dict) -> dict:
        """Run the preprocessing step.
        :param data: the data to preprocess
        :return: the preprocessed data with the event labels columns ("onset", "wakeup", "onset-NaN", "wakeup-NaN")
        :raises FileNotFoundError: If the events csv or id_encoding json file is not found
        :raises PPException: If the window column is not present
        """

        # If window column is present, raise an exception
        NO_WINDOW_COLUMN_ERROR = "No window column. Did you run SplitWindows before?"
        if "window" not in data[0].columns:
            logger.critical(NO_WINDOW_COLUMN_ERROR)
            raise PPException(NO_WINDOW_COLUMN_ERROR)
        if "hot-asleep" in data[0].columns:
            logger.warning(
                "Hot encoded columns are present (hot-NaN, hot-awake, hot-asleep) for state segmentation models. This can cause issues when also adding regression labels."
                "Make sure your model takes the correct features.")
        if "state-onset" in data[0].columns:
            logger.warning(
                "State-onset column is present, for state segmentation models. This can cause issues when also adding regression labels."
                "Make sure your model takes the correct features.")

        self._events = pd.read_csv(self.events_path)
        self._id_encoding = json.load(open(self.id_encoding_path))
        res = self.preprocess(data)
        del self._events
        return res

    def preprocess(self, data: dict) -> dict:
        """Adds the event labels to the data.

        We add the onset and wakeup event label to the window.
        If an event is missing, we label it as -1 and set NaN onset/wakeup to 1.
        Furthermore, if there are 2 events in the same window,
        we pick the events that are successive and leave the other event out.
        This does not happen much < 0.5% of the time.

        :param data: The dataframe to add the event labels to
        :return: The dataframe with the event labels ("onset", "wakeup", "onset-NaN", "wakeup-NaN")
        """

        # Set onset and wakeup to -1
        for sid in data.keys():
            data[sid]["onset"] = np.int16(-1)
            data[sid]["wakeup"] = np.int16(-1)
            data[sid]["onset-NaN"] = np.int8(1)
            data[sid]["wakeup-NaN"] = np.int8(1)

        # apply encoding to events
        self._events['series_id'] = self._events['series_id'].map(
            self._id_encoding)

        # iterate over the series and set the awake column
        tqdm.pandas()
        for sid in tqdm(data.keys()):
            data[sid] = self.fill_series_labels(data[sid])
        return data

    def fill_series_labels(self, series: pd.DataFrame) -> pd.DataFrame:
        """
        Fill the onset/wakeup column for the series
        :param series: a series_id group
        :return: the series with the onset/wakeup and onset-NaN/wakeup-Nan columns filled
        """
        series_id = series['series_id'].iloc[0]
        current_events = self._events[self._events["series_id"] == series_id]

        # Only get non-nan values and convert to int
        current_onsets = current_events[current_events["event"]
                                        == "onset"]["step"].dropna().astype(int).values
        current_wakeups = current_events[current_events["event"]
                                         == "wakeup"]["step"].dropna().astype(int).values

        # Step at which the window starts
        window_start = series["step"].iloc[0]

        # Update the current_onsets and current_wakeups to be relative to the window
        current_onsets -= window_start
        current_wakeups -= window_start

        # For each onset, fill the respective window with onset value and set the NaN onset to 0
        for onset in current_onsets:
            window_no, window_onset = divmod(onset, data_info.window_size)
            window = series[series["window"] == window_no]
            window["onset"] = np.int16(window_onset)
            window["onset-NaN"] = np.int8(0)

            # Update the window in the series
            series[series["window"] == window_no] = window

        # Do the same for the wakeups
        for wakeup in current_wakeups:
            window_no, window_wakeup = divmod(wakeup, data_info.window_size)
            window = series[series["window"] == window_no]
            window["wakeup"] = np.int16(window_wakeup)
            window["wakeup-NaN"] = np.int8(0)

            # Update the window in the series
            series[series["window"] == window_no] = window

        return series
