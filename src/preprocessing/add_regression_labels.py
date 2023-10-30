import numpy as np
import pandas as pd
from tqdm import tqdm

from ..logger.logger import logger
from ..preprocessing.pp import PP, PPException

import json


class AddRegressionLabels(PP):
    """Preprocessing step that adds the event labels to the data

    This will add the event labels to the data by using the event data.
    """

    def __init__(self, events_path: str, id_encoding_path: str, window_size: int = 17280, **kwargs: dict) -> None:
        """Initialize the AddRegressionLabels class
        
        :param events_path: the path to the events csv file
        :param id_encoding_path: the path to the encoding file of the series id
        :param window_size: the size of the window
        """
        super().__init__(**kwargs | {"kind": "add_regression_labels"})

        self.events_path: str = events_path
        self.events: pd.DataFrame = pd.DataFrame()
        self.id_encoding_path: str = id_encoding_path
        self.id_encoding: dict = {}
        self.window_size = window_size

    def __repr__(self) -> str:
        """Return a string representation of a AddRegressionLabels object"""
        return f"{self.__class__.__name__}(events_path={self.events_path}, id_encoding_path={self.id_encoding_path}, window_size={self.window_size})"

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the preprocessing step.
        :param data: the data to preprocess
        :return: the preprocessed data with the event labels columns ("onset", "wakeup", "onset-NaN", "wakeup-NaN")
        :raises FileNotFoundError: If the events csv or id_encoding json file is not found
        :raises PPException: If the window column is not present
        """

        # If window column is present, raise an exception
        NO_WINDOW_COLUMN_ERROR = "No window column. Did you run SplitWindows before?"
        if "window" not in data.columns:
            logger.critical(NO_WINDOW_COLUMN_ERROR)
            raise PPException(NO_WINDOW_COLUMN_ERROR)
        if "hot-asleep" in data.columns:
            logger.warning(
                "Hot encoded columns are present (hot-NaN, hot-awake, hot-asleep) for state segmentation models. This can cause issues when also adding regression labels."
                "Make sure your model takes the correct features.")
        if "state-onset" in data.columns:
            logger.warning("State-onset column is present, for state segmentation models. This can cause issues when also adding regression labels."
                           "Make sure your model takes the correct features.")

        self.events = pd.read_csv(self.events_path)
        self.id_encoding = json.load(open(self.id_encoding_path))
        res = self.preprocess(data)
        del self.events
        return res

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
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
        data["onset"] = np.int16(-1)
        data["wakeup"] = np.int16(-1)
        data["onset-NaN"] = np.int8(1)
        data["wakeup-NaN"] = np.int8(1)

        # apply encoding to events
        self.events['series_id'] = self.events['series_id'].map(
            self.id_encoding)

        # iterate over the series and set the awake column
        tqdm.pandas()
        data = (data
                .groupby('series_id')
                .progress_apply(self.fill_series_labels)
                .reset_index(drop=True))
        return data

    def fill_series_labels(self, series: pd.DataFrame) -> pd.DataFrame:
        """
        Fill the onset/wakeup column for the series
        :param series: a series_id group
        :return: the series with the onset/wakeup and onset-NaN/wakeup-Nan columns filled
        """
        series_id = series['series_id'].iloc[0]
        current_events = self.events[self.events["series_id"] == series_id]

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
            window_no = onset // self.window_size
            window = series[series["window"] == window_no]
            window["onset"] = np.int16(onset % self.window_size)
            window["onset-NaN"] = np.int8(0)

            # Update the window in the series
            series[series["window"] == window_no] = window

        # Do the same for the wakeups
        for wakeup in current_wakeups:
            window_no = wakeup // self.window_size
            window = series[series["window"] == window_no]
            window["wakeup"] = np.int16(wakeup % self.window_size)
            window["wakeup-NaN"] = np.int8(0)

            # Update the window in the series
            series[series["window"] == window_no] = window

        return series
