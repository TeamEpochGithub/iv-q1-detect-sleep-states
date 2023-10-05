import numpy as np
import pandas as pd

from src.preprocessing.pp import PP, PPException
from ..logger.logger import logger
from tqdm import tqdm


class AddRegressionLabels(PP):
    """Preprocessing step that adds the event labels to the data

    This will add the event labels to the data by using the event data.
    """

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds the event labels to the data. We add the onset and wakeup event label to the window. If an event is missing, we label it as -1 and set NaN onset/wakeup to 1.
        Furthermore, if there are 2 events in the same window, we pick the events that are successive and leave the other event out. This does not happen much < 0.5% of the
        time.
        :param data: The dataframe to add the event labels to
        :return: The dataframe with the event labels
        """
        tqdm.pandas()

        if "window" not in data.columns:
            logger.critical("No window column. Did you run SplitWindows before?")
            raise PPException("No window column. Did you run SplitWindows before?")
        if "awake" not in data.columns:
            logger.critical("No awake column. Did you run AddStateLabels before?")
            raise PPException("No awake column. Did you run AddStateLabels before?")

        # Set onset and wakeup to -1
        data["onset"] = np.int16(-1)
        data["wakeup"] = np.int16(-1)
        data["onset-NaN"] = np.int8(1)
        data["wakeup-NaN"] = np.int8(1)

        # Find transitions from 1 to 0 (excluding 2-1 and 1-2 transitions)
        onsets = data[(data['awake'].diff() == -1) & (data['awake'].shift() == 1)]
        awakes = data[(data['awake'].diff() == 1) & (data['awake'].shift() == 0)]

        onsets.groupby([data['series_id'], data['window']]).progress_apply(fill_onset, data=data, is_onset=True)
        awakes.groupby([data['series_id'], data['window']]).progress_apply(fill_onset, data=data, is_onset=False)

        return data


def fill_onset(group: pd.DataFrame, data: pd.DataFrame, is_onset: bool) -> None:
    """
    Fill the onset/wakeup column for the group
    :param group: a series_id and window group
    :param data: the complete dataframe
    :param is_onset: boolean for if it is an onset event
    """
    series_id = group['series_id'].iloc[0]
    window = group['window'].iloc[0]
    events = group['step'].tolist()

    # This should never happen
    if len(events) >= 2:
        if is_onset:
            logger.warn(f"--- Found {len(events)} onsets in 1 window. This should never happen...")
            logger.debug(f"--- ERROR: {events} onsets")
        else:
            logger.warn(f"--- Found {len(events)} awakes in 1 window. This should never happen...")
            logger.debug(f"--- ERROR: {events} awakes")
    # raise PPException("Found 2 onsets and 2 awake transitions in 1 window... This should never happen...")

    if is_onset:
        if len(events) == 1 or len(events) == 2:
            # Fill onset for all the rows in the window + series_id with the first onset event
            data.loc[(data['series_id'] == series_id) & (data['window'] == window), 'onset'] = np.int16(events[0])
            data.loc[(data['series_id'] == series_id) & (data['window'] == window), 'onset-NaN'] = np.int8(0)
    else:
        if len(events) == 1:
            data.loc[(data['series_id'] == series_id) & (data['window'] == window), 'wakeup'] = np.int16(events[0])
            data.loc[(data['series_id'] == series_id) & (data['window'] == window), 'wakeup-NaN'] = np.int8(0)
        elif len(events) == 2:
            data.loc[(data['series_id'] == series_id) & (data['window'] == window), 'wakeup'] = np.int16(events[1])
            data.loc[(data['series_id'] == series_id) & (data['window'] == window), 'wakeup-NaN'] = np.int8(0)
