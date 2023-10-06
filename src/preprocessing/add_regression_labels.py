import numpy as np
import pandas as pd
from tqdm import tqdm

from ..logger.logger import logger
from ..preprocessing.pp import PP, PPException


class AddRegressionLabels(PP):
    """Preprocessing step that adds the event labels to the data

    This will add the event labels to the data by using the event data.
    """

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds the event labels to the data.

        We add the onset and wakeup event label to the window.
        If an event is missing, we label it as -1 and set NaN onset/wakeup to 1.
        Furthermore, if there are 2 events in the same window,
        we pick the events that are successive and leave the other event out.
        This does not happen much < 0.5% of the time.

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

        # Create a hashmap to map (window, series_id) to the first index

        # Group the DataFrame by 'window' and 'series_id' and get the first index of each group
        first_indices = data.groupby(['window', 'series_id']).apply(lambda group: group.index[0])
        # Convert the resulting Series to a dictionary
        window_series_map = first_indices.to_dict()

        # Find transitions from 1 to 0 (excluding 2-1 and 1-2 transitions)
        onsets = data[(data['awake'].diff() == -1) & (data['awake'].shift() == 1)]

        # Find transitions from 0 to 1 (excluding 2-1 and 1-2 transitions)
        awakes = data[(data['awake'].diff() == 1) & (data['awake'].shift() == 0)]

        # Fill the onset and wakeup columns
        onsets.groupby([data['series_id'], data['window']]).progress_apply(fill_onset, data=data, d=window_series_map, is_onset=True)
        awakes.groupby([data['series_id'], data['window']]).progress_apply(fill_onset, data=data, d=window_series_map, is_onset=False)

        # Set the NaN onset/wakeup to 1
        return data


def fill_onset(group: pd.DataFrame, data: pd.DataFrame, d: dict, is_onset: bool) -> None:
    """
    Fill the onset/wakeup column for the group
    :param group: a series_id and window group
    :param data: the complete dataframe
    :param map: a hashmap to map (window, series_id) to the first index
    :param is_onset: boolean for if it is an onset event
    """
    series_id = group['series_id'].iloc[0]
    window = group['window'].iloc[0]
    events = group['step'].tolist()

    # Get the start
    id_start = d[(window, series_id)]

    # TODO this is hardcoded, but should be changed to a variable #99
    id_end = id_start + 17280

    if is_onset:
        if len(events) == 1 or len(events) == 2:
            # Update the 'onset' and 'onset-NaN' columns using NumPy
            data.iloc[id_start:id_end, data.columns.get_indexer(['onset', 'onset-NaN'])] = [np.int16(events[0]), np.int8(0)]
    else:
        if len(events) == 1:
            # Update the 'wakeup' and 'wakeup-NaN' columns using NumPy
            data.iloc[id_start:id_end, data.columns.get_indexer(['wakeup', 'wakeup-NaN'])] = [np.int16(events[0]), np.int8(0)]
        elif len(events) == 2:
            # Update the 'wakeup' and 'wakeup-NaN' columns using NumPy
            data.iloc[id_start:id_end, data.columns.get_indexer(['wakeup', 'wakeup-NaN'])] = [np.int16(events[1]), np.int8(0)]
    if len(events) >= 2:
        message = f"--- Found {len(events)} onsets" if is_onset else f"--- Found {len(events)} awakes"
        logger.warn(f"{message} in 1 window. This should never happen...")
        logger.debug(f"--- ERROR: {events} {'onsets' if is_onset else 'awakes'}")
