import numpy as np
import pandas as pd

from src.preprocessing.pp import PP, PPException
from ..logger.logger import logger


class AddEventLabels(PP):
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

        if "window" not in data.columns:
            logger.critical("No window column. Did you run SplitWindows before?")
            raise PPException("No window column. Did you run SplitWindows before?")
        if "awake" not in data.columns:
            logger.critical("No awake column. Did you run AddStateLabels before?")
            raise PPException("No awake column. Did you run AddStateLabels before?")

        # Find transitions from 1 to 0 (excluding 2-1 and 1-2 transitions)
        sleep_onsets = data[(data['awake'].diff() == -1) & (data['awake'].shift() == 1)]["step"].tolist()

        # Find transitions from 0 to 1 (excluding 1-2 and 2-1 transitions)
        sleep_awakes = data[(data['awake'].diff() == 1) & (data['awake'].shift() == 0)]["step"].tolist()

        # This should never happen
        if len(sleep_onsets) >= 2 and len(sleep_awakes) >= 2:
            logger.warn("--- Found 2 onsets and 2 awake transitions in 1 window... This should never happen...")
            # raise PPException("Found 2 onsets and 2 awake transitions in 1 window... This should never happen...")

        if abs(len(sleep_onsets) - len(sleep_awakes)) > 1:
            logger.warn("--- Found more than 1 missing event in 1 window... This should never happen...")
            # raise PPException("Found more than 1 missing event in 1 window... This should never happen...")

        # If we have 1/2 sleep onsets, we pick first onset
        if len(sleep_onsets) == 1 or len(sleep_onsets) == 2:
            data["onset"] = np.int16(sleep_onsets[0])
            data["onset-NaN"] = np.int8(0)

        if len(sleep_awakes) == 1:
            data["wakeup"] = np.int16(sleep_awakes[0])
            data["wakeup-NaN"] = np.int8(0)
        elif len(sleep_awakes) == 2:
            data["wakeup"] = np.int16(sleep_awakes[1])
            data["wakeup-NaN"] = np.int8(0)

        # If we have 0 sleep onsets, we set onset to -1 and onset-NaN to 1
        if len(sleep_onsets) == 0:
            data["onset"] = np.int16(-1)
            data["onset-NaN"] = np.int8(1)
        # If we have 0 sleep awakes, we set wakeup to -1 and wakeup-NaN to 1
        if len(sleep_awakes) == 0:
            data["wakeup"] = np.int16(-1)
            data["wakeup-NaN"] = np.int8(1)
            return data

        return data
