import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from ..logger.logger import logger
from ..preprocessing.pp import PP, PPException


@dataclass
class AddEventLabels(PP):
    """Adds event state labels to each row of the data

    The event state labels results in two new columns: "state_onset" and "state_wakeup".
    The values are 0 for no event and 1 for event.

    :param events_path: the path to the events csv file
    :param id_encoding_path: the path to the encoding file of the series id
    :param smoothing: the sigma value for the gaussian smoothing
    :param steepness: the steepness of the gaussian smoothing
    """
    events_path: str
    id_encoding_path: str
    smoothing: int = 0
    steepness: int = 1

    _events: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False, compare=False)
    _id_encoding: dict = field(init=False, default_factory=dict, repr=False, compare=False)

    def run(self, data: dict) -> dict:
        """Run the preprocessing step.

        :param data: the data to preprocess
        :return: the preprocessed data
        :raises FileNotFoundError: If the events csv or id_encoding json file is not found
        """

        # If window column is present, raise an exception
        if "window" in data[0].columns:
            logger.critical("Window column is present, this preprocessing step should be run before SplitWindows")
            raise PPException("Window column is present, this preprocessing step should be run before SplitWindows")
        if "hot-awake" in data[0].columns:
            logger.warning(
                "Hot encoded columns are present (hot-NaN, hot-awake, hot-asleep, hot-unlabeled)"
                " for state segmentation models. This can cause issues when also adding event labels."
                "Make sure your model takes the correct features.")
        if "onset" in data[0].columns:
            logger.warning("Onset column is present, for regression models. "
                           "This can cause issues when also adding event labels."
                           "Make sure your model takes the correct features.")

        self._events = pd.read_csv(self.events_path)
        res = self.preprocess(data)
        del self._events
        return res

    def preprocess(self, data: dict) -> dict:
        """Preprocess the data by adding state labels to each row of the data.

        :param data: the data without state labels
        :return: the data with state labels added to the "awake" column
        """

        # iterate over the series and set the awake column
        for i in data.keys():
            data[i] = self.fill_series_labels(data[i], i)
        return data

    # TODO Add type hints and PyDoc comments to fill_series_labels and custom_score_array
    def fill_series_labels(self, series: pd.DataFrame, series_id: int) -> pd.DataFrame:
        series["state-onset"] = 0.0
        series["state-wakeup"] = 0.0
        current_events = self._events[self._events["series_id"] == series_id]

        # Only get non-nan values and convert to int
        current_onsets = current_events[current_events["event"] == "onset"]["step"].dropna().astype(int).values
        current_wakeups = current_events[current_events["event"] == "wakeup"]["step"].dropna().astype(int).values

        # Set the state_onset and state_wakeup columns to 1 at the event steps
        column_onset = series["state-onset"].values
        column_wakeup = series["state-wakeup"].values

        # Set the state_onset and state_wakeup columns to 1 at the event steps
        column_onset[current_onsets] = 1.0
        column_wakeup[current_wakeups] = 1.0

        # Apply the custom scoring function to the current_onsets and current_wakeups
        column_onset = self.custom_score_array(column_onset)
        column_wakeup = self.custom_score_array(column_wakeup)

        # Set the state_onset and state_wakeup columns to 1 at the event steps
        series["state-onset"] = column_onset
        series["state-wakeup"] = column_wakeup

        # Apply a gaussian label smoothing to the 1's of the state_onset and state_wakeup columns and save as float 32
        # This is done to prevent overfitting to the exact event step
        series["state-onset"] = gaussian_filter(series["state-onset"], sigma=self.smoothing).astype(np.float32)
        series["state-wakeup"] = gaussian_filter(series["state-wakeup"], sigma=self.smoothing).astype(np.float32)

        return series

    def custom_score_array(self, input_array):
        # Define the maximum distances for different scores
        tolerances = [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
        if self.steepness > 0:
            tolerances = [tolerance // self.steepness for tolerance in tolerances]
        else:
            tolerances = [tolerance * -self.steepness for tolerance in tolerances]
        scores = list(np.round(np.linspace(1, 0.1, len(tolerances)), 1))

        # Create a list of tuples
        distances_and_scores = list(zip(tolerances, scores))[::-1]
        distances_and_scores.append((0, 0))

        # Create an array to store the final scores
        result = np.zeros_like(input_array, dtype=float)

        # Find the indices of '1' in the input array using nonzero
        ones_indices = np.nonzero(input_array)[0]

        for idx in ones_indices:
            # We fill in from wide to close
            for ((curr_distance, curr_score), (next_distance, next_score)) \
                    in zip(distances_and_scores, distances_and_scores[1:]):
                # Get the bounds to fill in the scores and make sure they are not out of bounds.
                lower_bound = max(0, idx - curr_distance)
                lower_next_bound = max(0, idx - next_distance)

                upper_bound = min(len(input_array), idx + curr_distance + 1)
                upper_next_bound = min(len(input_array), idx + next_distance + 1)

                # Fill in the scores in the result array
                result[lower_bound:lower_next_bound] = curr_score
                result[upper_next_bound:upper_bound] = curr_score

            # For the first tolerance window, override the result to 1.0
            result[idx] = 1.0

        return result
