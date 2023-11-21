import numpy as np
from scipy.signal import find_peaks


def one_hot_to_state(one_hot: np.ndarray) -> np.ndarray:
    """Convert a one-hot encoded array to a state array,
    where the state is the index of the one-hot encoding. Normally used for predictions.
    param: one_hot numpy array, 0=sleep, 1=awake, 2=non-wear
    """
    return np.argmax(one_hot, axis=0)


def pred_to_event_state(predictions: np.ndarray, thresh: float, n_events: int = 1) -> tuple:
    """Convert an event segmentation prediction to an onset and event. Normally used for predictions.
    param: 3d numpy array (channel, window_size) of event states for each timestep, 0=no state > 0=state
    param: thresh float, threshold for the prediction to be considered a state
    param: n_events int, number of events to return
    """
    assert predictions.shape[1] == 2, "Predictions should be 3d array with shape (window_size, 2)"

    # Set onset and awake to nan
    onset = np.nan
    awake = np.nan
    onset_conf = np.nan
    awake_conf = np.nan

    # Get max of each channel
    maxes = np.max(predictions, axis=0)

    # If n_events is 1, return the max of each channel
    if n_events == 1:

        # If onset is above threshold of making a prediction, set onset
        if maxes[0] > thresh:
            onset = np.argmax(predictions[:, 0])
            onset_conf = maxes[0]

        # If awake is above threshold of making a prediction, set awake
        if maxes[1] > thresh:
            awake = np.argmax(predictions[:, 1])
            awake_conf = maxes[1]

        return np.array([onset]), np.array([awake]), np.array([onset_conf]), np.array([awake_conf])

    # Return every step as a prediction
    if n_events == -1:
        return np.arange(len(predictions[:, 0])), np.arange(len(predictions[:, 1])), predictions[:, 0], predictions[:, 1]

    # Find peaks in the predictions
    o_peaks, o_properties = find_peaks(predictions[:, 0], height=thresh, width=1, distance=100)

    a_peaks, a_properties = find_peaks(predictions[:, 1], height=thresh, width=1, distance=100)

    # Sort the o_peaks indices based on the peak_heights
    o_sorted = np.argsort(o_properties["peak_heights"])[::-1]
    a_sorted = np.argsort(a_properties["peak_heights"])[::-1]

    # Sort o_peaks according to the sorted indices
    o_peaks = o_peaks[o_sorted]
    a_peaks = a_peaks[a_sorted]

    # Get the top n_events
    o_peaks = o_peaks[:n_events]
    a_peaks = a_peaks[:n_events]

    # Get the confidences of the indices of o_peaks and a_peaks
    onset_conf = predictions[o_peaks, 0]
    awake_conf = predictions[a_peaks, 1]

    start_a = len(o_peaks) if len(o_peaks) > 0 else 0
    # Make sure that all after each onset there is an awake. Do this by adding o_peaks[i] + 1 to a_peaks list for every i. And insert from the beginning
    # Give this a confidence value of 0
    for i in range(len(o_peaks)):
        a_peaks = np.insert(a_peaks, i, o_peaks[i] + 1)
        awake_conf = np.insert(awake_conf, i, 0)

    # Make sure that all after each awake there is an onset. Do this by adding a_peaks[i] - 1 to o_peaks list for every i.
    for i in range(start_a, len(a_peaks)):
        o_peaks = np.append(o_peaks, a_peaks[i] - 1)
        onset_conf = np.append(onset_conf, 0)

    # Return onset and awake and the max values of those indices
    assert len(o_peaks) == len(a_peaks), "Onset and awake peaks should be the same length"
    return o_peaks, a_peaks, onset_conf, awake_conf


def find_events(pred: np.ndarray, median_filter_size: int = None) -> tuple:
    """Given a numpy array of integer encoded predictions for a single day window,
     find the onsets and awakes of sleep events.
     param: pred numpy array, 0=sleep, 1=awake, 2=non-wear
     """

    # Round the predictions to 0 or 1 or 2
    pred = np.round(pred)

    # Apply a median filter on pred with the median_filter parameter
    if median_filter_size is not None:
        pred = np.convolve(pred, np.ones(median_filter_size),
                           'same') / median_filter_size

    pred = np.round(pred)

    # Find onset indices where the awake state goes from 1 to 0
    onsets = np.where((pred[:-1] == 1) & (pred[1:] == 0))[0]
    # Find awake indices where the awake state goes from 0 to 1
    awakes = np.where((pred[:-1] == 0) & (pred[1:] == 1))[0]

    # TODO: make this work for a single onset or a single awake
    if np.size(onsets) == 0 or np.size(awakes) == 0:
        return np.nan, np.nan

    # iterate through every (valid) combination of onset and awake
    scores = np.zeros([len(onsets), len(awakes)], dtype=float)
    for o_idx, onset in enumerate(onsets):
        for a_idx, awake in enumerate(awakes):
            scores[o_idx, a_idx] = score_event_combo(pred, onset, awake)

    # return the best combination with argmax
    best_o, best_a = np.unravel_index(scores.argmax(), scores.shape)
    if np.max(scores.flat) == 0:
        return np.nan, np.nan
    return onsets[best_o], awakes[best_a]


def score_event_combo(pred, onset, awake):
    """Score an assignment of onset and awake events based on a per-timestep score"""
    if onset > awake:
        return 0

    during = pred[onset:awake]
    sleep_len = len(during)
    interruption = sum(during == 1)
    thirty_minutes = (30 * 60 / 5)

    if sleep_len < thirty_minutes:
        return 0
    if interruption > thirty_minutes:
        return 0
    return sleep_len - interruption
