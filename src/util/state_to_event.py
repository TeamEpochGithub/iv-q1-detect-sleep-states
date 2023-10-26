import numpy as np


def one_hot_to_state(one_hot: np.ndarray) -> np.ndarray:
    """Convert a one-hot encoded array to a state array,
    where the state is the index of the one-hot encoding. Normally used for predictions.
    param: one_hot numpy array, 0=sleep, 1=awake, 2=non-wear
    """
    return np.argmax(one_hot, axis=0)


def pred_to_event_state(predictions: np.ndarray, thresh: float) -> tuple:
    """Convert an event segmentation prediction to an onset and event. Normally used for predictions.
    param: 2d numpy array (labels, window_size) of event states for each timestep, 0=no state > 0=state
    param: thresh float, threshold for the prediction to be considered a state
    """
    # Set onset and awake to nan
    onset = np.nan
    awake = np.nan

    # Get max of each channel
    maxes = np.max(predictions, axis=1)

    # If onset is above threshold of making a prediction, set onset
    if maxes[0] > thresh:
        onset = np.argmax(predictions[0])

    # If awake is above threshold of making a prediction, set awake
    if maxes[1] > thresh:
        awake = np.argmax(predictions[1])

    return onset, awake


def find_events(pred: np.ndarray, median_filter_size: int = None) -> tuple:
    """Given a numpy array of integer encoded predictions for a single day window,
     find the onsets and awakes of sleep events.
     param: pred numpy array, 0=sleep, 1=awake, 2=non-wear
     """

    # Round the predictions to 0 or 1 or 2
    pred = np.round(pred)

    # Apply a median filter on pred with the median_filter parameter
    if median_filter_size is not None:
        pred = np.convolve(pred, np.ones(median_filter_size), 'same') / median_filter_size

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
