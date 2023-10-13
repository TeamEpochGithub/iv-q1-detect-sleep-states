import numpy as np


def find_events(pred: np.ndarray, median_filter_size: int = None):
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
