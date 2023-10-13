import numpy as np


def find_events(pred: np.ndarray):
    """Given a numpy array of integer encoded predictions for a single day window,
     find the onsets and awakes of sleep events.
     param: pred numpy array, 0=sleep, 1=awake, 2=non-wear
     """
    pred = np.clip(pred, 0, 1)  # TODO: make this work for non-wear
    transition = np.diff(np.round(pred))
    onsets = np.where(transition == -1)[0]
    awakes = np.where(transition == 1)[0]

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
