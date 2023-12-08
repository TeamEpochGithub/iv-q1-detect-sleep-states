import numpy as np
import pandas as pd

TOLERANCES: list[int] = [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]


def score_fast(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """Compute the score like in event_detection_ap.py but using the faster computation.

    This is an attempt at making the interface of the score function the same as in the original event_detection_ap.py
    file. This is done so that we can use the same code for computing the score.

    Most of the code below was copied from https://www.kaggle.com/code/chauyh/kagglechildsleep-fast-ap-metric-computation/notebook,
    so don't judge if it's not up to our standards.

    :param solution: the solution dataframe of shape (n_events, 5) with columns (series_id, step, event, onset, wakeup)
    :param submission: the submission dataframe of shape (n_events, 5) with columns (series_id, step, event, onset, wakeup)
    :return: the score (but faster!)
    """
    all_series_ids = list(submission["series_id"].unique())
    all_series_ids = np.array(all_series_ids, dtype="object")

    events = solution
    per_seriesid_events = {}
    for series_id in all_series_ids:
        per_seriesid_events[series_id] = {
            "onset": [], "wakeup": []
        }
        series_events = events.loc[events["series_id"] == series_id]
        onsets = series_events.loc[series_events["event"] == "onset"]["step"]
        wakeups = series_events.loc[series_events["event"] == "wakeup"]["step"]
        if len(onsets) > 0:
            per_seriesid_events[series_id]["onset"].extend(onsets.to_numpy(np.int32))
        if len(wakeups) > 0:
            per_seriesid_events[series_id]["wakeup"].extend(wakeups.to_numpy(np.int32))

    gt_events = per_seriesid_events

    predicted_events = {}
    for series_id in submission["series_id"].unique():
        series_submission = submission.loc[submission["series_id"] == series_id]
        series_onset = series_submission.loc[series_submission["event"] == "onset"]
        series_wakeup = series_submission.loc[series_submission["event"] == "wakeup"]
        predicted_events[series_id] = {
            "onset": series_onset["step"].to_numpy(dtype=np.int32),
            "onset_proba": series_onset["score"].to_numpy(dtype=np.float32),
            "wakeup": series_wakeup["step"].to_numpy(dtype=np.int32),
            "wakeup_proba": series_wakeup["score"].to_numpy(dtype=np.float32)
        }

    ap_onset_metrics = [EventMetrics(name="", tolerance=tolerance) for tolerance in TOLERANCES]
    ap_wakeup_metrics = [EventMetrics(name="", tolerance=tolerance) for tolerance in TOLERANCES]

    for series_id in all_series_ids:
        # get the ground truth
        gt_onset_locs = gt_events[series_id]["onset"]
        gt_wakeup_locs = gt_events[series_id]["wakeup"]

        # get the predictions
        preds_onset = predicted_events[series_id]["onset"]
        preds_wakeup = predicted_events[series_id]["wakeup"]
        onset_IOU_probas = predicted_events[series_id]["onset_proba"]
        wakeup_IOU_probas = predicted_events[series_id]["wakeup_proba"]

        # add info
        for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
            ap_onset_metric.add(pred_locs=preds_onset, pred_probas=onset_IOU_probas, gt_locs=gt_onset_locs)
            ap_wakeup_metric.add(pred_locs=preds_wakeup, pred_probas=wakeup_IOU_probas, gt_locs=gt_wakeup_locs)

    # compute average precision
    ap_onset_precisions, ap_onset_recalls, ap_onset_average_precisions, ap_onset_probas = [], [], [], []
    ap_wakeup_precisions, ap_wakeup_recalls, ap_wakeup_average_precisions, ap_wakeup_probas = [], [], [], []
    for ap_onset_metric, ap_wakeup_metric in zip(ap_onset_metrics, ap_wakeup_metrics):
        ap_onset_precision, ap_onset_recall, ap_onset_average_precision, ap_onset_proba = ap_onset_metric.get()
        ap_wakeup_precision, ap_wakeup_recall, ap_wakeup_average_precision, ap_wakeup_proba = ap_wakeup_metric.get()
        ap_onset_precisions.append(ap_onset_precision)
        ap_onset_recalls.append(ap_onset_recall)
        ap_onset_average_precisions.append(ap_onset_average_precision)
        ap_onset_probas.append(ap_onset_proba)
        ap_wakeup_precisions.append(ap_wakeup_precision)
        ap_wakeup_recalls.append(ap_wakeup_recall)
        ap_wakeup_average_precisions.append(ap_wakeup_average_precision)
        ap_wakeup_probas.append(ap_wakeup_proba)

    return np.mean((np.mean(ap_onset_average_precisions), np.mean(ap_wakeup_average_precisions)))


def match_series(pred_locs: np.ndarray, pred_probas: np.ndarray, gt_locs: list | np.ndarray, tolerance=12 * 30):
    """
    Probably faster algorithm for matching, since the gt are disjoint (within tolerance)

    pred_locs: predicted locations of events, assume sorted in ascending order
    pred_probas: predicted probabilities of events
    gt_locs: ground truth locations of events (either list[int] or np.ndarray or int32 type)
    """
    assert pred_locs.shape == pred_probas.shape, "pred_locs {} and pred_probas {} must have the same shape".format(
        pred_locs.shape, pred_probas.shape)
    assert len(pred_locs.shape) == 1, "pred_locs {} and pred_probas {} must be 1D".format(pred_locs.shape,
                                                                                          pred_probas.shape)
    matches = np.zeros_like(pred_locs, dtype=bool)

    if isinstance(gt_locs, list):
        gt_locs = np.array(gt_locs, dtype=np.int32)
    else:
        assert isinstance(gt_locs, np.ndarray), "gt_locs must be list or np.ndarray"
        assert gt_locs.dtype == np.int32, "gt_locs must be int32 type"

    # lie within (event_loc - tolerance, event_loc + tolerance), where event_loc in gt_locs
    idx_lows = np.searchsorted(pred_locs, gt_locs - tolerance + 1)
    idx_highs = np.searchsorted(pred_locs, gt_locs + tolerance)
    for k in range(len(gt_locs)):
        idx_low, idx_high = idx_lows[k], idx_highs[k]
        if idx_low == idx_high:
            continue
        # find argmax within range
        max_location = idx_low + np.argmax(pred_probas[idx_low:idx_high])
        matches[max_location] = True
    return matches


class EventMetrics:
    def __init__(self, name: str, tolerance=12 * 30):
        self.name = name
        self.tolerance = tolerance

        self.matches = []
        self.probas = []
        self.num_positive = 0

    def add(self, pred_locs: np.ndarray, pred_probas: np.ndarray, gt_locs):
        matches = match_series(pred_locs, pred_probas, gt_locs, tolerance=self.tolerance)
        self.matches.append(matches)
        self.probas.append(pred_probas)
        self.num_positive += len(gt_locs)

    def get(self):
        matches = np.concatenate(self.matches, axis=0)
        probas = np.concatenate(self.probas, axis=0)

        # sort by probas in descending order
        idxs = np.argsort(probas)[::-1]
        matches = matches[idxs]
        probas = probas[idxs]

        # compute precision and recall curve (using Kaggle code)
        distinct_value_indices = np.where(np.diff(probas))[0]
        threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
        probas = probas[threshold_idxs]

        # Matches become TPs and non-matches FPs as confidence threshold decreases
        tps = np.cumsum(matches)[threshold_idxs]
        fps = np.cumsum(~matches)[threshold_idxs]

        precision = tps / (tps + fps)
        precision[np.isnan(precision)] = 0
        recall = tps / self.num_positive  # total number of ground truths might be different than total number of matches

        # Stop when full recall attained and reverse the outputs so recall is non-increasing.
        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)

        # Final precision is 1 and final recall is 0 and final proba is 1
        precision, recall, probas = np.r_[precision[sl], 1], np.r_[recall[sl], 0], np.r_[probas[sl], 1]

        # compute average precision
        average_precision = -np.sum(np.diff(recall) * np.array(precision)[:-1])

        return precision, recall, average_precision, probas

    def write_to_dict(self, x: dict):
        x[self.name] = self.get()

    def reset(self):
        self.matches.clear()
        self.probas.clear()
        self.num_positive = 0
