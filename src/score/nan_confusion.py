import pandas as pd
import wandb

from src import data_info
from src.logger.logger import logger


def compute_nan_confusion_matrix(submission: pd.DataFrame, solution: pd.DataFrame, window_info: pd.DataFrame) \
        -> (float, float, float, float):
    """Computes a confusion matrix, based on whether any prediction should be made for a window or not.

    :param submission: The submission dataframe with columns "series_id", "step", "event", and "score"
    :param solution: The solution dataframe with columns "series_id", "step", "event", and "score"
    :param window_info: The window_info dataframe with columns "series_id", "f_similarity_nan", "onset", "wakeup", "onset_confidence", and "wakeup_confidence"
    :return: The confusion matrix as a tuple of floats (true positive rate, true negative rate, false positive rate, false negative rate)
    """

    logger.info('Computing confusion matrix for making predictions or not per window')
    window_info = window_info.copy()
    window_info.set_index(['series_id', 'window'], inplace=True)
    first_offsets: pd.DataFrame = window_info.groupby(level=0).first()['step']

    submission_counts: dict[tuple[str, int], int] = {}
    solution_counts: dict[tuple[str, int], int] = {}

    for index, row in submission.dropna().iterrows():
        sid: str = row['series_id']
        window: int = (row['step'] - first_offsets[sid]) // data_info.window_size
        submission_counts[(sid, window)] = submission_counts.get((sid, window), 0) + 1

    for index, row in solution.dropna().iterrows():
        sid: str = row['series_id']
        window: int = (row['step'] - first_offsets[sid]) // data_info.window_size
        solution_counts[(sid, window)] = solution_counts.get((sid, window), 0) + 1

    total_preds: set = set(submission_counts.keys()).union(set(solution_counts.keys()))

    true_positives: int = [
        submission_counts.get((series_id, window), 0) > 0 and solution_counts.get((series_id, window), 0) > 0
        for series_id, window in total_preds].count(True)
    true_negatives: int = [
        submission_counts.get((series_id, window), 0) == 0 and solution_counts.get((series_id, window), 0) == 0
        for series_id, window in total_preds].count(True)
    false_positives: int = [
        submission_counts.get((series_id, window), 0) > 0 and solution_counts.get((series_id, window), 0) == 0
        for series_id, window in total_preds].count(True)
    false_negatives: int = [
        submission_counts.get((series_id, window), 0) == 0 and solution_counts.get((series_id, window), 0) > 0
        for series_id, window in total_preds].count(True)

    true_positive_rate: float = true_positives / len(total_preds)
    true_negative_rate: float = true_negatives / len(total_preds)
    false_positive_rate: float = false_positives / len(total_preds)
    false_negative_rate: float = false_negatives / len(total_preds)

    logger.info(f'True positives: {true_positives} ({true_positive_rate * 100:.2f}%)')
    logger.info(f'True negatives: {true_negatives} ({true_negative_rate * 100:.2f}%)')
    logger.info(f'False positives: {false_positives} ({false_positive_rate * 100:.2f}%)')
    logger.info(f'False negatives: {false_negatives} ({false_negative_rate * 100:.2f}%)')
    if wandb.run is not None:
        wandb.log({
            'TP': true_positive_rate,
            'TN': true_negative_rate,
            'FP': false_positive_rate,
            'FN': false_negative_rate,
        })

    return true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate
