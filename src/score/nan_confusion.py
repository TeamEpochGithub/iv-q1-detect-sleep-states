import pandas as pd
import wandb

from src.logger.logger import logger

window_size = (24 * 60 * 60) // 5


def compute_nan_confusion_matrix(submission: pd.DataFrame, solution: pd.DataFrame, window_info):
    """Computes a confusion matrix, based on whether any prediction should be made for a window or not."""

    logger.info('Computing confusion matrix for making predictions or not per window')
    window_info.set_index(['series_id', 'window'], inplace=True)
    first_offsets = window_info.groupby(level=0).first()['step']

    window_info['submissions'] = 0
    window_info['solutions'] = 0

    for index, row in submission.dropna().iterrows():
        sid = row['series_id']
        window = (row['step'] - first_offsets[sid]) // window_size
        window_info.loc[(sid, window), 'submissions'] += 1

    for index, row in solution.dropna().iterrows():
        sid = row['series_id']
        window = (row['step'] - first_offsets[sid]) // window_size
        window_info.loc[(sid, window), 'solutions'] += 1

    true_positives = sum((window_info['submissions'] > 0) & (window_info['solutions'] > 0))
    true_negatives = sum((window_info['submissions'] == 0) & (window_info['solutions'] == 0))
    false_positives = sum((window_info['submissions'] > 0) & (window_info['solutions'] == 0))
    false_negatives = sum((window_info['submissions'] == 0) & (window_info['solutions'] > 0))

    logger.info(f'True positives: {true_positives} ({true_positives / len(window_info) *100:.2f}%)')
    logger.info(f'True negatives: {true_negatives} ({true_negatives / len(window_info) *100:.2f}%)')
    logger.info(f'False positives: {false_positives} ({false_positives / len(window_info) *100:.2f}%)')
    logger.info(f'False negatives: {false_negatives} ({false_negatives / len(window_info) *100:.2f}%)')
    if wandb.run is not None:
        wandb.log({
            'TP': true_positives / len(window_info),
            'TN': true_negatives / len(window_info),
            'FP': false_positives / len(window_info),
            'FN': false_negatives / len(window_info),
        })
