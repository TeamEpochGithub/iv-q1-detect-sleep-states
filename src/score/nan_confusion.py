import pandas as pd

window_size = (24 * 60 * 60) // 5


def compute_nan_confusion_matrix(submission: pd.DataFrame, solution: pd.DataFrame, window_info):
    """Computes a confusion matrix, based on whether any prediction should be made for a window or not."""

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
    # TODO: log to terminal (and to wandb, percentages?)