import json
import warnings

import numpy as np
import pandas as pd
import wandb

from src.util.submissionformat import to_submission_format
from .scoring import score
from ..logger.logger import logger

_TOLERANCES = {
    'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
    'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
}

_COLUMN_NAMES = {
    'series_id_column_name': 'series_id',
    'time_column_name': 'step',
    'event_column_name': 'event',
    'score_column_name': 'score',
}


def verify_submission(submission: pd.DataFrame) -> None:
    """Verify that there are no consecutive onsets or wakeups

    :param submission: the submission dataframe of shape (n_events, 5) with columns (series_id, step, event, onset, wakeup)
    :raises ScoringException: if there are consecutive onsets or wakeups
    """
    same_event = submission['event'] == submission['event'].shift(1)
    same_series = submission['series_id'] == submission['series_id'].shift(1)
    same = submission[same_event & same_series]
    if len(same) > 0:
        logger.critical(f'Submission contains {len(same)} consecutive equal events')
        logger.critical(same)
        raise ScoringException('Submission contains consecutive equal events')


def compute_score_full(submission: pd.DataFrame, solution: pd.DataFrame) -> float:
    """Compute the score for the entire dataset

    :param submission: the submission dataframe of shape (n_events, 5) with columns (series_id, step, event, onset, wakeup)
    :param solution: the solution dataframe of shape (n_events, 5) with columns (series_id, step, event, onset, wakeup)
    :return: the score for the entire dataset
    """
    verify_submission(submission)

    # Count the number of labelled series in the submission and solution
    submission_sids = submission['series_id'].unique()
    solution_not_all_nan = (solution
                            .groupby('series_id')
                            .filter(lambda x: not np.isnan(x['step']).all()))
    solution_ids = solution_not_all_nan['series_id'].unique()
    logger.debug(f'Submission contains predictions for {len(submission_sids)} series')
    logger.debug(f'solution has {len(solution_ids)} series with at least 1 non-nan prediction)')

    # Compute the score for the entire dataset
    score_full = score(solution, submission, _TOLERANCES, **_COLUMN_NAMES)
    logger.info(f'Score for all {len(solution["series_id"].unique())} series: {score_full}')

    return score_full


def compute_score_clean(submission: pd.DataFrame, solution: pd.DataFrame) -> float:
    """Compute the score for the clean series

    :param submission: the submission dataframe of shape (n_events, 5) with columns (series_id, step, event, onset, wakeup)
    :param solution: the solution dataframe of shape (n_events, 5) with columns (series_id, step, event, onset, wakeup)
    :return: the score for the clean series or NaN if there are no clean series
    """
    verify_submission(submission)

    # Filter on clean series (series with no nans in the solution)
    solution_no_nan = (solution
                       .groupby('series_id')
                       .filter(lambda x: not np.isnan(x['step']).any()))
    solution_no_nan_ids = solution_no_nan['series_id'].unique()
    submission_filtered_no_nan = (submission
                                  .groupby('series_id')
                                  .filter(lambda x: x['series_id'].iloc[0] in solution_no_nan_ids))

    # Compute the score for the clean series
    score_clean = np.NaN
    if len(solution_no_nan_ids) == 0 or len(submission_filtered_no_nan) == 0:
        logger.info(f'No clean series to compute non-nan score with,'
                    f' submission has none of the {len(solution_no_nan_ids)} clean series')
    else:
        score_clean = score(solution_no_nan, submission_filtered_no_nan, _TOLERANCES, **_COLUMN_NAMES)
        logger.info(f'Score for the {len(solution_no_nan_ids)} clean series: {score_clean}')

    return score_clean


def from_numpy_to_submission_format(y_true: np.ndarray, y_pred: np.ndarray, featured_data: pd.DataFrame,
                                    train_validate_idx: np.array, validate_idx: np.array,
                                    downsampling_factor: int = 1, window_size: int = 17280) -> (
        pd.DataFrame, pd.DataFrame):
    """Tries to turn the numpy y_true and y_pred into a solution and submission dataframes...

    ...but it fails.

    Yeah, this is the ugly function I was talking about.
    The resulting submission and solution aren't even the same length...

    Also, note that the input order is y_true, y_pred, whereas the output order is submission, solution.
    The order of y_true and y_pred is conventional, but the output is swapped in the output
    since the compute_score functions expect it that way and I was told not to change those.

    :param y_true: (UNUSED???) the solution numpy array of shape (X_test_cv.shape[0], window_size, n_labels) (may differ based on preprocessing and feature engineering steps)
    :param y_pred: the submission numpy array of shape (X_test_cv.shape[0], 2)
    :param featured_data: the entire dataset after preprocessing and feature engineering, but before pretraining of shape (size, n_features)
    :param train_validate_idx: the indices of the entire train and validation set of shape (featured_data[0], )
    :param validate_idx: the indices of the selected test set during the cross validation of shape (y_pred[0], )
    :param downsampling_factor: the factor by which the test data has been downsampled during the pretraining
    :return: the submission [0] and solution [1] which can be used by compute_score_full & compute_score_clean
    """
    # Get the complete train/test data
    train_test_main = featured_data.iloc[train_validate_idx]

    total_arr = []
    # Reconstruct the original indices to access the data from train_main
    for i in validate_idx:
        # TODO Use the downsampling factor here and don't hardcode the window_size
        arr = np.arange(i * 17280, (i + 1) * 17280)
        total_arr.append(arr)
    validate_idx = np.concatenate(total_arr)

    # Complete labelled data of current test split
    test_cv = train_test_main.iloc[validate_idx]

    # Prepare submission (prediction of the model)
    window_info_test_cv = (test_cv[['series_id', 'window', 'step']]
                           .groupby(['series_id', 'window'])
                           .apply(lambda x: x.iloc[0]))
    # FIXME window_info for some series starts with a very large step, instead of 0, close to the uint32 limit of 4294967295, likely due to integer underflow

    # Retrieve submission made by the model on the train split in the cv
    submission = to_submission_format(y_pred, window_info_test_cv)

    # Prepare solution
    test_series_ids = window_info_test_cv['series_id'].unique()
    # TODO Get the solution from y_true instead of loading these files
    # Load the encoding
    with open('./series_id_encoding.json', 'r') as f:
        encoding = json.load(f)
    decoding = {v: k for k, v in encoding.items()}
    test_series_ids = [decoding[sid] for sid in test_series_ids]

    # Load the train events from file
    solution_full = (pd.read_csv("data/raw/train_events.csv")
                     .groupby('series_id')
                     .filter(lambda x: x['series_id'].iloc[0] in test_series_ids)
                     .reset_index(drop=True))

    # Apply the series_id encoding
    solution_full['series_id'] = solution_full['series_id'].map(encoding).astype('int')

    # Convert dtypes, because they are fucked for no reason
    solution_full['step'] = solution_full['step'].astype(float).astype('Int32')
    # Convert step to int32 and 16
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_cv['step'] = test_cv['step'].astype('int32')
        test_cv['series_id'] = test_cv['series_id'].astype('int16')

    # Get the part from the entire train events that
    # FIXME This part here deletes NaN steps, resulting in a shorter dataframe
    solution_match = pd.merge(solution_full[['series_id', 'event', 'step']], test_cv[['series_id', 'step']],
                              on=['series_id', 'step'], how='inner')

    # Decoding series_id with the encoding object
    solution_match['series_id'] = solution_match['series_id'].map(decoding)

    # FIXME Something in here causes a warning later in src\score\scoring.py:238: RuntimeWarning: overflow encountered in scalar subtract
    return submission, solution_match


def log_scores_to_wandb(score_full: float, score_clean: float) -> None:
    """Log the scores to wandb

    :param score_full: the score for all series
    :param score_clean: the score for the clean series
    """
    if wandb.run is None:
        return

    wandb.log({'score': score_full})
    if score_clean is not np.NaN:
        wandb.log({'score_clean': score_clean})


class ScoringException(Exception):
    """Exception raised when the submission is not valid"""
    pass


if __name__ == '__main__':
    import coloredlogs

    coloredlogs.install()

    submission = pd.read_csv('./submission.csv')
    solution = pd.read_csv('./data/raw/train_events.csv')
    log_scores_to_wandb(compute_score_full(submission, solution), compute_score_clean(submission, solution))
