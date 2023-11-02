import json
from collections import Counter

import numpy as np
import pandas as pd
import wandb

from src import data_info
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

_unique_series = []


def verify_cv(submission: pd.DataFrame, solution: pd.DataFrame) -> None:
    """Verify that there are no consecutive onsets or wakeups

    :param submission: the submission dataframe of shape (n_events, 5) with columns (series_id, step, event, score)
    :param solution: the solution dataframe of shape (n_events, 5) with columns (series_id, step, event)
    :raises ScoringException: if there are consecutive onsets or wakeups
    """
    # Assert that submission series_id exists in solution
    submission_series_ids = submission['series_id'].unique()
    solution_series_ids = solution['series_id'].unique()

    if not np.all([sid in solution_series_ids for sid in submission_series_ids]):
        logger.critical(f'Submission contains series ids that are not in the solution: {submission_series_ids}')
        raise ScoringException('Submission contains series ids that are not in the solution')

    # Extend unique series ids and assert that there are no duplicates
    # Log the duplicate series if they exist
    duplicates = [k for k, v in Counter(_unique_series).items() if v > 1]
    if len(duplicates) > 0:
        logger.warning(f'Duplicate series ids: {duplicates}. This means you used no groups, or there is a bug in our code. Will currently crash')

    # Assert that there are no duplicate series ids in the current submission
    if len(_unique_series) != len(set(_unique_series)):
        logger.critical('Current validation fold contains series_id, that were also in the previous fold.')
        raise ScoringException('Submission contains duplicate series ids')

    same_event = submission['event'] == submission['event'].shift(1)
    same_series = submission['series_id'] == submission['series_id'].shift(1)
    same = submission[same_event & same_series]
    if len(same) > 0:
        logger.critical(f'Submission contains {len(same)} consecutive equal events')
        logger.critical(same)
        raise ScoringException('Submission contains consecutive equal events')


def compute_score_full(submission: pd.DataFrame, solution: pd.DataFrame) -> float:
    """Compute the score for the entire dataset

    :param submission: the submission dataframe of shape (n_events, 5) with columns (series_id, step, event)
    :param solution: the solution dataframe of shape (n_events, 5) with columns (series_id, step, event, onset, wakeup)
    :return: the score for the entire dataset
    """

    # Add the series ids to the list of unique series ids
    solution_series_ids = solution['series_id'].unique()
    _unique_series.extend(solution_series_ids)

    verify_cv(submission, solution)

    # Count the number of labelled series in the submission and solution
    submission_sids = submission['series_id'].unique()
    solution_not_all_nan = (solution
                            .groupby('series_id')
                            .filter(lambda x: not np.isnan(x['step']).all()))
    solution_ids = solution_not_all_nan['series_id'].unique()
    logger.debug(f'Submission contains predictions for {len(submission_sids)} series')
    logger.debug(f'solution has {len(solution_ids)} series with at least 1 non-nan prediction)')

    # Compute the score for the entire dataset
    score_full = score(solution.dropna(), submission, _TOLERANCES, **_COLUMN_NAMES)
    logger.info(f'Score for all {len(solution["series_id"].unique())} series: {score_full}')
    logger.info(f'Number of predicted events: {len(submission)} and number of true events / no nan: {len(solution)} / {len(solution.dropna())}')

    return score_full


def compute_score_clean(submission: pd.DataFrame, solution: pd.DataFrame) -> float:
    """Compute the score for the clean series

    :param submission: the submission dataframe of shape (n_events, 5) with columns (series_id, step, event, score)
    :param solution: the solution dataframe of shape (n_events, 5) with columns (series_id, step, event)
    :return: the score for the clean series or NaN if there are no clean series
    """
    verify_cv(submission, solution)

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
        logger.info(f'Number of predicted events: {len(submission_filtered_no_nan)} and number of true events: {len(solution_no_nan)}')

    return score_clean


def from_numpy_to_submission_format(train_df: pd.DataFrame, y_pred: np.ndarray, validate_idx: np.array) -> (pd.DataFrame, pd.DataFrame):
    """Tries to turn the numpy y_true and y_pred into a solution and submission dataframes...

    ...but it fails.

    Yeah, this is the ugly function I was talking about.
    The resulting submission and solution aren't even the same length...

    Also, note that the input order is y_true, y_pred, whereas the output order is submission, solution.
    The order of y_true and y_pred is conventional, but the output is swapped in the output
    since the compute_score functions expect it that way and I was told not to change those.

    :param train_df: the X_train from the main train test split (size, n_features)
    :param y_pred: the submission numpy array of shape (X_test_cv.shape[0], 2)
    :param featured_data: the entire dataset after preprocessing and feature engineering, but before pretraining of shape (size, n_features)
    :param train_validate_idx: the indices of the entire train and validation set of shape (featured_data[0], )
    :param validate_idx: the indices of the selected test set during the cross validation of shape (y_pred[0], )
    :param downsampling_factor: the factor by which the test data has been downsampled during the pretraining
    :return: the submission [0] and solution [1] which can be used by compute_score_full & compute_score_clean
    """

    total_arr = []
    # Reconstruct the original indices to access the data from train_main
    for i in validate_idx:
        arr = np.arange(i * data_info.window_size_before, (i + 1) * data_info.window_size_before)
        total_arr.append(arr)
    data_validate_idx = np.concatenate(total_arr)

    # Complete labelled data of current test split
    test_cv = train_df.iloc[data_validate_idx]
    # Prepare submission (prediction of the model)
    window_info_test_cv = (test_cv[['series_id', 'window', 'step']]
                           .groupby(['series_id', 'window'])
                           .apply(lambda x: x.iloc[0]))

    # Retrieve submission made by the model on the train split in the cv
    submission = to_submission_format(y_pred, window_info_test_cv)

    # Prepare solution
    test_series_ids = window_info_test_cv['series_id'].unique()
    # Load the encoding
    with open('./series_id_encoding.json', 'r') as f:
        encoding = json.load(f)
    decoding = {v: k for k, v in encoding.items()}
    test_series_ids = [decoding[sid] for sid in test_series_ids]

    # Load the train events from file
    solution = (pd.read_csv("data/raw/train_events.csv")
                .groupby('series_id')
                .filter(lambda x: x['series_id'].iloc[0] in test_series_ids)
                .reset_index(drop=True))

    # Check if the test series ids from the cv are the same as the solution series ids
    assert (solution["series_id"].unique() == test_series_ids).all()

    return submission, solution


def log_scores_to_wandb(score_full: float, score_clean: float) -> None:
    """Log the scores to wandb

    :param score_full: the score for all series
    :param score_clean: the score for the clean series
    """
    if wandb.run is None:
        return

    if data_info.stage == 'cv':
        wandb.log({'cv_score': score_full})
    else:
        wandb.log({'score': score_full})

    if score_clean is not np.NaN:
        if data_info.stage == 'cv':
            wandb.log({'cv_score_clean': score_clean})
        else:
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
