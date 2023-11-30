from collections import Counter

import numpy as np
import pandas as pd
import wandb

from src import data_info
from src.util.submissionformat import to_submission_format
from .scoring import score
from ..logger.logger import logger

_TOLERANCES: dict[str, list[int]] = {
    'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
    'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
}

_COLUMN_NAMES: dict[str, str] = {
    'series_id_column_name': 'series_id',
    'time_column_name': 'step',
    'event_column_name': 'event',
    'score_column_name': 'score',
}


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
    duplicates = [k for k, v in Counter(data_info.cv_unique_series).items() if v > 1]
    if len(duplicates) > 0:
        logger.warning(f'Duplicate series ids: {duplicates}. This means you used no groups, or there is a bug in our code. Will currently crash')

    # Assert that there are no duplicate series ids in the current submission
    if len(data_info.cv_unique_series) != len(set(data_info.cv_unique_series)):
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

    if data_info.stage == 'cv':
        # Add the series ids to the list of unique series ids
        solution_series_ids = solution['series_id'].unique()
        data_info.cv_unique_series.extend(solution_series_ids)
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
    score_clean = 0
    if len(solution_no_nan_ids) == 0 or len(submission_filtered_no_nan) == 0:
        logger.info(f'No clean series to compute non-nan score with,'
                    f' submission has none of the {len(solution_no_nan_ids)} clean series')
    else:
        score_clean = score(solution_no_nan, submission_filtered_no_nan, _TOLERANCES, **_COLUMN_NAMES)
        logger.info(f'Score for the {len(solution_no_nan_ids)} clean series: {score_clean}')
        logger.info(f'Number of predicted events: {len(submission_filtered_no_nan)} and number of true events: {len(solution_no_nan)}')

    return score_clean


def from_numpy_to_submission_format(validate_window_info: pd.DataFrame, y_pred: np.ndarray,) -> (pd.DataFrame, pd.DataFrame):
    """Turn the numpy y_pred and the train events file into a solution and submission dataframes.

    While you probably want to compare y_pred with y_true, it seems that that is not possible in our case.
    Therefore, have to load the solution file from disk very time again...

    Also, note that the input order is y_true, y_pred, whereas the output order is submission, solution.
    The order of y_true and y_pred is conventional, but the output is swapped in the output
    since the compute_score functions expect it that way and I was told not to change those.

    :param train_df: the X_train from the main train test split (size, n_features)
    :param y_pred: the submission numpy array of shape (X_test_cv.shape[0], 2)
    :param validate_idx: the indices of the selected test set during the cross validation of shape (y_pred[0], )
    :return: the submission [0] and solution [1] which can be used by compute_score_full & compute_score_clean
    """

    # Combine predictions with window info to generate submission dataframe with series ids and proper offsets
    submission = to_submission_format(y_pred, validate_window_info)

    # Prepare solution
    test_series_ids = validate_window_info['series_id'].unique()

    # Load the train events from file
    solution = (pd.read_csv("data/raw/train_events.csv")
                .groupby('series_id')
                .filter(lambda x: x['series_id'].iloc[0] in test_series_ids)
                .reset_index(drop=True))

    # Check if the test series ids from the cv are the same as the solution series ids
    assert (solution["series_id"].unique() == test_series_ids).all()

    return submission, solution


def log_scores_to_wandb(scores: np.ndarray | list[float], scorer: list) -> None:
    """Log the scores to wandb
    :param scores: the mean scores
    :param scorer: the scorers
    """
    assert len(scores) == len(scorer)

    if wandb.run is None:
        return
    prepend = "cv_" if data_info.stage == 'cv' else ""

    for s, t in zip(scores, scorer):
        match t:
            case "score_full":
                wandb.log({prepend + "score": s})
            case "score_clean":
                wandb.log({prepend + "score_clean": s})


class ScoringException(Exception):
    """Exception raised when the submission is not valid"""
    pass


if __name__ == '__main__':
    import coloredlogs

    coloredlogs.install()

    submission = pd.read_csv('./submission.csv')
    solution = pd.read_csv('./data/raw/train_events.csv')
    log_scores_to_wandb(compute_score_full(submission, solution), compute_score_clean(submission, solution))
