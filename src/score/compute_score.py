from collections.abc import Callable

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from src.util.state_to_event import one_hot_to_state, find_events
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

    :param submission: the submission dataframe of shape (n_events, 5)
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

    :param submission: the submission dataframe of shape (n_events, 5)
    :param solution: the solution dataframe of shape (n_events, 5)
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

    :param submission: the submission dataframe of shape (n_events, 5)
    :param solution: the solution dataframe of shape (n_events, 5)
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


def one_hot_to_prediction_format(y: np.ndarray, downsampling_factor: int = 1) -> np.array:
    """ Convert a one-hot encoded array to a state array,
    where the state is the index of the one-hot encoding. Normally used for predictions.

    :param y: the one-hot encoded array of shape (n_windows, window_size, 3) with values 0=sleep, 1=awake, 2=non-wear
    :param downsampling_factor: the downsampling factor required for upsampling
    :return: an event array of shape (n_windows, 2)
    """
    if downsampling_factor > 1:
        y = np.repeat(y, downsampling_factor)

    y_res = []

    for y_window in tqdm(y, desc="Converting predictions to events", unit="window"):
        # Convert to relative window event timestamps
        y_window = one_hot_to_state(y_window)
        events = find_events(y_window, median_filter_size=15)
        y_res.append(events)

    return np.array(y_res)


def compute_score_full_from_numpy(solution: np.ndarray, submission: np.ndarray, train_idx_main: np.array,
                                  test_idx_cv: np.array, featured_data: pd.DataFrame,
                                  downsampling_factor: int = 1) -> float:
    """Compute the score for the entire dataset

    :param submission: the submission numpy array of shape (X_test.shape[0], 2)
    :param solution: the solution numpy array of shape (X_test.shape[0], window_size, n_labels)
    :return: the score for the entire dataset
    """
    # TODO Create window_info
    train_main = featured_data.iloc[train_idx_main].reset_index()

    total_arr = []
    # Reconstruct the orginal indices to access the data from train_main
    for i in test_idx_cv:
        arr = np.arange(i * 17280, (i + 1) * 17280)
        total_arr.append(arr)
    test_idx = np.concatenate(total_arr)
    test_cv = train_main.iloc[test_idx]

    # indices = train_idx[[train_idx[idx] for idx in test_idx]]

    window_info = (test_cv[['series_id', 'window', 'step']]
                   .groupby(['series_id', 'window'])
                   .apply(lambda x: x.iloc[0]))


    # Retrieve submission made by the model on the train split in the cv
    submission = to_submission_format(submission, window_info)

    # TODO Read train CSV and match on series_is and step
    # Convert to solution format of test
    solution = to_submission_format(one_hot_to_prediction_format(solution[:, :, -3:], downsampling_factor), window_info)
    return compute_score_full(submission, solution)


def make_scorer(score_func: Callable, **kwargs: dict) -> Callable:
    """Make a scorer function that can be used in cross validation

    :param score_func: the scoring function with signature score_func(y_true: np.array, y_pred: np.array, **kwargs: dict) -> float
    :param kwargs: the key word arguments for the scoring function
    :return: the scorer function with signature score_func(y_true: np.array, y_pred: np.array) -> float
    """
    return lambda y_true, y_pred, **kwargs2: score_func(y_true, y_pred, **(kwargs | kwargs2))


def log_scores_to_wandb(score_full: float, score_clean: float) -> None:
    """Log the scores to both console and wandb if logging to wandb

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
