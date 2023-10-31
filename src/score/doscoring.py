import numpy as np
import pandas as pd
import wandb

from src.logger.logger import logger
from src.score.scoring import score

tolerances = {
    'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
    'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
}

column_names = {
    'series_id_column_name': 'series_id',
    'time_column_name': 'step',
    'event_column_name': 'event',
    'score_column_name': 'score',
}


class ScoringError(Exception):
    pass


def compute_scores(submission: pd.DataFrame, solution: pd.DataFrame):
    # Verify that there are no consecutive onsets or wakeups
    same_event = submission['event'] == submission['event'].shift(1)
    same_series = submission['series_id'] == submission['series_id'].shift(1)
    same = submission[same_event & same_series]
    if len(same) > 0:
        logger.critical(f'Submission contains {len(same)} consecutive equal events')
        logger.critical(same)
        raise ScoringError('Submission contains consecutive equal events')

    # Count the number of labelled series in the submission and solution
    submission_sids = submission['series_id'].unique()
    solution_not_all_nan = (solution
                            .groupby('series_id')
                            .filter(lambda x: not np.isnan(x['step']).all()))

    solution_ids_not_all_nan = solution_not_all_nan['series_id'].unique()
    logger.debug(f'Submission contains predictions for {len(submission_sids)} series')
    logger.debug(f'solution has {len(solution_ids_not_all_nan)} series with at least 1 non-nan prediction)')

    # Compute the score for the entire dataset
    result = score(solution, submission, tolerances, **column_names)
    logger.info(f'Score for all {len(solution["series_id"].unique())} series: {result}')

    # Filter on clean series (series with no nans in the solution)
    solution_no_nan = (solution
                       .groupby('series_id')
                       .filter(lambda x: not np.isnan(x['step']).any()))
    solution_no_nan_ids = solution_no_nan['series_id'].unique()
    submission_filtered_no_nan = (submission
                                  .groupby('series_id')
                                  .filter(lambda x: x['series_id'].iloc[0] in solution_no_nan_ids))

    # Log the scores to WandB
    if wandb.run is not None:
        wandb.log({'score': result})

    # Compute the score for the clean series
    if len(solution_no_nan_ids) == 0 or len(submission_filtered_no_nan) == 0:
        logger.info(f'No clean series to compute non-nan score with,'
                    f' submission has none of the {len(solution_no_nan_ids)} clean series')
    else:
        result = score(solution_no_nan, submission_filtered_no_nan, tolerances, **column_names)
        if wandb.run is not None:
            wandb.log({'score_clean': result})
        logger.info(f'Score for the {len(solution_no_nan_ids)} clean series: {result}')
        return result


if __name__ == '__main__':
    import coloredlogs

    coloredlogs.install()
    submission = pd.read_csv('./submission.csv')
    solution = pd.read_csv('./data/raw/train_events.csv')
    compute_scores(submission, solution)
