import pandas as pd
import numpy as np


def submit(test_series_path, submit=False):

    test = pd.read_parquet(test_series_path)
    test["timestamp"] = pd.to_datetime(test["timestamp"], utc=True)
    test["day"] = test["timestamp"].dt.day
    test["hour"] = test["timestamp"].dt.hour
    submission = test[["series_id", 'step', 'day', 'hour']].copy()
    # Alternate event with onset and awake
    # Make a new column event and fill it with onset if hour is larger than 18 else awake if less than 12
    submission['event'] = np.where(submission['hour'] > 19, 'onset', 'awake')
    # Drop row with onset event if hour is more than 22
    submission = submission.drop(
        submission[(submission['hour'] > 22) & (submission['event'] == 'onset')].index)
    # Drop row with awake event if hour is more than 12
    submission = submission.drop(
        submission[(submission['hour'] > 9) & (submission['hour'] < 6) & (submission['event'] == 'awake')].index)

    # Keep random onset and awake event per day per series_id
    submission = submission.groupby(['series_id', 'day', 'event']).sample(1)
    submission.drop(['day', 'hour'], axis=1, inplace=True)
    submission['score'] = 1
    submission = submission.reset_index(drop=True).reset_index(names="row_id")

    if submit:
        submission.to_csv("submission.csv", index=False)
