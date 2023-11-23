import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def to_submission_format(predictions: np.ndarray, window_info: pd.DataFrame) -> pd.DataFrame:
    """ Combine predictions with window info to create a dataframe suitable for submission.csv or scoring
    :param predictions tuple of 2d array with shape ((window, 2), (confidence, 2)), with onset and wakeup steps and confidences for each window
    :return: A dataframe suitable for submission.csv or scoring, with row_id, series_id, step, event and score columns
    """

    step_predictions = predictions[0]
    confidences = predictions[1]
    # add the predictions with window offsets

    onsets = [x[0] + y for x, y in zip(step_predictions, window_info['step'])]
    wakeups = [x[1] + y for x, y in zip(step_predictions, window_info['step'])]
    onset_conf = [x[0] for x in confidences]
    wakeup_conf = [x[1] for x in confidences]

    # Add the predictions
    window_info['onset'] = onsets
    window_info['wakeup'] = wakeups

    # Add the confidences
    window_info['onset_confidence'] = onset_conf
    window_info['wakeup_confidence'] = wakeup_conf

    # Explode all the onset, wakeup, onset_confidence and wakeup_confidence columns
    window_info = window_info.explode(["onset", "wakeup", "onset_confidence", "wakeup_confidence"], ignore_index=True)

    window_info = window_info.drop('step', axis=1)

    # Drop all nans
    window_info = window_info.dropna()

    # create a new dataframe, by converting every onset and wakeup column values to two rows,
    # one with event='onset' and the other with event='awake'
    # and then sort by series_id and window (ascending)

    df = window_info.melt(id_vars=['series_id', 'window'], value_vars=['onset', 'wakeup'], var_name='event', value_name='step').sort_values(
        by=['series_id', 'window', 'step'])

    df_conf = window_info.melt(id_vars=['series_id', 'window'], value_vars=['onset_confidence', 'wakeup_confidence'], var_name='event',
                               value_name='confidence').sort_values(by=['series_id', 'window', 'confidence'])

    df['score'] = df_conf['confidence']

    # Drop the window column
    df = df.drop('window', axis=1)

    # Remove duplicate events
    df = df.drop(df[df['event'].eq(df['event'].shift())].index)

    # create the row_id index
    df.reset_index(drop=True, inplace=True)
    df.index.name = 'row_id'

    df['step'] = df['step'].astype(float)
    df["score"] = df["score"].astype(float)

    # Drop all rows that have a step value < 0
    df = df[df['step'] >= 0]

    with open('./series_id_encoding.json', 'r') as f:
        encoding = json.load(f)
    decoding = {v: k for k, v in encoding.items()}
    df['series_id'] = df['series_id'].map(decoding)

    return df
