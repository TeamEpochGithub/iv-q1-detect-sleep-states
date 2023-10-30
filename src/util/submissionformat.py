import json

import numpy as np
import pandas as pd


def to_submission_format(predictions: np.ndarray, window_info: pd.DataFrame) -> pd.DataFrame:
    """ Combine predictions with window info to create a dataframe suitable for submission.csv or scoring
    :param predictions tuple of 2d array with shape ((window, 2), (confidence, 2)), with onset and wakeup steps and confidences for each window
    :return: A dataframe suitable for submission.csv or scoring, with row_id, series_id, step, event and score columns
    """

    step_predictions = predictions[0]
    confidences = predictions[1]
    # add the predictions with window offsets
    window_info['onset'] = step_predictions[:, 0] + window_info['step']
    window_info['wakeup'] = step_predictions[:, 1] + window_info['step']

    # add the confidences
    window_info['onset_confidence'] = confidences[:, 0]
    window_info['wakeup_confidence'] = confidences[:, 1]

    window_info = window_info.drop('step', axis=1)

    # Drop all nans
    window_info = window_info.dropna()

    # create a new dataframe, by converting every onset and wakeup column values to two rows,
    # one with event='onset' and the other with event='awake'
    # and then sort by series_id and window (ascending)

    df = window_info.melt(id_vars=['series_id', 'window'], value_vars=['onset', 'wakeup'], var_name='event',
                          value_name='step').sort_values(['series_id', 'window'])

    df_conf = window_info.melt(id_vars=['series_id', 'window'], value_vars=['onset_confidence', 'wakeup_confidence'], var_name='event',
                          value_name='confidence').sort_values(['series_id', 'window'])

    df['score'] = df_conf['confidence']

    # Drop the window column
    df = df.drop('window', axis=1)

    # create the row_id index
    df.reset_index(drop=True, inplace=True)
    df.index.name = 'row_id'

    # df['step'] = df['step'].astype(int)

    with open('./series_id_encoding.json', 'r') as f:
        encoding = json.load(f)
    decoding = {v: k for k, v in encoding.items()}
    df['series_id'] = df['series_id'].map(decoding)

    return df
