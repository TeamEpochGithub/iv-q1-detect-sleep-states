import json

import numpy as np
import pandas as pd


def to_submission_format(predictions: np.ndarray, window_info: pd.DataFrame) -> pd.DataFrame:
    """ Combine predictions with window info to create a dataframe suitable for submission.csv or scoring
    :param predictions 3D array with shape (window, 2), with onset and wakeup steps
    :return: A dataframe suitable for submission.csv or scoring, with row_id, series_id, step, event and score columns
    """

    # add the predictions with window offsets
    window_info['onset'] = predictions[:, 0] + window_info['step']
    window_info['wakeup'] = predictions[:, 1] + window_info['step']
    window_info = window_info.drop('step', axis=1)

    #Drop all nans
    window_info = window_info.dropna()

    # TODO Add confidences to the predictions and use them here
    # create a new dataframe, by converting every onset and wakeup column values to two rows,
    # one with event='onset' and the other with event='awake'
    # and then sort by series_id and window (ascending)

    df = window_info.melt(id_vars=['series_id', 'window'], value_vars=['onset', 'wakeup'], var_name='event', value_name='step').sort_values(['series_id', 'window'])

    # Drop the window column
    df = df.drop('window', axis=1)

    # add a confidence score hardcoded of 1.0 for now
    df['score'] = 1.0

    # create the row_id index
    df.reset_index(drop=True, inplace=True)
    df.index.name = 'row_id'

    # df['step'] = df['step'].astype(int)

    with open('./series_id_encoding.json', 'r') as f:
        encoding = json.load(f)
    decoding = {v: k for k, v in encoding.items()}
    df['series_id'] = df['series_id'].map(decoding)

    return df
