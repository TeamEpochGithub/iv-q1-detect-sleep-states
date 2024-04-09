import gc

import numpy as np
import pandas as pd

from src import data_info
from src.logger.logger import logger


def to_submission_format(predictions: np.ndarray, window_info: pd.DataFrame) -> pd.DataFrame:
    """ Combine predictions with window info to create a dataframe suitable for submission.csv or scoring
    :param predictions tuple of 2d array with shape ((window, 2), (confidence, 2)), with onset and wakeup steps and confidences for each window
    :return: A dataframe suitable for submission.csv or scoring, with row_id, series_id, step, event and score columns
    """

    step_predictions = predictions[0]
    confidences = predictions[1]
    # add the predictions with window offsets
    res = window_info.copy()

    onsets = [x[0] + y for x, y in zip(step_predictions, res['step'])]
    wakeups = [x[1] + y for x, y in zip(step_predictions, res['step'])]
    onset_conf = [x[0] for x in confidences]
    wakeup_conf = [x[1] for x in confidences]

    # Add the prediction
    res['onset'] = onsets
    res['wakeup'] = wakeups

    # Add the confidences
    res['onset_confidence'] = onset_conf
    res['wakeup_confidence'] = wakeup_conf

    # Explode all the onset, wakeup, onset_confidence and wakeup_confidence columns
    res = res.explode(["onset", "wakeup", "onset_confidence", "wakeup_confidence"], ignore_index=True)

    res = res.drop('step', axis=1)

    # Drop all nans
    res = res.dropna()

    # create a new dataframe, by converting every onset and wakeup column values to two rows,
    # one with event='onset' and the other with event='awake'
    # and then sort by series_id and window (ascending)

    df = res.melt(id_vars=['series_id', 'window'], value_vars=['onset', 'wakeup'], var_name='event',
                  value_name='step').sort_values(
        by=['series_id', 'window', 'step'])

    df_conf = res.melt(id_vars=['series_id', 'window'], value_vars=['onset_confidence', 'wakeup_confidence'],
                       var_name='event',
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
    return df


def set_window_info(data: dict) -> None:
    """Given the dictionary of processed series, set the global variable window_info.
    If set for a second time, instead of overwriting, confirm that it is the same"""

    logger.info("Setting window info")

    # get the first step of each window for each series
    res_list = []
    sids = list(data.keys())
    sids.sort()
    for sid in sids:
        res = data[sid].groupby('window').first()['step'].reset_index()
        res['series_id'] = sid
        res_list.append(res)

    res_concat = pd.concat(res_list).reset_index(drop=True)

    del res_list
    gc.collect()

    if data_info.window_info is None:
        data_info.window_info = res_concat
    else:
        data_info.window_info = res_concat
        assert data_info.window_info.equals(res_concat), "Window info is not the same as before"
