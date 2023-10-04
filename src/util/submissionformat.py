import pandas as pd
import json


def to_submission_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: A dataframe with series_id, onset and wakeup columns (possibly NaN)
    :return: A dataframe suitable for submission.csv or scoring, with row_id, series_id, step, event and score columns
    """

    # create a new dataframe, by converting every onset and wakeup column values to two rows,
    # one with event='onset' and the other with event='awake'
    # and then sort by series_id and onset (ascending)
    df = (df.melt(id_vars='series_id', value_vars=['onset', 'wakeup'], var_name='event', value_name='step')
          .dropna()
          .sort_values(['series_id', 'step']))

    # add a confidence score hardcoded of 1.0 for now
    df['score'] = 1.0

    # create the row_id index
    df.reset_index(drop=True, inplace=True)
    df.index.name = 'row_id'

    df['step'] = df['step'].astype(int)

    with open('./series_id_encoding.json', 'r') as f:
        encoding = json.load(f)
    decoding = {v: k for k, v in encoding.items()}
    df['series_id'] = df['series_id'].map(decoding)

    return df
