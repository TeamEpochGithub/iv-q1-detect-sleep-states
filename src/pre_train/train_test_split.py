import pandas as pd
import numpy as np


def train_test_split(df: pd.Dataframe, test_size: int = 0.2) -> (np.Array, np.Array, np.Array, np.Array):
    X_train = None
    X_test = None
    Y_train = None
    Y_test = None

    # Split data into train and test on series id

    exclude_x = ['timestamp', 'window', 'step', 'awake']
    keep_y_train_columns = []
    if 'awake' in df.columns:
        keep_y_train_columns.append('awake')
    x_columns = df.columns.drop(exclude_x)
    X_featured_data = df[x_columns].to_numpy().reshape(-1, 17280, len(x_columns))
    Y_featured_data = df[keep_y_train_columns].to_numpy().reshape(-1, 17280, len(keep_y_train_columns))

    return df.iloc[:-test_size], df.iloc[-test_size:]
