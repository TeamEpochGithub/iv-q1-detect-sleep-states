import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from .standardization import standardize


def train_test_split(df: pd.DataFrame, test_size: int = 0.2, standardize_method: str = "standard") -> (np.array, np.array, np.array, np.array):
    X_train = None
    X_test = None
    Y_train = None
    Y_test = None

    # Split data into train and test on series id using gss
    groups = df["series_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=groups))

    # Standardize data
    train_idx = standardize(df.iloc[train_idx], method=standardize_method)
    test_idx = standardize(df.iloc[test_idx], method=standardize_method)

    X_train, Y_train = split_on_labels(train_idx)
    X_test, Y_test = split_on_labels(test_idx)

    return X_train, X_test, Y_train, Y_test


def split_on_labels(df: pd.DataFrame) -> (np.array, np.array):

    # Only keep feature columns which start with f_
    x_data = df[['enmo', 'anglez']]

    keep_y_train_columns = []
    if 'awake' in df.columns:
        keep_y_train_columns.append('awake')
    if 'onset' in df.columns:
        keep_y_train_columns.append('onset')
    if 'wakeup' in df.columns:
        keep_y_train_columns.append('wakeup')
    if 'onset-NaN' in df.columns:
        keep_y_train_columns.append('onset-NaN')
    if 'wakeup-NaN' in df.columns:
        keep_y_train_columns.append('wakeup-NaN')
    if 'hot-asleep' in df.columns:
        keep_y_train_columns.append('hot-asleep')
    if 'hot-awake' in df.columns:
        keep_y_train_columns.append('hot-awake')
    if 'hot-NaN' in df.columns:
        keep_y_train_columns.append('hot-NaN')

    X_data = x_data.to_numpy().reshape(-1, 17280, len(x_data.columns))
    Y_data = df[keep_y_train_columns].to_numpy(
    ).reshape(-1, 17280, len(keep_y_train_columns))
    return X_data, Y_data
