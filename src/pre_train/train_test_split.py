import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from .standardization import standardize


def train_test_split(df: pd.DataFrame, test_size: int = 0.2, standardize_method: str = "standard") \
        -> (np.array, np.array, np.array, np.array, np.array, np.array):

    # Split data into train and test on series id using gss
    groups = df["series_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]

    # Standardize data
    train_data = standardize(train_data, method=standardize_method)
    test_data = standardize(test_data, method=standardize_method)

    X_train, y_train = split_on_labels(train_data)
    X_test, y_test = split_on_labels(test_data)
    return X_train, X_test, y_train, y_test, train_idx, test_idx


def split_on_labels(df: pd.DataFrame) -> (np.array, np.array):
    feature_cols = [col for col in df.columns if col.startswith('f_')]
    x_data = df[['enmo', 'anglez'] + feature_cols]

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

    X_data = x_data.to_numpy(dtype='float32').reshape(-1, 17280, len(x_data.columns))
    Y_data = df[keep_y_train_columns].to_numpy(dtype='float32').reshape(-1, 17280, len(keep_y_train_columns))
    return X_data, Y_data
