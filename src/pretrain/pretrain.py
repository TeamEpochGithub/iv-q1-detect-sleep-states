from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .. import data_info
from ..logger.logger import logger
from ..pretrain.downsampler import Downsampler
from ..scaler.scaler import Scaler


class Pretrain:
    """This class is used to prepare the data for training

    It's main functionality is to split the data into train and test sets,
    standardize the data according to the train set, split the data into features and labels,
    and convert the data to a numpy array.
    """

    def __init__(self, scaler: Scaler, downsampler: Downsampler, test_size: float):

        """Initialize the pretrain object

        :param scaler: the scaler to use
        :param downsampler: the downsampler to use
        :param test_size: the size of the test set
        """
        self.scaler = scaler
        self.downsampler = downsampler
        self.test_size = test_size

    @staticmethod
    def from_config(config: dict) -> Pretrain:
        """Create a pretrain object from the config

        :param config: the config to create the pretrain object from
        :return: the pretrain object
        """
        # Instantiate downsampler object from config
        downsampler = None
        if config.get("downsample") is not None:
            downsampler = Downsampler(**config['downsample'])

        # Instantiate scaler object from config
        scaler = Scaler(**config['scaler'])
        test_size = config["test_size"]

        return Pretrain(scaler, downsampler, test_size)

    def pretrain_split(self, data: dict) -> (
            np.array, np.array, np.array, np.array, np.array, np.array, np.array):
        """Prepare the data for training

        It splits the data into train and test sets, standardizes the data according to the train set,
        splits the data into features and labels, and converts the data to a numpy array of shape (window, window_size, n_features).

        :param df: the dataframe to pretrain on
        :return: the train data, test data, train labels, test labels, train indices, test indices, and groups
        """

        sids = list(data.keys())
        sids.sort()
        train_ids, test_ids = self.train_test_split(data, test_size=self.test_size)

        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []
        groups_list = []

        for sid in sids:
            X, y = self.split_on_labels(data[sid])

            # Apply downsampling
            if self.downsampler is not None:
                logger.info(f"Downsampling data with factor {data_info.downsampling_factor}")
                data_info.window_size_before = data_info.window_size
                data_info.window_size = data_info.window_size // data_info.downsampling_factor
                X = self.downsampler.downsampleX(X)
                y = self.downsampler.downsampleY(y)
                group = sid * np.ones(X.shape[0])

                if sid in train_ids:
                    X_train_list.append(X)
                    y_train_list.append(y)
                    groups_list.append(group)
                else:
                    X_test_list.append(X)
                    y_test_list.append(y)

        X_train = pd.concat(X_train_list)
        y_train = pd.concat(y_train_list)
        X_test = pd.concat(X_test_list)
        y_test = pd.concat(y_test_list)
        groups = np.concatenate(groups_list)

        # Store column names
        data_info.X_columns = {column: i for i, column in enumerate(X_train.columns)}
        data_info.y_columns = {column: i for i, column in enumerate(y_train.columns)}

        # Apply scaler and convert to numpy
        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_test = self.scaler.transform(X_test).astype(np.float32)
        y_train = y_train.to_numpy(dtype=np.float32)
        y_test = y_test.to_numpy(dtype=np.float32)

        X_train = self.to_windows(X_train)
        X_test = self.to_windows(X_test)
        y_train = self.to_windows(y_train)
        y_test = self.to_windows(y_test)

        return X_train, X_test, y_train, y_test, train_ids, test_ids, groups

    def pretrain_final(self, data: dict) -> (np.array, np.array, np.array):
        """Prepare the data for training

        It splits the data into train and test sets, standardizes the data according to the train set,
        splits the data into features and labels, and converts the data to a numpy array of shape (window, window_size, n_features).

        :param data: the dataframes to pretrain on
        :return: the train data, test data, train labels, test labels, train indices and test indices
        """

        X_list = []
        y_list = []
        groups_list = []

        sids = list(data.keys())
        sids.sort()
        for sid in sids:
            X_data, y_data = self.split_on_labels(data[sid])

            # Apply downsampling
            if self.downsampler is not None:
                logger.info(f"Downsampling data with factor {data_info.downsampling_factor}")
                data_info.window_size_before = data_info.window_size
                data_info.window_size = data_info.window_size // data_info.downsampling_factor
                X_data = self.downsampler.downsampleX(X_data)
                y_data = self.downsampler.downsampleY(y_data)

            X_list.append(X_data)
            y_list.append(y_data)

            groups_series = sid * np.ones(X_data.shape[0])
            groups_list.append(groups_series)

            del data[sid]
            gc.collect()

        X_data = pd.concat(X_list)
        y_data = pd.concat(y_list)
        groups = np.concatenate(groups_list)

        # Store column names
        data_info.X_columns = {column: i for i, column in enumerate(X_data.columns)}
        data_info.y_columns = {column: i for i, column in enumerate(y_data.columns)}

        # Apply scaler
        X_data = self.scaler.fit_transform(X_data).astype(np.float32)
        y_data = y_data.to_numpy(dtype=np.float32)
        gc.collect()

        X_data = self.to_windows(X_data)
        y_data = self.to_windows(y_data)


        return X_data, y_data, groups

    def preprocess(self, data: dict) -> np.array:
        """Prepare the data for submission

        The data is supposed to be processed the same way as for the training and testing data.

        :param data: the dataframes to preprocess with the series id as key
        :return: the processed data
        """

        results = []

        sids = list(data.keys())
        sids.sort()
        for sid in sids:
            x_data = self.get_features(data[sid])

            # Apply downsampling
            if self.downsampler is not None:
                logger.info(f"Downsampling data with factor {data_info.downsampling_factor}")
                data_info.window_size_before = data_info.window_size
                data_info.window_size = data_info.window_size // data_info.downsampling_factor
                x_data = self.downsampler.downsampleX(x_data)

            data_info.X_columns = {column: i for i, column in enumerate(x_data.columns)}

            x_data = self.scaler.transform(x_data).astype(np.float32)
            x_data = self.to_windows(x_data)
            results.append(x_data)

            del data[sid]
            gc.collect()

        concat = np.concatenate(results, axis=0)
        return concat

    @staticmethod
    def train_test_split(data: dict, test_size: float = 0.2) -> (pd.DataFrame, pd.DataFrame, np.array, np.array):
        """Split data into train and test on series id using GroupShuffleSplit

        :param data: the dictionary of dataframes to split
        :return: the train sids and test sids
        """

        sids = list(data.keys())
        sids.sort()
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(sids, groups=sids))
        train_series = sids[train_idx]
        test_series = sids[test_idx]

        return train_series, test_series

    @staticmethod
    def get_features(df: pd.DataFrame) -> pd.DataFrame:
        """Split the labels from the features

        :param df: the dataframe to split
        :return: the data and features
        """
        df = df.rename(columns={"enmo": "f_enmo", "anglez": "f_anglez"})
        feature_cols = [col for col in df.columns if col.startswith('f_')]

        return df[feature_cols]

    @staticmethod
    def split_on_labels(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """Split the data from the labels

        :param df: the dataframe to split
        :return: the data + features (1) and labels (2)
        """
        # Rename enmo and anglez to f_enmo and f_anglez
        df = df.rename(columns={"enmo": "f_enmo", "anglez": "f_anglez"})
        feature_cols = [col for col in df.columns if col.startswith('f_')]

        keep_columns: list[str] = ["awake", "onset", "wakeup", "onset-NaN", "wakeup-NaN",
                                   "hot-asleep", "hot-awake", "hot-NaN", "hot-unlabeled", "state-onset", "state-wakeup",
                                   "series_id"]
        keep_y_train_columns: list = [column for column in keep_columns if column in df.columns]

        return df[feature_cols], df[keep_y_train_columns]

    @staticmethod
    def to_windows(arr: np.ndarray) -> np.array:
        """Convert an array to a 3D tensor with shape (window, window_size, n_features)

        It's really just a simple reshape, but specifically for the windows.
        window_size is the number of steps in a window.

        :param arr: the array to convert, with shape (dataset length, number of columns)
        :return: the numpy array of shape (window, window_size, n_features)
        """
        return arr.reshape(-1, data_info.window_size, arr.shape[-1])
