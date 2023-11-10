import os
import pickle

import numpy as np
import pandas as pd

from src import data_info
from src.configs.load_config import ConfigLoader
from src.logger.logger import logger
from src.pretrain.pretrain import Pretrain
from src.util.hash_config import hash_config


def get_pretrain_split_cache(config_loader: ConfigLoader, featured_data: pd.DataFrame, save_output: bool = True) \
        -> (np.array, np.array, np.array, np.array, np.array, np.array, np.array):
    """Get the pretrain data, either from the cache or by preprocessing the data

    :param config_loader: the config loader to use
    :param featured_data: the data after preprocessing and feature engineering with shape (n_samples, n_features)
    :param save_output: whether to save the output to a cache
    :return: the train data, test data, train labels, test labels, train indices, test indices, and groups or None if the cache was not saved
    """
    pretrain_config_hash: str = hash_config(config_loader.get_pretrain_config())
    path: str = config_loader.get_pp_out() + '/pretrain_' + pretrain_config_hash + '.pkl'

    if os.path.exists(path):
        logger.info(f'Reading existing files from: {path}')
        X_train, X_test, y_train, y_test, train_idx, test_idx, groups, data_info.X_columns, data_info.y_columns\
            = pickle.load(open(path, "rb"))

        logger.info('Finished reading')
    else:
        logger.info(f"No pretrain cache found at {path}.")
        logger.info("Get pretraining parameters from config and initialize pretrain")
        pretrain: Pretrain = config_loader.get_pretraining()

        logger.info(
            "Pretraining with scaler " + str(pretrain.scaler.kind) + " and test size of " + str(pretrain.test_size))

        # Split data into train/test and validation
        # Use numpy.reshape to turn the data into a 3D tensor with shape (window, n_timesteps, n_features)
        logger.info("Splitting data into train and test...")
        data_info.substage = "pretrain_split"

        X_train, X_test, y_train, y_test, train_idx, test_idx, groups = pretrain.pretrain_split(featured_data)

        if save_output:
            logger.info(f"Saving pretrain cache to {path}")
            pickle.dump((X_train, X_test, y_train, y_test, train_idx, test_idx, groups,
                         data_info.X_columns, data_info.y_columns), open(path, "wb"))

    return X_train, X_test, y_train, y_test, train_idx, test_idx, groups
