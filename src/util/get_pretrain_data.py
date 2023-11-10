import os

import numpy as np

from src.configs.load_config import ConfigLoader
from src.logger.logger import logger
from src.util.hash_config import hash_config


def get_pretrain_split_data(config_loader: ConfigLoader) \
        -> tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array] | None:
    """Get the pretrain data, either from the cache or by preprocessing the data

    :param config_loader: the config loader to use
    :param save_output: whether to save the output to the cache
    :return: the train data, test data, train labels, test labels, train indices, test indices, and groups or None if the cache was not used
    """
    config_hash: str = hash_config(config_loader.get_pretrain_config())
    path: str = config_loader.get_pp_out() + '/'

    if os.path.exists(path + config_hash):
        logger.info(f'Reading existing files at: {path}{config_hash}/')

        X_train: np.ndarray = np.load(path + config_hash + '/X_train.npy')
        X_test: np.ndarray = np.load(path + config_hash + '/X_test.npy')
        y_train: np.ndarray = np.load(path + config_hash + '/y_train.npy')
        y_test: np.ndarray = np.load(path + config_hash + '/y_test.npy')
        train_idx: np.ndarray = np.load(path + config_hash + '/train_idx.npy')
        test_idx: np.ndarray = np.load(path + config_hash + '/test_idx.npy')
        groups: np.ndarray = np.load(path + config_hash + '/groups.npy')

        logger.info('Finished reading')
        return X_train, X_test, y_train, y_test, train_idx, test_idx, groups
    else:
        logger.info(
            f"No pretrain cache found at {path}{config_hash}/. Starting with preprocessing and feature engineering.")
        return None
