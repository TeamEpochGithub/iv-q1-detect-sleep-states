# This is the base class for feature engineering
import os

import pandas as pd

from ..logger.logger import logger


class FE:
    def __init__(self, config):
        # Init function
        self.config = config

    def feature_engineering(self, data):
        # Feature engineering function
        raise NotImplementedError

    def run(self, data, fe_s, pp_s):
        # Summarize the code below
        # Check if the prev path exists
        # If it does, read from it
        # If it doesn't, calculate the current path to save the data
        # If it doesn't exist, apply the feature engineering and save the features
        data_path = 'data/features/' + '_'.join(pp_s) + '_'
        feature_path = data_path + '_'.join(fe_s) + '.parquet'
        if os.path.exists(feature_path):
            logger.info(f'--- Feature already exists, reading from {feature_path}')
            processed = pd.read_parquet(feature_path)
            logger.info('--- Data read from {feature_path}')
        else:
            logger.info('--- Features do not exist, extracting features')
            processed = self.feature_engineering(data)
            logger.info(f'--- Features have been extracted, ready to save the data to {feature_path}')
            if not isinstance(processed, pd.DataFrame):
                raise TypeError('Preprocessing step did not return a pandas DataFrame')
            else:
                processed.to_parquet(feature_path, compression='zstd')
                logger.info(f'--- Features have been saved to {feature_path}')
        return processed
