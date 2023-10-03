# Base class for preprocessing
from ..logger.logger import logger
import os

import pandas as pd
import polars as pl


class PP:
    def __init__(self):
        self.use_pandas = True

    def preprocess(self, data):
        raise NotImplementedError

    def run(self, data, curr, save_result=True):
        # Check if the prev path exists
        path = 'data/processed/' + '_'.join(curr) + '.parquet'
        if os.path.exists(path):
            logger.info(f'--- Preprocessed data already exists, reading from {path}')
            # Read the data from the path with polars
            if self.use_pandas:
                processed = pd.read_parquet(path)
            else:
                processed = pl.read_parquet(path)
            logger.info(f'--- Done reading from {path}')
        else:
            # Recalculate the current path to save the data
            logger.info('--- Cache not found, preprocessing data...')
            processed = self.preprocess(data)

            if save_result:
                processed.to_parquet(path, compression='zstd')
                logger.info(f'--- Done preprocessing, saved to {path}')
            else:
                logger.info('--- Done preprocessing, not saving result')

        return processed
