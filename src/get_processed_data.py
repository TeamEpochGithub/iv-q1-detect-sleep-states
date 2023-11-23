import gc
import os
import tracemalloc

import pandas as pd
import numpy as np
import json

from src import data_info
from src.configs.load_model_config import ModelConfigLoader
from src.feature_engineering.feature_engineering import FE
from src.logger.logger import logger
from src.preprocessing.pp import PP
from src.util.hash_config import hash_config
from tqdm import tqdm


_STEP_HASH_LENGTH = 5


def log_memory():
    size, peak = tracemalloc.get_traced_memory()
    logger.debug(
        f"Current memory usage is {size / 10 ** 6:.2f} MB; Peak was {peak / 10 ** 6:.2f} MB")
    tracemalloc.reset_peak()


def get_processed_data(config: ModelConfigLoader, training=True, save_output=True) -> pd.DataFrame:
    # Get the preprocessing and feature engineering steps
    pp_steps: list[PP] = config.get_pp_steps(training=training)
    fe_steps = config.get_fe_steps()

    steps: list[PP | FE] = pp_steps + fe_steps
    step_names: list[str] = [step.__class__.__name__ for step in steps]
    step_hashes: list[str] = [hash_config(step, _STEP_HASH_LENGTH) for step in
                              steps]  # I think it looks a little silly to use hash_config here...

    logger.info(f'--- Running preprocessing & feature engineering steps: {steps}')

    assert config.config['preprocessing'][0]['kind'] == 'mem_reduce', 'The first preprocessing step must be mem_reduce'

    i: int = 0
    processed: dict = {}
    for i in range(len(step_hashes), -1, -1):
        path = config.get_processed_out() + '/' + '_'.join(step_hashes[:i])
        # check if the final result of the preprocessing exists
        if os.path.exists(path) and i != 0:
            logger.info(f'Reading existing files at: {path}')
            # The test data will have different ids so we need to read the encoding
            assert os.path.exists(config.config['preprocessing'][0]['id_encoding_path']), 'The id encoding file does not exist, run mem_reduce again'
            ids = json.load(open(config.config['preprocessing'][0]['id_encoding_path']))
            # read the files within the folder in to a dict of dfs
            for k in ids.values():
                filename = path + '/' + str(k) + '.parquet'
                processed[k] = pd.read_parquet(filename)
            gc.collect()
            logger.info('Finished reading')
            break
        else:
            logger.debug(f'File not found at: {path}')

    tracemalloc.start()
    log_memory()
    if i == 0:
        series_path = config.get_train_series_path(
        ) if training else config.get_test_series_path()
        logger.info(f'No files found, reading from: {series_path}')
        # read the raw data
        processed = pd.read_parquet(series_path)
        logger.info(f'Data read from: {series_path}')

    # now using i run the preprocessing steps that were not applied
    for j, step in enumerate(step_hashes[i:]):
        # for the raw dataframe logging memeory takes so long that it is not worth it
        if isinstance(processed, dict):
            logger.debug(f'Memory usage of processed data: {mem_usage(processed) / 1e6:.2f} MB')
        log_memory()
        path = config.get_processed_out() + '/' + '_'.join(step_hashes[:i + j + 1])
        # step is the string name of the step to apply
        step = steps[i + j]
        logger.info(f'--- Applying step: {step_names[i + j]}')
        data_info.substage = step_hashes[i + j]
        # only mem reduce uses the datfarme input and the rest will use dicts so this
        # should be fine as long as mem_reduce is used first
        processed = step.run(processed)
        gc.collect()

        # save the result
        logger.info('--- Step was applied')
        if save_output:
            logger.info(f'--- Saving to: {path}')
            if not os.path.exists(path):
                os.makedirs(path)
            for sid in tqdm(processed.keys()):
                processed[sid].to_parquet(path + '/' + str(sid) + '.parquet')

            logger.info('--- Finished saving')
    log_memory()
    logger.debug(
        f'Memory usage of processed dataframe: {mem_usage(processed).sum() / 1e6:.2f} MB')
    tracemalloc.stop()
    # once processing is done we need to return a single dataframe
    logger.info('--- Combining dataframes')
    # add back the removd series_id column
    for sid in processed.keys():
        processed[sid]['series_id'] = sid
        processed[sid]['series_id'].astype(np.uint16)
    # combine the dataframes in the dict in to a single df
    # while adding back the series id column to them
    processed_df = pd.concat(processed.values(), ignore_index=True)
    del processed
    gc.collect()
    logger.info('--- Finished combining dataframes')
    return processed_df


def mem_usage(data: dict | pd.DataFrame) -> int:
    return sum(df.memory_usage().sum() for df in data.values())
