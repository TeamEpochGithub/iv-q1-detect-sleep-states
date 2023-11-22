import gc
import os
import tracemalloc

import pandas as pd

from src import data_info
from src.configs.load_model_config import ModelConfigLoader
from src.feature_engineering.feature_engineering import FE
from src.logger.logger import logger
from src.preprocessing.pp import PP
from src.util.hash_config import hash_config

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

    i: int = 0
    processed: pd.DataFrame = pd.DataFrame()
    for i in range(len(step_hashes), -1, -1):
        path = config.get_processed_out() + '/' + '_'.join(step_hashes[:i]) + '.parquet'
        # check if the final result of the preprocessing exists
        if os.path.exists(path):
            logger.info(f'Reading existing file at: {path}')
            processed = pd.read_parquet(path)
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
        processed = pd.read_parquet(series_path)
        logger.info(f'Data read from: {series_path}')

    # now using i run the preprocessing steps that were not applied
    for j, step in enumerate(step_hashes[i:]):
        log_memory()
        logger.debug(f'Memory usage of processed dataframe: {processed.memory_usage().sum() / 1e6:.2f} MB')
        path = config.get_processed_out() + '/' + '_'.join(step_hashes[:i + j + 1]) + '.parquet'
        # step is the string name of the step to apply
        step = steps[i + j]
        logger.info(f'--- Applying step: {step_names[i + j]}')
        data_info.substage = step_hashes[i + j]

        processed = step.run(processed)
        gc.collect()

        # save the result
        logger.info('--- Step was applied')
        if save_output:
            logger.info(f'--- Saving to: {path}')
            processed.to_parquet(path)
            logger.info('--- Finished saving')
    log_memory()
    logger.debug(
        f'Memory usage of processed dataframe: {processed.memory_usage().sum() / 1e6:.2f} MB')
    tracemalloc.stop()
    return processed
