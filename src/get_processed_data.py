import gc
import os
import tracemalloc

import pandas as pd
from tqdm import tqdm
import shutil

from src import data_info
from src.configs.load_model_config import ModelConfigLoader
from src.feature_engineering.feature_engineering import FE
from src.logger.logger import logger
from src.preprocessing.pp import PP
from src.util.hash_config import hash_config
from src.util.submissionformat import set_window_info

_STEP_HASH_LENGTH = 5


def log_memory():
    size, peak = tracemalloc.get_traced_memory()
    logger.debug(
        f"Current memory usage is {size / 10 ** 6:.2f} MB; Peak was {peak / 10 ** 6:.2f} MB")
    tracemalloc.reset_peak()


def get_processed_data(config: ModelConfigLoader, training=True, save_output=True) -> dict:
    """
    Run pp and fe steps when not cached, returns the data as a dict of dataframes.
    One df per series. Keys are series ids (not encoded)

    """

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

    # iterate backwards through steps to find the last step that is already cached
    for i in range(len(step_hashes), -1, -1):
        path = config.get_processed_out() + '/' + '_'.join(step_hashes[:i])
        # check if the final result of the preprocessing exists
        if os.path.exists(path) and i != 0:
            logger.info(f'Reading existing files at: {path}')
            for filename in tqdm(os.listdir(path)):
                sid = filename.split('.')[0]
                processed[sid] = pd.read_parquet(path + '/' + filename)
            gc.collect()
            logger.info('Finished reading')
            break
        else:
            logger.debug(f'File not found at: {path}')

    tracemalloc.start()
    log_memory()

    # if no steps were cached, read the raw data
    if i == 0:
        series_path = config.get_train_series_path(
        ) if training else config.get_test_series_path()
        logger.info(f'No files found, reading from: {series_path}')
        processed = pd.read_parquet(series_path)
        logger.info(f'Data read from: {series_path}')

    # run all remaining steps
    for j, step in enumerate(step_hashes[i:]):
        # for the raw dataframe logging memory takes so long that it is not worth it
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

            try:
                for sid in tqdm(processed.keys()):
                    processed[sid].to_parquet(path + '/' + str(sid) + '.parquet', compression='zstd')
            except KeyboardInterrupt:  # delete the folder
                logger.info('KeyboardInterrupt: saving aborted, deleting cache folder')
                shutil.rmtree(path)
                raise KeyboardInterrupt

            logger.info('--- Finished saving')
    log_memory()
    logger.debug(
        f'Total memory usage of processed dataframes: {mem_usage(processed).sum() / 1e6:.2f} MB')
    tracemalloc.stop()

    set_window_info(processed)
    return processed


def mem_usage(data: dict) -> int:
    return sum(df.memory_usage().sum() for df in data.values())
