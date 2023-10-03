# Base class for preprocessing
from ..logger.logger import logger
import os

import pandas as pd
import gc


class PP:
    def __init__(self):
        self.use_pandas = True

    def preprocess(self, data):
        raise NotImplementedError

    def run(self, data, config):
        processed = data
        steps, step_names = config.get_pp_steps()
        for i in range(len(step_names), -1, -1):
            path = config.get_pp_out() + '/' + '_'.join(step_names[:i]) + '.parquet'
            # check if the final result of the preprocessing exists
            if os.path.exists(path):
                logger.info(f'Found existing file at: {path}')
                logger.info(f'Reading from: {path}')
                processed = pd.read_parquet(path)
                logger.info(f'Data read from: {path}')
                break
            else:
                if i == 0:
                    logger.info(f'No files found, reading from: {config.get_pp_in()}')
                else:
                    logger.info(f'File not found at: {path}')
                # find the latest version of the preprocessing
                # inside this loop
                continue
        # if no files were found, i=0, read the unprocessed data here
        if i == 0:
            logger.info(f'No files found, reading from: {config.get_pp_in()}')
            processed = pd.read_parquet(config.get_pp_in() + '/train_series.parquet')
            logger.info(f'Data read from: {config.get_pp_in()}')
        # now using i run the preprocessing steps that were not applied
        for j, step in enumerate(step_names[i:]):
            path = config.get_pp_out() + '/' + '_'.join(step_names[:i+j+1]) + '.parquet'
            # step is the string name of the step to apply
            step = steps[i+j]
            logger.info(f'Applying preprocessing step: {step_names[i+j]}')
            processed = step.preprocess(processed)
            gc.collect()
            # save the result
            logger.info('Preprocessing was applied')
            logger.info(f'Saving to: {path}')
            processed.to_parquet(path)
            logger.info(f'Saved to: {path}')
        return processed
