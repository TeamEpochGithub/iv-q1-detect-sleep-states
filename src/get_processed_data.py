from src.logger.logger import logger
import os
import pandas as pd
import gc


def get_processed_data(config, series_path, save_output=True):
    pp_steps, pp_step_names = config.get_pp_steps()
    fe_steps, fe_step_names = config.get_features()
    fe_steps = [fe_steps[key] for key in fe_steps]
    step_names = pp_step_names + fe_step_names
    steps = pp_steps + fe_steps

    for i in range(len(step_names), -1, -1):
        path = config.get_pp_out() + '/' + '_'.join(step_names[:i]) + '.parquet'
        # check if the final result of the preprocessing exists
        if os.path.exists(path):
            logger.info(f'--- Found existing file at: {path}')
            logger.info(f'--- Reading from: {path}')
            processed = pd.read_parquet(path)
            logger.info(f'--- Data read from: {path}')
            break
        else:
            logger.info(f'--- File not found at: {path}')

    if i == 0:
        logger.info(f'--- No files found, reading from: {series_path}')
        processed = pd.read_parquet(series_path)
        logger.info(f'--- Data read from: {series_path}')

    # now using i run the preprocessing steps that were not applied
    for j, step in enumerate(step_names[i:]):
        path = config.get_pp_out() + '/' + '_'.join(step_names[:i+j+1]) + '.parquet'
        # step is the string name of the step to apply
        step = steps[i+j]
        logger.info(f'--- Applying step: {step_names[i+j]}')
        processed = step.run(processed)
        print(gc.collect())
        # save the result
        logger.info('--- Step was applied')
        if save_output:
            logger.info(f'--- Saving to: {path}')
            processed.to_parquet(path)
            logger.info(f'--- Saved to: {path}')
    return processed
