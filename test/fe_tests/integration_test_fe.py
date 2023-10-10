'''THIS IS NOT MEANT TO BE A UNITTEST'''

import os
import time

import pandas as pd

from src.configs.load_config import ConfigLoader

if __name__ == "__main__":

    config = ConfigLoader("test/test_config.json")
    # remove the preprocessing steps from the config
    # start the timer
    start_time = time.time()
    # read the last preprocessing steps output
    # for testing purposes if the file in preprocessing doesnt exist
    # read from the raw data
    if os.path.exists(config.get_pp_in() + '_' + '_'.join(config.config['preprocessing']) + '.parquet'):
        df = pd.read_parquet(config.get_fe_in() + '_' + '_'.join(config.config['preprocessing']) + '.parquet')
    else:
        df = pd.read_parquet(config.get_pp_in() + '/train_series.parquet')
    # Initialize feature engineering steps
    fe_steps, fe_s = config.get_features()
    # Initialize the data
    featured_data = df
    # Get the feature engineering steps as a list of str to make the paths

    # Print data before fe
    print(featured_data.head())
    print(featured_data.shape)
    for i, fe_step in enumerate(fe_steps):
        # Passes the current fe list and the pp list
        feature = fe_steps[fe_step].run(df, fe_s[:i + 1], config.config["preprocessing"])
        # Add feature to featured_data
        featured_data = pd.concat([featured_data, feature], axis=1)
    # end the timer
    end_time = time.time()
    # calculate the elapsed time
    elapsed_time = end_time - start_time
    # print the elapsed time
    print(f"Elapsed time for reading 32float parquet: {elapsed_time:.6f} seconds")
    # print the data after fe
    print(featured_data.head())
    print(featured_data.shape)
