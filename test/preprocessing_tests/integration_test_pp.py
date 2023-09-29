'''THIS IS NOT MEANT TO BE A UNITTEST'''
from src.configs.load_config import ConfigLoader
import time
import polars as pl


if __name__ == "__main__":

    config = ConfigLoader("test/test_config.json")
    start_time = time.time()
    # use polars to read parquet because that is significantly faster
    df = pl.read_parquet(config.get_pp_in() + "/train_series.parquet")
    end_time = time.time()
    df = df.to_pandas()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for reading 32float parquet: {elapsed_time:.6f} seconds")
    # view the data
    print('memory usage before:')

    print(df.memory_usage(deep=True).sum()/(1024*1024))
    # Print the elapsed time

    # Initialize preprocessing steps
    pp_steps, pp_s = config.get_pp_steps()
    processed = df
    # Get the preprocessing steps as a list of str to make the paths
    for i, step in enumerate(pp_steps):
        # Passes the current list because it's needed to write to if the path doesn't exist
        processed = step.run(processed, pp_s[:i+1])

    print('memory usage after:')
    print(processed.memory_usage(deep=True).sum()/(1024*1024))
