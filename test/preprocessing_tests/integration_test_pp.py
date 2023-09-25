'''THIS IS NOT MEANT TO BE A UNITTEST'''
from src.configs.load_config import ConfigLoader
import pandas as pd
import time

if __name__ == "__main__":

    config = ConfigLoader("src/configs/config.json")
    config.config["preprocessing"] = []
    start_time = time.time()
    df = pd.read_parquet(config.get_pp_in() + "/train_series.parquet")
    # Print the elapsed time

    # Initialize preprocessing steps
    pp_steps, pp_s = config.get_pp_steps()
    processed = df
    # get the preprocessing steps as a list of str to make the paths
    for i, step in enumerate(pp_steps):
        # passes the current list because its needed to write to if the path doesnt exist
        processed = step.run(processed, pp_s[:i+1])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for reading 32float parquet: {elapsed_time:.6f} seconds")
