'''THIS IS NOT MEANT TO BE A UNITTEST'''
from src.configs.load_config import ConfigLoader
import time
from src.get_processed_data import get_processed_data


if __name__ == "__main__":

    config = ConfigLoader("test/test_config.json")
    start_time = time.time()
    # Passes the current list because it's needed to write to if the path doesn't exist
    processed = get_processed_data(config, series_path='data/raw/test_series.parquet', save_output=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for preprocessing: {elapsed_time:.6f} seconds")
    print('memory usage after:')
    print(processed.memory_usage(deep=True).sum()/(1024*1024))
    print(processed.dtypes)
