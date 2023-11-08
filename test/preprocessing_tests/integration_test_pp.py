'''THIS IS NOT MEANT TO BE A UNITTEST'''
import json
import time

from src.configs.load_config import ConfigLoader
from src.get_processed_data import get_processed_data

if __name__ == "__main__":

    config_loader = ConfigLoader("test/test_config.json")
    start_time = time.time()
    # Passes the current list because it's needed to write to if the path doesn't exist
    processed = get_processed_data(config_loader, save_output=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for preprocessing: {elapsed_time:.6f} seconds")
    print('memory usage after:')
    print(processed.memory_usage(deep=True).sum() / (1024 * 1024))
    print(processed.dtypes)
    import pandas as pd

    events = pd.read_csv('data/raw/train_events.csv')
    f = open('series_id_encoding.json')
    encoding = json.load(f)
    events['series_id'] = events['series_id'].map(encoding)

    for id in events['series_id'].unique():
        # print for each id the non nan ids in events
        print(events.loc[events['series_id'] == id, 'step'].dropna())
        # now use pandas shift to see when the state changes
        print(processed.loc[processed['series_id'] == id].head())
        break
