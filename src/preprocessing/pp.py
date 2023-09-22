# Base class for preprocessing
import pandas as pd
import os


class PP:
    def __init__(self, config):
        self.config = config

    def preprocess(self, data):
        raise NotImplementedError

    def run(self, data, curr):
        # check if the prev path exists
        path = '../data/processed/' + '_'.join(curr[:-1]) + '.parquet'
        if os.path.exists(path):
            print(f'Preprocessed data already exists, reading from {path}')
            processed = pd.read_parquet(path)
            print(f'Data read from {path}')
        else:
            # recalculate the current path to save the data
            path = '../data/processed/' + '_'.join(curr) + '.parquet'
            print('Preprocessed data does not exist, applying preprocessing')
            processed = self.preprocess(data)
            print('Preprocessing has been applied, ready to save the data')
            if not isinstance(processed, pd.DataFrame):
                raise TypeError('Preprocessing step did not return a pandas DataFrame')
            else:
                processed.to_parquet(path)
                print(f'Preprocessed data has been saved to {path}')
        return processed
