# Base class for preprocessing
import os
import pandas as pd


class PP:
    def __init__(self):
        self.use_pandas = True

    def preprocess(self, data):
        raise NotImplementedError

    def run(self, data, curr):
        # Check if the prev path exists
        path = 'data/processed/' + '_'.join(curr) + '.parquet'
        if os.path.exists(path):
            print(f'Preprocessed data already exists, reading from {path}')
            # Read the data from the path with polars
            processed = pd.read_parquet(path)
            print(f'Data read from {path}')
        else:
            # Recalculate the current path to save the data
            print('Preprocessed data does not exist, applying preprocessing')
            processed = self.preprocess(data)
            processed.to_parquet(path, compression='zstd')

        return processed


class PPException(Exception):
    pass
