# Base class for preprocessing
import os
import pandas as pd


class PP:
    def __init__(self):
        self.use_pandas = True

    def preprocess(self, data):
        raise NotImplementedError

    def run(self, data, curr, config):
        # Check if the prev path exists
        # path = 'data/processed/' + '_'.join(curr) + '.parquet'
        # if os.path.exists(path):
        #     print(f'Preprocessed data already exists, reading from {path}')
        #     # Read the data from the path with polars
        #     if self.use_pandas:
        #         processed = pd.read_parquet(path)
        #     else:
        #         processed = pl.read_parquet(path)
        #     print(f'Data read from {path}')
        # else:
        #     # Recalculate the current path to save the data
        #     print('Preprocessed data does not exist, applying preprocessing')
        #     processed = self.preprocess(data)
        #     processed.to_parquet(path, compression='zstd')
        processed = data
        for i in range(len(curr), -1, -1):
            path = 'data/processed/' + '_'.join(curr[:i]) + '.parquet'
            # check if the final result of the preprocessing exists
            if os.path.exists(path):
                print('Found existing file at:', path)
                processed = pd.read_parquet(path)
                break
            else:
                print('File not found at:', path)
                # find the latest version of the preprocessing
                # inside this loop
                continue
        # now using i run the preprocessing steps that were not applied
        for j, step in enumerate(curr[i:]):
            path = 'data/processed/' + '_'.join(curr[:i+j+1]) + '.parquet'
            # step is the string name of the step to apply
            # use configloader to read the class
            steps, _ = config.get_pp_steps()
            step = steps[i+j]
            processed = step.preprocess(processed)
            # save the result
            print('Preprocessing was applied')
            print('Saving to:', path)
            processed.to_parquet(path)

        return processed
