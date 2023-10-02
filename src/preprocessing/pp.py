# Base class for preprocessing
import os
import pandas as pd


class PP:
    def __init__(self):
        self.use_pandas = True

    def preprocess(self, data):
        raise NotImplementedError

    def run(self, data, all):
        # Check if the prev path exists

        for i in range(len(all), -1, -1):
            path = 'data/processed/' + '_'.join(all[:i]) + '.parquet'
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
        for step in all[i:]:
            path = path = 'data/processed/' + '_'.join(all[:i]) + '.parquet'
            processed = step.preprocess(processed)
            # save the result
            processed.to_parquet(path)
            
        return processed
