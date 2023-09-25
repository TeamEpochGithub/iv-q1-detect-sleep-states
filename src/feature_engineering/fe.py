# This is the base class for feature engineering
import os
import pandas as pd


class FE:
    def __init__(self):
        # Init function
        pass

    def fe(self, data):
        # Feature engineering function
        raise NotImplementedError

    def run(self, data, fe_s, pp_s):
        # Summarize the code below
        # Check if the prev path exists
        # If it does, read from it
        # If it doesnt, calculate the current path to save the data
        # If it doesnt exist, apply the feature engineering and save the features
        data_path = 'data/features/' + '_'.join(pp_s) + '_'
        feature_path = data_path + '_'.join(fe_s) + '.parquet'
        if os.path.exists(feature_path):
            print(f'Feature already exists, reading from {feature_path}')
            processed = pd.read_parquet(feature_path)
            print(f'Data read from {feature_path}')
        else:
            print('Features do not exist, extracting features')
            processed = self.fe(data)
            print(f'Features have been extracted, ready to save the data to {feature_path}')
            if not isinstance(processed, pd.DataFrame):
                raise TypeError('Preprocessing step did not return a pandas DataFrame')
            else:
                processed.to_parquet(feature_path, compression='zstd')
                print(f'Features have been saved to {feature_path}')
        return processed
