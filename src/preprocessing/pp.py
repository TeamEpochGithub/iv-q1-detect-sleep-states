# Base class for preprocessing
import os
import polars as pl


class PP:
    def __init__(self):
        pass

    def preprocess(self, data):
        raise NotImplementedError

    def run(self, data, curr):
        # Check if the prev path exists
        path = 'data/processed/' + '_'.join(curr) + '.parquet'
        if os.path.exists(path):
            print(f'Preprocessed data already exists, reading from {path}')
            # Read the data from the path with polars
            processed = pl.read_parquet(path)
            print(f'Data read from {path}')
            # convert polars dataframe back to pandas dataframe
            processed = processed.to_pandas()
        else:
            # Recalculate the current path to save the data
            print('Preprocessed data does not exist, applying preprocessing')
            processed = self.preprocess(data)
            print(f'Preprocessing has been applied, ready to save the data to {path}')
            if isinstance(processed, pl.DataFrame):
                print(f'The data is a polars dataframe, saving to {path}')
                processed.write_parquet(path, compression='zstd')
                print(f'Preprocessed data has been saved to {path}')
                print('Converting polars dataframe to pandas dataframe')
                processed = processed.to_pandas()
            else:
                # if we got a pandas dataframe, just save it
                print(f'The data is a pandas dataframe, saving to {path}')
                processed.to_parquet(path, compression='zstd')

        return processed
