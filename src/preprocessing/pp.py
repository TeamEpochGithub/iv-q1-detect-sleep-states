# Base class for preprocessing
import os
import polars as pl


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
            processed = pl.read_parquet(path)
            print(f'Data read from {path}')
            # If we want to use a pandas dataframe, convert to pandas
            if self.use_pandas:
                # convert polars dataframe back to pandas dataframe
                print('Converting polars dataframe to pandas dataframe')
                processed = processed.to_pandas()
                print('Conversion complete')
        else:
            # Recalculate the current path to save the data
            print('Preprocessed data does not exist, applying preprocessing')
            if self.use_pandas:
                if isinstance(data, pl.DataFrame):
                    print('The data is a polars dataframe, converting to pandas dataframe')
                    data = data.to_pandas()
                    print('Conversion complete')
            processed = self.preprocess(data)
            print(f'Preprocessing has been applied, ready to save the data to {path}')
            # Save the data to the path
            if isinstance(processed, pl.DataFrame):
                print(f'The data is a polars dataframe, saving to {path}')
                processed.write_parquet(path, compression='zstd')
                print(f'Preprocessed data has been saved to {path}')
                if self.use_pandas:
                    print('Converting polars dataframe to pandas dataframe')
                    processed = processed.to_pandas()
                    print('Conversion complete')
            else:
                # if we got a pandas dataframe, just save it
                print(f'The data is a pandas dataframe, saving to {path}')
                processed.to_parquet(path, compression='zstd')

        return processed
