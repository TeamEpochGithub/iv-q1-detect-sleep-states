# This class is to reduce memory usage of dataframe
from ..preprocessing.pp import PP
import json
import pandas as pd
import polars as pl


class MemReduce(PP):

    def preprocess(self, data):
        df = self.reduce_mem_usage(data)
        return df

    def reduce_mem_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        # we should make the series id in to an int16
        # and save an encoding (a dict) as a json file somewhere
        # so we can decode it later
        encoding = dict(zip(data['series_id'].unique(), range(len(data['series_id'].unique()))))
        with open('series_id_encoding.json', 'w') as f:
            json.dump(encoding, f)

        data['series_id'] = data['series_id'].map(encoding)
        data['series_id'] = data['series_id'].astype('int16')

        # putting this after the int16 conversion makes it reduce memory
        # a lot more than if we put it before
        if not self.use_pandas:
            if isinstance(data, pd.DataFrame):
                data = pl.from_pandas(data)
        data = pl.from_pandas(data)
        data = data.with_columns(pl.col("timestamp").str.slice(0, 18))
        data = data.with_columns(pl.col("timestamp").str.to_datetime(format="%Y-%m-%dT%H:%M:%S").cast(pl.Datetime))
        # remove the timestamp column
        data = data.to_pandas()
        return data
