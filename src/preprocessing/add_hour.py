import polars as pl
from src.preprocessing.pp import PP
import pandas as pd


class AddHour(PP):
    def __init__(self):
        pass

    def preprocess(self, data):
        # data['timestamp'] = data['timestamp'].cast(pl.Datetime)
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
        print('mark1')
        data = data.with_columns(pl.col("timestamp").str.slice(11, 8).alias("time"))
        data = data.with_columns(pl.col("time").str.to_datetime(format="%H:%M:%S").cast(pl.Datetime))
        print(data.head())
        # data.with_columns(pl.col('timestamp').str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S").cast(pl.Datetime))
        print('mark2')
        # convert polars dataframe back to pandas dataframe
        data = data.to_pandas()
        return data
