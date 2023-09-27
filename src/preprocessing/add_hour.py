import polars as pl
from src.preprocessing.pp import PP
import pandas as pd


class AddHour(PP):
    def __init__(self):
        pass

    def preprocess(self, data):
        # convert pandas dataframe to polars dataframe to make column operations faster
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
        print(data.head())
        # do the datetime operations on the polars dataframe
        data = data.with_columns(pl.col("timestamp").str.slice(11, 8).alias("time"))
        data = data.with_columns(pl.col("time").str.to_datetime(format="%H:%M:%S").cast(pl.Datetime))
        print(data.head())
        # convert polars dataframe back to pandas dataframe
        data = data.to_pandas()
        return data
