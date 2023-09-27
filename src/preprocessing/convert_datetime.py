import polars as pl
from src.preprocessing.pp import PP
import pandas as pd


class ConvertDatetime(PP):
    def __init__(self):
        self.use_pandas = False

    def preprocess(self, data):
        # Convert the timestamp column to datetime

        # Check if the class uses pandas or polars
        if not self.use_pandas:
            if isinstance(data, pd.DataFrame):
                data = pl.from_pandas(data)
        data = data.with_columns(pl.col("timestamp").str.slice(0, 18).alias("datetime"))
        data = data.with_columns(pl.col("datetime").str.to_datetime(format="%Y-%m-%dT%H:%M:%S").cast(pl.Datetime))
        # remove the timestamp column
        data = data.drop("timestamp")
        return data
