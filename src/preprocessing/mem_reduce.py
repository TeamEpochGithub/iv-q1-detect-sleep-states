import gc
import json

import numpy as np
import pandas as pd
import polars as pl

from ..logger.logger import logger
from ..preprocessing.pp import PP


class MemReduce(PP):
    """Preprocessing step that reduces the memory usage of the data

    It will reduce the memory usage of the data by changing the data types of the columns.
    """

    def preprocess(self, data):
        df = self.reduce_mem_usage(data)
        return df

    def reduce_mem_usage(self, data: pd.DataFrame, filename=None) -> pd.DataFrame:
        # TODO Don't hardcode the file name, add it as a parameter in the config.json #99
        if filename is None:
            filename = 'series_id_encoding.json'

        # we should make the series id in to an int16
        # and save an encoding (a dict) as a json file somewhere
        # so, we can decode it later
        encoding = dict(zip(data['series_id'].unique(), range(len(data['series_id'].unique()))))
        # TODO Don't open the file here to make this method testable
        with open(filename, 'w') as f:
            json.dump(encoding, f)
        logger.debug(f"------ Done saving series encoding to {filename}")
        data['series_id'] = data['series_id'].map(encoding)
        data['series_id'] = data['series_id'].astype('int16')

        data = pl.from_pandas(data)
        gc.collect()
        data = data.with_columns(pl.col("timestamp").str.slice(0, 19))
        data = data.with_columns(pl.col("timestamp").str.to_datetime(format="%Y-%m-%dT%H:%M:%S").cast(pl.Datetime))
        logger.debug("------ Done converting timestamp to datetime")
        data = data.to_pandas()
        gc.collect()
        pad_type = {'step': np.uint32, 'series_id': np.uint16, 'enmo': np.float32,
                    'anglez': np.float32, 'timestamp': 'datetime64[ns]'}
        data = data.astype(pad_type)
        return data
