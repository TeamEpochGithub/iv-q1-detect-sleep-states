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
        return self.reduce_mem_usage(data)

    def reduce_mem_usage(self, data: pd.DataFrame, filename=None) -> pd.DataFrame:
        # TODO Don't hardcode the file name, add it as a parameter in the config.json #99
        if filename is None:
            filename = 'series_id_encoding.json'

        # we should make the series id in to an int16
        # and save an encoding (a dict) as a json file somewhere
        # so, we can decode it later
        sids = data['series_id'].unique()
        encoding = dict(zip(sids, range(len(sids))))
        # TODO Don't open the file here to make this method testable
        with open(filename, 'w') as f:
            json.dump(encoding, f)

        logger.debug(f"------ Done saving series encoding to {filename}")
        data['series_id'] = data['series_id'].map(encoding).astype('int16')

        timestamp_pl = pl.from_pandas(pd.Series(data.timestamp, copy=False))
        timestamp_pl = timestamp_pl.str.slice(0, 19)
        timestamp_pl = timestamp_pl.str.to_datetime(format="%Y-%m-%dT%H:%M:%S", time_unit='ms')
        logger.debug("------ Done converting timestamp to datetime")
        data['timestamp'] = timestamp_pl

        del timestamp_pl
        gc.collect()

        return data
