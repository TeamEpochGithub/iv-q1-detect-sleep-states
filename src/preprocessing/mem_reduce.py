import gc
import json
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

from ..logger.logger import logger
from ..preprocessing.pp import PP


class MemReduce(PP):
    """Preprocessing step that reduces the memory usage of the data

    It will reduce the memory usage of the data by changing the data types of the columns.
    """

    def __init__(self, id_encoding_path: Optional[str] = None, **kwargs) -> None:
        """Initialize the MemReduce class.

        :param id_encoding_path: the path to the encoding file of the series id
        """
        super().__init__(**kwargs | {"kind": "mem_reduce"})

        self.id_encoding_path: str = id_encoding_path
        self.encoding = {}

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the preprocessing step.

        :param data: the data to preprocess
        :return: the preprocessed data
        """

        return self.preprocess(data)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by reducing the memory usage of the data.

        :param data: the dataframe to preprocess
        :return: the preprocessed dataframe
        """
        return self.reduce_mem_usage(data)

    def reduce_mem_usage(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reduce the memory usage of the data.

        :param data: the dataframe to reduce
        :return: the reduced dataframe
        """
        # we should make the series id in to an int16
        # and save an encoding (a dict) as a json file somewhere
        # so, we can decode it later
        sids = data['series_id'].unique()
        encoding = dict(zip(sids, range(len(sids))))

        if self.id_encoding_path is None:
            logger.warning("No encoding path given, not saving encoding")
        else:
            logger.debug(f"------ Saving series encoding to {self.id_encoding_path}")

            # TODO Don't write the file here to make this method testable
            with open(self.id_encoding_path, 'w') as f:
                json.dump(encoding, f)

            logger.debug(f"------ Done saving series encoding to {self.id_encoding_path}")

        data['series_id'] = data['series_id'].map(encoding).astype('int16')

        timestamp_pl = pl.from_pandas(pd.Series(data.timestamp, copy=False))
        timestamp_pl = timestamp_pl.str.slice(0, 19)
        timestamp_pl = timestamp_pl.str.to_datetime(format="%Y-%m-%dT%H:%M:%S", time_unit='ms')
        logger.debug("------ Done converting timestamp to datetime")
        data['timestamp'] = timestamp_pl

        del timestamp_pl
        gc.collect()

        pad_type = {'step': np.uint32, 'series_id': np.uint16, 'enmo': np.float32,
                    'anglez': np.float32, 'timestamp': 'datetime64[ns]'}
        data = data.astype(pad_type)
        gc.collect()

        return data
