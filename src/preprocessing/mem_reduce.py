import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from ..logger.logger import logger
from ..preprocessing.pp import PP

PAD_TYPE: dict[str, np.dtype | str] = {'step': np.int32, 'enmo': np.float32, 'anglez': np.float32,
                                       'timestamp': 'datetime64[ns, UTC]'}


@dataclass
class MemReduce(PP):
    """Preprocessing step that reduces the memory usage of the data

    It will reduce the memory usage of the data by changing the data types of the columns.
    """

    def preprocess(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Preprocess the data by reducing the memory usage of the data.

        :param data: the dataframe with columns 'series_id', 'step', 'timestamp', 'enmo', and 'anglez'
        :return: the dataframe with the same columns using more efficient data types
        """
        return self.reduce_mem_usage(data)

    def reduce_mem_usage(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Reduce the memory usage of the data.

        :param data: the dataframe with columns 'series_id', 'step', 'timestamp', 'enmo', and 'anglez'
        :return: the dataframe with the same columns using more efficient data types
        """

        # convert series_id to int temporarily
        sids = data['series_id'].unique()
        mapping = {sid: i for i, sid in enumerate(sids)}
        data['series_id'] = data['series_id'].map(mapping)

        # Using Polars to convert timestamp to datetime because it is much faster
        timestamp_pl = pl.from_pandas(pd.Series(data.timestamp, copy=False))
        timestamp_pl = timestamp_pl.str.to_datetime(format="%Y-%m-%dT%H:%M:%S%z", time_unit='ns')
        logger.debug("------ Done converting timestamp to datetime")
        data['timestamp'] = timestamp_pl.to_pandas()

        del timestamp_pl
        gc.collect()

        data = data.astype(PAD_TYPE)
        gc.collect()

        # store each series in a different dict entry
        dfs_dict = {}
        for name, encoded in tqdm(mapping.items(), desc="Storing each series in a different dict entry"):
            dfs_dict[name] = data[data['series_id'] == encoded].drop(columns=['series_id']).reset_index(drop=True)
        del data
        gc.collect()
        return dfs_dict
