import gc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from ..logger.logger import logger
from ..preprocessing.pp import PP


@dataclass
class MemReduce(PP):
    """Preprocessing step that reduces the memory usage of the data

    It will reduce the memory usage of the data by changing the data types of the columns.
    """

    # (encodings are deprecated)
    id_encoding_path: str | None = None
    _encoding: dict = field(init=False, default_factory=dict, repr=False, compare=False)

    def run(self, data: pd.DataFrame) -> dict:
        """Run the preprocessing step.

        :param data: the data to preprocess
        :return: the preprocessed data
        """

        return self.preprocess(data)

    def preprocess(self, data: pd.DataFrame) -> dict:
        """Preprocess the data by reducing the memory usage of the data.

        :param data: the dataframe to preprocess
        :return: the preprocessed dataframe
        """
        return self.reduce_mem_usage(data)

    def reduce_mem_usage(self, data: pd.DataFrame) -> dict:
        """Reduce the memory usage of the data.

        :param data: the dataframe to reduce
        :return: the reduced dataframe
        """

        # convert series_id to int temporarily
        sids = data['series_id'].unique()
        mapping = {sid: i for i, sid in enumerate(sids)}
        data['series_id'] = data['series_id'].map(mapping)

        # convert timestamp to datetime and utc
        timestamp_pl = pl.from_pandas(pd.Series(data.timestamp, copy=False))
        utc = timestamp_pl.str.slice(21, 1).cast(pl.UInt16)
        timestamp_pl = timestamp_pl.str.slice(0, 19)

        timestamp_pl = timestamp_pl.str.to_datetime(format="%Y-%m-%dT%H:%M:%S", time_unit='ms')
        logger.debug("------ Done converting timestamp to datetime")
        data['timestamp'] = timestamp_pl
        data['utc'] = utc

        del timestamp_pl
        gc.collect()

        # convert data to smaller formats
        pad_type = {'step': np.int32, 'enmo': np.float32,
                    'anglez': np.float32, 'timestamp': 'datetime64[ns]', 'utc': np.uint16}
        data = data.astype(pad_type)
        gc.collect()

        # store each series in a different dict entry
        dfs_dict = {}
        for name, encoded in tqdm(mapping.items(), desc="Storing each series in a different dict entry"):
            dfs_dict[name] = data[data['series_id'] == encoded].drop(columns=['series_id']).reset_index(drop=True)
        del data
        gc.collect()
        return dfs_dict
