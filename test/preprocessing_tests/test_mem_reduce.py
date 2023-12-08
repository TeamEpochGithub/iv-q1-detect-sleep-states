from unittest import TestCase

import pandas as pd

from src.get_processed_data import mem_usage
from src.preprocessing.mem_reduce import MemReduce


class Test(TestCase):
    def test_repr(self):
        self.assertEqual("MemReduce()", MemReduce().__repr__())

    def test_mem_reduce(self):
        # read the data
        mem_reducer = MemReduce()
        train_series = pd.read_parquet("test/test_series.parquet")
        # if ran on test series the unique id_s json file is overwritten
        # so we need to save it and then put it back
        # this is because the series_id's are different between the two
        # series

        series_mem_used_before = train_series.memory_usage().sum()
        print('data usage of test_series before mem_reduce:', series_mem_used_before)
        # now do the mem_reduce

        output = mem_reducer.reduce_mem_usage(train_series)
        series_mem_used_after = mem_usage(output)
        print('data usage of test_series after mem_reduce:', series_mem_used_after)

        # assert that memory usage for both dataframes went down
        self.assertTrue(series_mem_used_before > series_mem_used_after)

    def test_timestamp_convert(self) -> None:
        data = pd.DataFrame({
            "series_id": ["038441c925bb", "038441c925bb"],
            "timestamp": ["2018-08-14T15:30:00-0400", "2018-12-18T12:56:50-0500"],
            "step": [0, 1],
            "enmo": [0.0, 0.0],
            "anglez": [0.0, 0.0],
        })

        mem_reducer = MemReduce()
        data = mem_reducer.preprocess(data)
