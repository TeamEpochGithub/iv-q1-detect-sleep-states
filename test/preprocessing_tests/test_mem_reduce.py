from unittest import TestCase

import pandas as pd

from src.get_processed_data import mem_usage
from src.preprocessing.mem_reduce import MemReduce


class Test(TestCase):
    def test_repr(self):
        self.assertEqual("MemReduce(id_encoding_path='dummy')", MemReduce(id_encoding_path='dummy').__repr__())

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
