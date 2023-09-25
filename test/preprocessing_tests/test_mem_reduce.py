# This file is used to test different preprocessing steps
from unittest import TestCase
import pandas as pd
from src.preprocessing.mem_reduce import MemReduce


class Test(TestCase):
    def test_mem_reduce(self):
        # read the data
        mem_reducer = MemReduce()
        train_series = pd.read_parquet("test/test_series.parquet")

        print('data usage of test_series before mem_reduce:', '\n')
        series_mem_used_before = train_series.memory_usage().sum()

        # now do the mem_reduce

        print('data usage of test_series after mem_reduce:', '\n')
        series_mem_used_after = mem_reducer.reduce_mem_usage(train_series).memory_usage().sum()

        # assert that memory usage for both dataframes went down
        self.assertTrue(series_mem_used_before > series_mem_used_after)
