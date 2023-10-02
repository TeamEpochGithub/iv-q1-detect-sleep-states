# This file is used to test different preprocessing steps
from unittest import TestCase
import pandas as pd
from src.preprocessing.mem_reduce import MemReduce
import json


class Test(TestCase):
    def test_mem_reduce(self):
        # read the data
        mem_reducer = MemReduce()
        train_series = pd.read_parquet("test/test_series.parquet")
        # if ran on test series the unique id_s json file is overwritten
        # so we need to save it and then put it back
        # this is because the series_id's are different between the two
        # series
        with open('series_id_encoding.json', 'r') as f:
            encoding = json.load(f)

        series_mem_used_before = train_series.memory_usage().sum()
        print('data usage of test_series before mem_reduce:', series_mem_used_before)
        # now do the mem_reduce

        series_mem_used_after = mem_reducer.reduce_mem_usage(train_series).memory_usage().sum()
        print('data usage of test_series after mem_reduce:', series_mem_used_after)

        # put the encoding back
        with open('series_id_encoding.json', 'w') as f:
            json.dump(encoding, f)
        # assert that memory usage for both dataframes went down
        self.assertTrue(series_mem_used_before > series_mem_used_after)
