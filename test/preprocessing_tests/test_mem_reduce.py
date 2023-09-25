# This file is used to test different preprocessing steps
from unittest import TestCase
import pandas as pd
import sys
from src.preprocessing.mem_reduce import MemReduce


class Test(TestCase):
    def test_mem_reduce(self):
        sys.path.insert(1, '../data')
        # read the data
        train_events = pd.read_csv("data/train_events.csv")
        train_series = pd.read_parquet("data/train_series.parquet")

        print('data usage of train_series before mem_reduce:', '\n')
        series_mem_used_before = train_series.memory_usage().sum()
        print('data usage of train_events before mem_reduce:' '\n')
        event_mem_used_before = train_events.memory_usage().sum()

        # now do the mem_reduce

        print('data usage of train_series after mem_reduce:', '\n')
        series_mem_used_after = MemReduce.reduce_mem_usage(train_series).memory_usage().sum()
        print('data usage of train_events after mem_reduce:', '\n')
        event_mem_used_after = MemReduce.reduce_mem_usage(train_events).memory_usage().sum()

        # assert that memory usage for both dataframes went down
        self.assertTrue(series_mem_used_before > series_mem_used_after and event_mem_used_before > event_mem_used_after)
