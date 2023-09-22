# This file is used to test different preprocessing steps

import pandas as pd
import sys
from src.preprocessing import dataframe_mem_reduce

# read the data
sys.path.insert(1, '../data')
train_events = pd.read_csv("../data/train_events.csv")
train_series = pd.read_parquet("../data/train_series.parquet")


print('data usage of train_series before optimization:' + train_series.memory_usage())
print('data usage of train_events before optimization:' + train_events.memory_usage())

# now do the mem_reduce

print('data usage of train_series before optimization:' + dataframe_mem_reduce(train_series).memory_usage())
print('data usage of train_events before optimization:' + dataframe_mem_reduce(train_series).memory_usage())
