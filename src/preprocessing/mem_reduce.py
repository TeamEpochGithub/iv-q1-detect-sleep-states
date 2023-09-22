# This class is to reduce memory usage of dataframe
from src.preprocessing.pp import PP
from src.preprocessing.dataframe_mem_reduce import reduce_mem_usage


class MemReduce(PP):
    def __init__(self):
        # Initiate class
        pass

    def preprocess(self, df):
        df = reduce_mem_usage(df)
        return df
