# This class is to reduce memory usage of dataframe
from src.preprocessing.pp import PP


class MemReduce(PP):
    def __init__(self):
        # Initiate class
        pass

    def reduce_mem_usage(self, df):
        # Function to reduce memory usage of dataframe
        return df

    def run(self, df):
        # Function to run preprocessing step
        return self.reduce_mem_usage(df)
