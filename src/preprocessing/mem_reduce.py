# This class is to reduce memory usage of dataframe
from src.preprocessing.pp import PP
import numpy as np
import gc


class MemReduce(PP):

    def preprocess(self, df):
        df = self.reduce_mem_usage(df)
        return df

    def reduce_mem_usage(self, df):
        """
        Iterate through all numeric columns of a dataframe and modify the data type
        to reduce memory usage.
        """

        start_mem = df.memory_usage().sum() / 1024 ** 2
        print(f'Memory usage of dataframe is {start_mem:.2f} MB')

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object and col_type != 'datetime64[ns, UTC]':
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024 ** 2
        print(f'Memory usage after optimization is: {end_mem:.2f} MB')
        decrease = 100 * (start_mem - end_mem) / start_mem
        print(f'Decreased by {decrease:.2f}%')
        gc.collect()

        return df
