# This class is to reduce memory usage of dataframe
from ..preprocessing.pp import PP
import json


class MemReduce(PP):

    def preprocess(self, data):
        df = self.reduce_mem_usage(data)
        return df

    def reduce_mem_usage(self, data):
        # we should make the series id in to an int16
        # and save an encoding (a dict) as a json file somewhere
        # so we can decode it later
        encoding = dict(zip(data['series_id'].unique(), range(len(data['series_id'].unique()))))
        with open('series_id_encoding.json', 'w') as f:
            json.dump(encoding, f)
        data['series_id'] = data['series_id'].map(encoding)
        data['series_id'] = data['series_id'].astype('int16')
        return data
