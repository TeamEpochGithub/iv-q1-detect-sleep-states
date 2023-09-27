# a new class with the same structure as mem reduce
#
# Path: src/preprocessing/add_noise.py

from ..preprocessing.pp import PP
import numpy as np


class AddNoise(PP):
    def __init__(self):
        pass

    def preprocess(self, data):
        # Add noise to the data
        # Create a new column with the cumulative sum of the anglez column
        # Create a new column with the cumulative sum of the anglez column
        print(data.shape)
        data['anglez'] = data['anglez'] + np.random.normal(0, 0.1, len(data['anglez']))
        print(data.shape)
        return data
