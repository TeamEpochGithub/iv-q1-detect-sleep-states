from .feature_engineering import FE
import numpy as np


class Derivative(FE):

    def __init__(self, config):
        super().__init__(config)

    def feature_engineering(self, data):
        for id in data['series_id'].unique():
            data.loc[data['series_id'] == id, 'grad_anglez'] = np.gradient(data.loc[data['series_id'] == id, 'anglez'])
        return data
