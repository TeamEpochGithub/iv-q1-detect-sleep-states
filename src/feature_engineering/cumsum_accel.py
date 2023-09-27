from src.feature_engineering.fe import FE
import pandas as pd


class cumsum_accel(FE):
    def __init__(self):
        pass

    def fe(self, data):
        # Create a new column with the cumulative sum of the anglez column
        return pd.DataFrame({'cumsum_accel': data['anglez'].cumsum()})
