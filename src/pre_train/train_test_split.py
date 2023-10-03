import pandas as pd


def train_test_split(df: pd.Dataframe, test_size: int = 0.2) -> (pd.Dataframe, pd.Dataframe):
    return df.iloc[:-test_size], df.iloc[-test_size:]
