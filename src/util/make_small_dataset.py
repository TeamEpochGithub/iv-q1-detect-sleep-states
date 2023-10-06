import pandas as pd


def extract_first_n_series(n):
    """Loads the train series and events, filters on first N series,
     saves those to a new smaller raw dataset"""

    print('Loading series')
    train_series = pd.read_parquet('data/raw/train_series.parquet')
    train_events = pd.read_csv('data/raw/train_events.csv')
    print('Loaded series, filtering...')

    first_n_series = train_series['series_id'].unique()[:n]
    train_series = train_series[train_series['series_id'].isin(first_n_series)]
    train_events = train_events[train_events['series_id'].isin(first_n_series)]

    print('Made selection, saving...')
    train_series.to_parquet(f'data/raw/first_{n}_series.parquet')
    train_events.to_csv(f'data/raw/first_{n}_events.csv')
    print('Saved')

if __name__ == "__main__":
    extract_first_n_series(10)
