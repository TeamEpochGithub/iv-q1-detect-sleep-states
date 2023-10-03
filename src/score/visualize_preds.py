import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from src.get_processed_data import get_processed_data
from src.configs.load_config import ConfigLoader


def plot_preds_on_series(preds: pd.DataFrame, data: pd.DataFrame):
    '''The data is the featured adat we use at the end of the pipeline'''
    # for now the train events is hard coded
    # events = pd.read_csv('data/raw/train_events.csv')
    # get the unique series ids from the preds
    series_ids = preds['series_id'].unique()
    # loop through the series ids
    for id in series_ids:
        # after processing the data the series id is encoded to int
        # read the encoding json 
        f = open('series_id_encoding.json')
        id_encoding = json.load(f)
        id = id_encoding[id]
        current_series = data[data['series_id'] == id]
        # using a binary mask to index 1 out of n points to make plotting faster
        n = 250  # Set the value of n
        binary_mask = np.zeros(current_series.shape[0], dtype=int)
        binary_mask[::n] = 1
        plt.figure(figsize=(20, 20))
        # awake and series id are not features so -2
        num_features = len(current_series.columns) - 2
        for i, feature_column in enumerate(current_series.columns[2:]):  # Start from the third column
            plt.subplot(num_features, 1, i + 1)
            # sns.lineplot(data=train, x="step", y="anglez", hue="awake", linewidth=0.5)
            sns.lineplot(data=current_series[binary_mask], x='step', y=feature_column, hue='awake', linewidth=0.5)
            plt.xlabel('awake')
            plt.ylabel(feature_column)
            plt.title(f'Series ID: {id} - {feature_column}')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    preds = pd.read_csv('data/raw/train_events.csv')
    config = ConfigLoader("test/test_config.json")
    series_path = 'data/raw/train_series.parquet'
    featured_data = get_processed_data(config, series_path, save_output=True)
    plot_preds_on_series(preds, featured_data)
