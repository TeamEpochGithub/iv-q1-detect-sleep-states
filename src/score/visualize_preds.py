import numpy as np
import pandas as pd
import json
import plotly.express as px
from src.get_processed_data import get_processed_data
from src.configs.load_config import ConfigLoader


def plot_preds_on_series(preds: pd.DataFrame, data: pd.DataFrame, features_to_plot: list = None):
    '''The data is the featured adat we use at the end of the pipeline'''
    # for now the train events is hard coded
    # events = pd.read_csv('data/raw/train_events.csv')
    # get the unique series ids from the preds
    series_ids = preds['series_id'].unique()

    # Load the encoding JSON
    with open('series_id_encoding.json', 'r') as f:
        id_encoding = json.load(f)

    # Create an empty figure
    fig = px.line()

    # loop through the series ids
    for id in series_ids[:1]:
        # Encode the series ID
        id = id_encoding.get(str(id))

        if id is not None:
            current_series = data[data['series_id'] == id]
            real_events = pd.read_csv('data/raw/train_events.csv')
            # apply id encoding to events
            real_events['series_id'] = real_events['series_id'].map(id_encoding)
            current_events = real_events.loc[real_events['series_id'] == id]
            preds['series_id'] = preds['series_id'].map(id_encoding)
            current_preds = preds.loc[preds['series_id'] == id]
            awake_0_df = current_series[current_series['awake'] == 0]
            awake_1_df = current_series[current_series['awake'] == 1]
            awake_2_df = current_series[current_series['awake'] == 2]
            # current_series = current_series[binary_mask == 1]
            fig = px.line()
            for feature_to_plot in ['anglez', 'enmo']:
                # Create the plots for the current feature
                fig.add_scatter(x=awake_0_df['step'], y=awake_0_df[feature_to_plot], mode='lines', name=feature_to_plot+'Awake=0', line=dict(color='blue'))
                fig.add_scatter(x=awake_1_df['step'], y=awake_1_df[feature_to_plot], mode='lines', name=feature_to_plot+'Awake=1', line=dict(color='red'))
                fig.add_scatter(x=awake_2_df['step'], y=awake_2_df[feature_to_plot], mode='lines', name=feature_to_plot+'Awake=2', line=dict(color='green'))
                fig.update_xaxes(title='Timestamp')
                fig.update_yaxes(title='Feature values')
                fig.update_layout(title=f'Anglez for Series ID: {id}')
                fig.update_xaxes(tickvals=current_series['step'][::len(current_series) // 10], tickangle=45)
                # Show the first plot
            # before showing the figure make vertical lines for the current_events and current_preds
            for current_onset in current_events[current_events['event'] == 'onset']['step']:
                fig.add_vline(x=current_onset, line_width=1, line_dash="dash", line_color="black", name="onset")
            for current_wakeup in current_events[current_events['event'] == 'wakeup']['step']:
                fig.add_vline(x=current_wakeup, line_width=1, line_dash="dash", line_color="green", name="wakeup")
            fig.show()


if __name__ == "__main__":
    preds = pd.read_csv('data/raw/train_events.csv')
    config = ConfigLoader("test/test_config.json")
    series_path = 'data/raw/train_series.parquet'
    featured_data = get_processed_data(config, series_path, save_output=True)
    plot_preds_on_series(preds, featured_data)
