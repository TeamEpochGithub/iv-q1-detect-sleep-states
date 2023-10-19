import pandas as pd
import json
import plotly.graph_objects as go
from src.get_processed_data import get_processed_data
from src.configs.load_config import ConfigLoader
import warnings
from plotly_resampler import FigureResampler
import os
import numpy as np


def plot_preds_on_series(preds: pd.DataFrame, data: pd.DataFrame, events_path: str = 'data/raw/train_events.csv',
                         features_to_plot: list = None, number_of_series_to_plot: int = 1, show_plot: bool = False,
                         ids_to_plot: list = None, folder_path: str = '') -> None:
    """ This function plots the predictions on the series data. It also plots the real events on the series data.
        The predictions and the events are vertical lines on the plot. The vertical lines have annotations that show
        the real and predicted values and the timestamps. The annotations are colored according to the event type.

    Args:
        preds (pd.DataFrame): The predictions dataframe. It must have the columns series_id, step, event.
        data (pd.DataFrame): The series data. It must have the columns series_id, step, anglez, enmo, awake, timestamp.
        events_path (str, optional): The path to the events csv file. Defaults to 'data/raw/train_events.csv'.
        features_to_plot (list, optional): The features to plot. Defaults to None.
        number_of_series_to_plot (int, optional): The number of series to plot. Defaults to 1.
        show_plot (bool, optional): Whether to show the plot or not. Defaults to False.
        ids_to_plot (list, optional): The encoded ids of the series to plot. Defaults to None.
        """
    # Make a plrediction_plots folder if it doesnt exist
    # Load the encoding JSON
    with open('series_id_encoding.json', 'r') as f:
        id_encoding = json.load(f)
    # Make a decoding for the title
    id_decoding = {v: k for k, v in id_encoding.items()}
    real_events = pd.read_csv(events_path)
    # Apply id encoding to events
    real_events['series_id'] = real_events['series_id'].map(id_encoding)
    # Apply id encoding to preds
    preds['series_id'] = preds['series_id'].map(id_encoding)
    # Get the unique ids after encoding
    series_ids = preds['series_id'].unique()
    # If ids_to_plot is not None, plot only the ids in the list
    if ids_to_plot is not None:
        series_ids = ids_to_plot
        number_of_series_to_plot = len(ids_to_plot)
    # Loop through the series ids
    for id in series_ids[:number_of_series_to_plot]:
        current_series = data[data['series_id'] == id]
        # Take the events for the current series
        current_events = real_events.loc[real_events['series_id'] == id]
        # Take the preds for the current series
        current_preds = preds.loc[preds['series_id'] == id]
        # Create the figure

        fig = go.Figure()
        # If features_to_plot is None, plot all the features
        if features_to_plot is None:
            features_to_plot = data.columns.values
        for feature_to_plot in features_to_plot:
            # Some features are not meant to be plotted like step, series_id, awake, timestamp
            if feature_to_plot == 'anglez' or feature_to_plot == 'enmo' or 'f_' in feature_to_plot:
                mask = current_series['awake'] == 0
                x = current_series['step'].to_numpy(copy=True, dtype=np.float32)
                x[~mask] = np.nan
                fig.add_trace(go.Scatter(x=x, y=current_series[feature_to_plot].values,
                                         mode='lines', name=feature_to_plot+'Awake=0',
                                         line=dict(color='blue'), legendgroup=feature_to_plot+'Awake=0',
                                         showlegend=True))
                mask = current_series['awake'] == 1
                x = current_series['step'].to_numpy(copy=True, dtype=np.float32)
                x[~mask] = np.nan
                fig.add_trace(go.Scatter(x=x, y=current_series[feature_to_plot].values,
                                         mode='lines', name=feature_to_plot+'Awake=1',
                                         line=dict(color='red'), legendgroup=feature_to_plot+'Awake=1',
                                         showlegend=True))
                mask = current_series['awake'] == 2
                x = current_series['step'].to_numpy(copy=True, dtype=np.float32)
                x[~mask] = np.nan
                fig.add_trace(go.Scatter(x=x, y=current_series[feature_to_plot].values,
                                         mode='lines', name=feature_to_plot+'Awake=2',
                                         line=dict(color='green'), legendgroup=feature_to_plot+'Awake=2',
                                         showlegend=True))
                fig.update_xaxes(title='Timestamp')
                fig.update_yaxes(title='Feature values')
                fig.update_layout(
                    title=f'Anglez for Series ID: {id_decoding[id]}-{id}')
                fig.update_xaxes(tickvals=current_series['step'][::len(
                    current_series) // 10], tickangle=45)

        # Before showing the figure make vertical lines for the current_events and current_preds
        # the vertical lines have annotations that show the real and predicted values and the timestamps
        # and the annotations are colored according to the event type
        for current_onset in current_events[current_events['event'] == 'onset']['step'].dropna():
            fig.add_vline(x=current_onset, line_dash="dash", line_color="black", line_width=2,
                          annotation_text=f'<span style="color:black">real_onset:<br> {current_onset}</span>',
                          annotation_position="top", name=f'Vertical Line at x={current_onset}', annotation_textangle=315,
                          )
        for current_wakeup in current_events[current_events['event'] == 'wakeup']['step'].dropna():
            fig.add_vline(x=current_wakeup, line_dash="dash", line_color="green", name="wakeup", line_width=2,
                          annotation_text=f'<span style="color:green">real_wakeup:<br> {current_wakeup}</span>',
                          annotation_position="top", annotation_textangle=315)

        for current_pred_onset in current_preds[current_preds['event'] == 'onset']['step'].dropna():
            fig.add_vline(x=current_pred_onset, line_dash="dash", line_color="red", line_width=2,
                          annotation_text=f'<span style="color:red">pred_onset:<br> {current_pred_onset}</span>',
                          annotation_position="top", name=f'Vertical Line at x={current_pred_onset}', annotation_textangle=315,
                          )
        for current_pred_wakeup in current_preds[current_preds['event'] == 'wakeup']['step'].dropna():
            fig.add_vline(x=current_pred_wakeup, line_dash="dash", line_color="orange", name="wakeup", line_width=2,
                          annotation_text=f'<span style="color:orange"> pred_wakeup:<br> {current_pred_wakeup}</span>',
                          annotation_position="top", annotation_textangle=315,
                          )
        # The normal margin doesnt work for the annotations
        fig.update_layout(margin=dict(t=160))
        if show_plot:
            fig.show()
        else:
            # This downsamples the plotly figure
            # It is not necessary but it makes the saving 8x faster
            fig = FigureResampler(fig)
            # If the hash config dir doesnt exist make it
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            fig.write_image(folder_path + '/' + 'series_id--' +
                            f'{id_decoding[id]}-({id}).jpeg', width=2000, height=600)


if __name__ == "__main__":
    # For testing you need a submission.csv file in the main folder
    # and it read the processed data
    preds = pd.read_csv("submission.csv")
    config = ConfigLoader("config.json")
    series_path = 'data/raw/train_series.parquet'
    featured_data = get_processed_data(config, series_path, save_output=True)
    # Plot the predictions on the series data for the chosen series_ids
    plot_preds_on_series(preds, featured_data,
                         number_of_series_to_plot=5, show_plot=True)
