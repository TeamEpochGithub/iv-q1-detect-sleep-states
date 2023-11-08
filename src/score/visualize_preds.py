import pandas as pd
import json
from src.get_processed_data import get_processed_data
from src.configs.load_config import ConfigLoader
from src.score.util.plot_with_plotly import plot_w_plotly
from src.score.util.save_plots import save_plots
from src.score.util.make_pred_histogram import make_histogram
from src.score.util.make_global_histogram import make_global_histogram


def plot_preds_on_series(preds: pd.DataFrame, data: pd.DataFrame, events_path: str = 'data/raw/train_events.csv',
                         features_to_plot: list = None, number_of_series_to_plot: int = 1, show_plot: bool = False,
                         ids_to_plot: list = None, folder_path: str = '', save_figures: bool = True) -> None:
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
        if features_to_plot is None:
            features_to_plot = data.columns.values
        if show_plot:
            # If features_to_plot is None, plot all the features
            plot_w_plotly(current_series, current_events,
                          current_preds, id_decoding, id, features_to_plot)
        if save_figures:
            save_plots(current_series, current_events,
                       current_preds, id_decoding, id, features_to_plot, folder_path+'/series_plots')

            make_histogram(current_preds, current_events, folder_path, id_decoding, id)
    # now make the global histogram
    make_global_histogram(preds, real_events, folder_path=folder_path)


if __name__ == "__main__":
    # For testing you need a submission.csv file in the main folder
    # and it read the processed data
    preds = pd.read_csv("submission.csv")
    config_loader = ConfigLoader("config.json")
    series_path = 'data/raw/train_series.parquet'
    featured_data = get_processed_data(config_loader, series_path, save_output=True)
    # Plot the predictions on the series data for the chosen series_ids
    plot_preds_on_series(preds, featured_data,
                         number_of_series_to_plot=5, show_plot=True, save_figures=False)
