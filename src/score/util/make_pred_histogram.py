import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from src.logger.logger import logger
import os


def make_histogram(preds: pd.DataFrame, events: pd.DataFrame, folder_path: str, id_decoding: dict, series_id: int):

    tolerances = [0, 12, 36, 60, 90, 120, 150, 180, 240, 300, 360, 1000, 17280]
    scores = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, 0, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds.dropna(inplace=True)
        events.dropna(inplace=True)
    # Get the unique series ids
    # Loop through the series ids
    current_errors_onset = None
    current_errors_wakeup = None
    # the loop is here for running it on the complete data
    # in visualize preds this will be called once for each series id
    # for each id get the preds
    current_preds_onset = preds[(preds['event'] == 'onset')]['step'].to_numpy()
    current_preds_wakeup = preds[(preds['event'] == 'wakeup')]['step'].to_numpy()
    # and the events
    current_events_onset = events[(events['event'] == 'onset')]['step'].to_numpy()
    current_events_wakeup = events[(events['event'] == 'wakeup')]['step'].to_numpy()
    if not (len(current_preds_onset) == 0 or len(current_events_onset) == 0):
        current_errors_onset = match_preds(current_preds_onset, current_events_onset)

    # repeate the same process for the wakeups
    if not (len(current_preds_wakeup) == 0 or len(current_events_wakeup) == 0):
        current_errors_wakeup = match_preds(current_preds_wakeup, current_events_wakeup)

    # Now make the histogram
    if current_errors_onset is not None:
        hist_values, bin_edges = np.histogram(current_errors_onset, bins=tolerances)
        # make the bar names the ranges from the tolerances
        bar_labels = [f'{int(bin_edges[i]):d}-{int(bin_edges[i+1]):d}\n{scores[i]}' for i in range(len(bin_edges) - 1)]
        plt.figure(figsize=(20, 10))
        # replace the bin edges to be a range incremented by 10 starting from 0
        bin_edges = np.arange(0, 10*len(tolerances), 10)
        # Calculate the width of each bar
        bar_width = [5] * len(bin_edges[:-1])

        # Create a bar chart with custom widths and labels
        bars_onset = plt.bar(bin_edges[:-1], hist_values, width=bar_width, align='edge', tick_label=bar_labels)
        plt.xticks(rotation=45)
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.title('Histogram of the errors')
        for bar in bars_onset:
            height = bar.get_height()
            plt.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', rotation=45)

    # also make the histogram for the wakeups
    if current_errors_wakeup is not None:
        hist_values, bin_edges = np.histogram(current_errors_wakeup, bins=tolerances)
        # make the bar names the ranges from the tolerances
        bar_labels = [f'{int(bin_edges[i]):d}-{int(bin_edges[i+1]):d}\n{scores[i]}' for i in range(len(bin_edges) - 1)]
        # replace the bin edges to be a range incremented by 10 starting from 0
        bin_edges = np.arange(0, 10*len(tolerances), 10)
        # Calculate the width of each bar
        bar_width = [5] * len(bin_edges[:-1])

        # Create a bar chart with custom widths and labels
        bars_wakeup = plt.bar(bin_edges[:-1]+5, hist_values, width=bar_width, align='edge', tick_label=bar_labels)
        for bar in bars_wakeup:
            height = bar.get_height()
            plt.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', rotation=45)
    plt.legend(['onset', 'wakeup'])
    if not os.path.exists(folder_path + "/histograms"):
        os.makedirs(folder_path + "/histograms")
    plt.tight_layout()
    plt.savefig(folder_path + "/histograms" + "/" + "series_id--" + f"{id_decoding[series_id]}-({series_id}).png")
    logger.info(f"Histogram saved at {folder_path + '/histograms' + '/' + 'series_id--' + f'{id_decoding[series_id]}-({series_id}).png'}")
    plt.close()


def match_preds(preds: np.ndarray, events: np.ndarray):
    """ This function matches the predictions to the events and returns the errors fo each event.
        args: preds: the steps of the predictions for a series
              events: the steps of the events for a series
    """

    errors = []
    for event in events:
        # find the closest pred to the event
        # the closest error will be appended to errors
        errors.append(np.min(np.abs(preds - event)))

    return np.array(errors)
