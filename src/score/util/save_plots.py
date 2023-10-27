import os

import matplotlib.pyplot as plt
import numpy as np

from src.logger.logger import logger


def save_plots(current_series, current_events, current_preds, id_decoding, id, features_to_plot, folder_path):
    # get the max and min of the features to plot
    # from the dataframe for the vlines
    selected_columns = current_series.columns[current_series.columns.str.startswith("f_") |
                                              current_series.columns.isin(["anglez", "enmo"])]
    max_value = current_series[selected_columns].max().max()
    min_value = current_series[selected_columns].min().min()
    plt.figure(figsize=(80, 20))
    plt.vlines(x=current_events[current_events['event'] == 'onset']['step'].dropna(), colors='black',
               linestyles='dashed', label='real_onset', ymax=max_value, ymin=min_value)
    plt.vlines(x=current_events[current_events['event'] == 'wakeup']['step'].dropna(), colors='green',
               linestyles='dashed', label='real_wakeup', ymax=max_value, ymin=min_value)
    plt.vlines(x=current_preds[current_preds['event'] == 'onset']['step'].dropna(), colors='red',
               linestyles='dashed', label='pred_onset', ymax=max_value, ymin=min_value)
    plt.vlines(x=current_preds[current_preds['event'] == 'wakeup']['step'].dropna(), colors='orange',
               linestyles='dashed', label='pred_wakeup', ymax=max_value, ymin=min_value)

    for feature_to_plot in features_to_plot:
        # Some features are not meant to be plotted like step, series_id, awake, timestamp
        if feature_to_plot == 'anglez' or feature_to_plot == 'enmo' or 'f_' in feature_to_plot:
            mask = current_series['awake'] == 0
            x = current_series['step'].to_numpy(copy=True, dtype=np.float32)
            x[~mask] = np.nan
            plt.plot(x, current_series[feature_to_plot].values, color='blue')

            mask = current_series['awake'] == 1
            x = current_series['step'].to_numpy(copy=True, dtype=np.float32)
            x[~mask] = np.nan
            plt.plot(x, current_series[feature_to_plot].values, color='red')

            mask = current_series['awake'] == 2
            x = current_series['step'].to_numpy(copy=True, dtype=np.float32)
            x[~mask] = np.nan
            plt.plot(x, current_series[feature_to_plot].values, color='green')

            plt.xlabel('Timestamp')
            plt.ylabel('Feature values')
            plt.title(f'Anglez for Series ID: {id_decoding[id]}-{id}')
            plt.xticks(current_series['step'][::len(current_series) // 10])
            # rotate the xticks
            plt.xticks(rotation=45)

    # Add the legend
    plt.legend(['real_onset', 'real_wakeup', 'pred_onset', 'pred_wakeup', 'Awake=0', 'Awake=1', 'Awake=2'], loc='upper right')
    # If the hash config dir doesnt exist make it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path + '/' + 'series_id--' +
                f'{id_decoding[id]}-({id}).png')
    plt.close()
    logger.info(f'Plot saved at: {folder_path + "/" + "series_id--" + f"{id_decoding[id]}-({id}).png"}')
