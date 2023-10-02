# Create a class for ensemble learning

# Imports
import numpy as np
import pandas as pd


class Ensemble:

    # Init function
    def __init__(self, models=None, weight_matrix=None, combination_method="addition"):
        if models is None:
            self.models = []
        else:
            self.models = models

        if weight_matrix is None:
            self.weight_matrix = np.ones(len(self.models))
        else:
            self.weight_matrix = weight_matrix

    def pred(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prediction function for the ensemble.
        Feeds the models data window-by-window, averages their predictions
        and converts the window-relative steps to absolute steps since the start of the series

        :param data: complete dataset with engineered features
        :return: dataframe, example:
                   series_id  window     onset    wakeup
            0   038441c925bb     0.0    5046.0    8794.0
            1   038441c925bb     1.0   20336.0   23482.0
            2   038441c925bb     2.0   40049.0   44322.0
        """

        print("Predicting with ensemble")
        # Run each model
        predictions = []
        for model in self.models:
            # group data by series_id, apply model.pred to each group, and get the output pairs
            # get the step at the index of the prediction
            model_pred = (data
                          .groupby(['series_id', 'window'])
                          .apply(lambda x: pred_window(x, model))
                          .reset_index(0, drop=True))

            # split the series of tuples into two column
            predictions.append(model_pred.to_list())

        # Weight the predictions
        predictions = np.array(predictions)
        predictions = np.average(
            predictions, axis=0, weights=self.weight_matrix)

        # Return a dataframe with the prediction for each series_id and window
        output_df = (data.groupby(['series_id', 'window'])
                         .apply(lambda x: x.iloc[0][['series_id', 'window']])
                         .reset_index(drop=True))
        output_df['onset'] = predictions[:, 0]
        output_df['wakeup'] = predictions[:, 1]
        return output_df


def pred_window(window: pd.DataFrame, model):
    """
    Get the step value for this window predicted by the model
    :param window: one window of the data
    :param model:
    :return: predicted onset and wakeup, absolute (relative to start of series)
    """
    # get the step at the index of the prediction
    onset_rel, wakeup_rel = model.pred(window)
    onset = window['step'].iloc[onset_rel] if onset_rel is not np.nan else np.nan
    wakeup = window['step'].iloc[wakeup_rel] if wakeup_rel is not np.nan else np.nan
    return onset, wakeup
