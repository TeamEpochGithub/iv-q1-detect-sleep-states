'''This class is used to add the state labels like pseudo-NaN, asleep and dont predict to the data'''
from ..logger.logger import logger
from src.preprocessing.pp import PP
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


class AddStateLabels(PP):

    def __init__(self):
        self.id_encoding = {}

    def preprocess(self, data):
        # Initialize the columns
        data['NaN'] = 0
        data['awake'] = 0
        # Read the events dataframe
        events = pd.read_csv('data/raw/train_events.csv')
        # apply ecoding to events
        if os.path.exists('series_id_encoding.json'):
            f = open('series_id_encoding.json')
            self.id_encoding = json.load(f)
        events['series_id'] = events['series_id'].map(self.id_encoding)
        events_copy = events.copy()
        events_copy.dropna(inplace=True)
        # This part does some needed pp for getting the NaN series
        series_has_nan = events.groupby(
            'series_id')['step'].apply(lambda x: x.isnull().any())
        series_has_nan.value_counts()
        df_has_NaN = series_has_nan.to_frame()
        df_has_NaN.reset_index(inplace=True)

        # Finds series ids without NaN
        not_nan = df_has_NaN.loc[df_has_NaN.step == 0]["series_id"].to_list()
        # The series that do not have Nans but have missing labels at the end
        weird_series = ["0cfc06c129cc", "31011ade7c0a",
                        "55a47ff9dc8a", "a596ad0b82aa", "a9a2f7fac455"]
        # convert the weird series
        for i in range(len(weird_series)):
            weird_series[i] = self.id_encoding[weird_series[i]]
        # if mem reduce has been applied before the encoding must be used

        # Firstly we loop with the series without NaN
        for i, id in tqdm(enumerate(not_nan)):
            # Get the current series
            # Save the current series to the data
            # use the encoding to get the original series id
            current_series = self.get_train_series(data, events_copy, id)
            current_series['awake'] = current_series['awake'].shift(-1).ffill()#.fillna(current_series['awake'].iloc[-1])
            # this is needed because pandas is stupid
            awake_arr = current_series['awake'].to_numpy()
            # update the data awake column with the current series awake column
            data.loc[data['series_id'] == id, 'awake'] = awake_arr

        logger.debug("------ Finished handling series without NaN")
        # After handling the series without NaN we handle the weird cases
        # and add 2s for the awake labels
        for i, id in tqdm(enumerate(weird_series)):
            # get the events with the current series id
            current_events = events[events["series_id"] == id]
            # get the last item of the current events
            last_event = current_events.tail(1)
            # set awake of current series to 2 for all rows after last_event
            data.loc[(data['series_id'] == id) & (data['step'] >
                                                  last_event['step'].values[0]), 'awake'] = 2

        logger.debug("------ Finished handling weird series")
        # magic code i copied from EDA-Hugo to do the NaN stuff
        df_filled = events.copy()
        onset_mask = df_filled['event'] == 'onset'
        wakeup_mask = df_filled['event'] == 'wakeup'
        df_filled.loc[onset_mask, 'step'] = df_filled.groupby('series_id')[
            'step'].ffill().bfill()
        df_filled.loc[wakeup_mask, 'step'] = df_filled.groupby('series_id')[
            'step'].bfill().ffill()
        nan_events = df_filled[pd.isnull(df_filled['timestamp'])].copy()
        nan_series = nan_events['series_id'].unique()

        # now loop with the series with NaN
        for i, id in tqdm(enumerate(nan_series)):
            current_series = data[data['series_id'] == id]
            # make a decoder to decode the id back to the original id
            # get the current series id

            current_series = self.get_nan_train_series(
                current_series, nan_events, id)
            data.loc[data['series_id'] == id, 'NaN'] = current_series['NaN'].to_numpy()[:data.loc[data['series_id'] == id].shape[0]]
            current_series = self.get_train_series(current_series, events, id)
            current_series['awake'] = current_series['awake'].shift(-1).ffill()
            current_series.loc[current_series['NaN'] == 1, 'awake'] = 2
            awake_arr = current_series['awake'].to_numpy()
            data.loc[data['series_id'] == id, 'awake'] = awake_arr[:data.loc[data['series_id'] == id].shape[0]]

        logger.debug("------ Finished handling series with NaN")

        if 'NaN' in data.columns:
            data.drop(columns=['NaN'], inplace=True)
        # convert the awake column to int8
        data['awake'] = data['awake'].astype('int8')
        return data

    # This is copied over from EDA-Hugo
    def get_nan_train_series(self, current_series, train_events, series):
        current_events = train_events[train_events["series_id"] == series].copy(
        )

        current_events["pseudo-NaN"] = current_events["event"].replace(
            {"onset": 1, "wakeup": 0})
        train = pd.merge(current_series, current_events[[
                         'step', 'pseudo-NaN']], on='step', how='left')

        # Set before filling so setup pseudo-NaN / sleeps
        train["pseudo-NaN"] = train["pseudo-NaN"].bfill(axis='rows')
        # Set future NaNs to 1 if step > last pseudo-NaN record
        train["NaN"] = np.where(
            train["pseudo-NaN"].isnull(), "unlabelled", "labelled")
        # Set NaNs to 1 if asleep
        train["NaN"] = np.where(train["pseudo-NaN"] == 0, "NaN", train["NaN"])
        # Fill final pseudo-NaNs
        train['pseudo-NaN'] = train['pseudo-NaN'].fillna(1)  # pseudo-NaN
        train["pseudo-NaN"] = train["pseudo-NaN"].astype("int")
        mask = (train["NaN"] == "unlabelled") | (train["NaN"] == "NaN")
        train.loc[mask, "NaN"] = 1
        train.loc[~mask, "NaN"] = 0
        train["NaN"] = train["NaN"].astype("int")
        return train

    # This is copied over from EDA-Hugo
    def get_train_series(self, train_series, train_events, series):
        current_events = train_events[train_events["series_id"] == series]
        current_series = train_series[train_series["series_id"] == series]
        # cleaning etc.
        current_events = current_events.dropna()
        current_events["step"] = current_events["step"].astype("int")
        current_events["awake"] = current_events["event"].replace(
            {"onset": 1, "wakeup": 0})

        train = pd.merge(current_series, current_events[[
                         'step', 'awake']], on='step', how='left')
        if 'awake_y' in train.columns:
            train.rename(columns={'awake_y': 'awake'}, inplace=True)
        if 'awake_x' in train.columns:
            train.drop(columns=['awake_x'], inplace=True)
        train["awake"] = train["awake"].bfill(axis='rows')

        train['awake'] = train['awake'].fillna(1)  # awake
        train["awake"] = train["awake"].astype("int")

        return train
