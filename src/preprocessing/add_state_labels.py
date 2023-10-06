import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..logger.logger import logger
from ..preprocessing.pp import PP


class AddStateLabels(PP):
    """Adds state labels to each row of the data

    The state labels are added to the "awake" column based on the events csv file.
    The values are 0 for asleep, 1 for awake, and 2 for unlabeled.
    """

    def __init__(self, events_path: str, **kwargs) -> None:
        """Initialize the AddStateLabels class.

        :param events_path: the path to the events csv file
        :raises PPException: If no path to the events csv file is given
        """
        super().__init__(**kwargs)

        self.events_path: str = events_path
        self.events: pd.DataFrame = pd.DataFrame()

        # TODO Don't hardcode the file name, add it as a parameter in the config.json #99
        self.id_encoding_path: str = "series_id_encoding.json"
        self.id_encoding: dict = {}

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the preprocessing step.

        :param data: the data to preprocess
        :return: the preprocessed data
        :raises FileNotFoundError: If the events csv or id_encoding json file is not found
        """
        self.events = pd.read_csv(self.events_path)
        self.id_encoding = json.load(open(self.id_encoding_path))
        return self.preprocess(data)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by adding state labels to each row of the data.

        :param data: the data without state labels
        :return: the data with state labels added to the "awake" column
        """
        # Initialize the columns
        data['NaN'] = 0
        data['awake'] = 0

        # apply encoding to events
        self.events['series_id'] = self.events['series_id'].map(self.id_encoding)

        events_copy = self.events.copy()
        events_copy.dropna(inplace=True)
        # This part does some needed pp for getting the NaN series
        series_has_nan = self.events.groupby(
            'series_id')['step'].apply(lambda x: x.isnull().any())
        series_has_nan.value_counts()
        df_has_nan = series_has_nan.to_frame()
        df_has_nan.reset_index(inplace=True)

        # Finds series ids without NaN
        not_nan = df_has_nan.loc[df_has_nan.step == 0]["series_id"].to_list()
        # The series that do not have Nans but have missing labels at the end
        weird_series = ["0cfc06c129cc", "31011ade7c0a",
                        "55a47ff9dc8a", "a596ad0b82aa", "a9a2f7fac455"]
        # apply encoding to weird series
        weird_series_encoded = []
        for id in weird_series:
            if id in self.id_encoding:
                weird_series_encoded.append(self.id_encoding[id])

        # Firstly we loop with the series without NaN
        for i, id in tqdm(enumerate(not_nan)):
            current_series = self.get_train_series(data, events_copy, id)
            # after getting the awake column apply shift and fill to
            # set the event occurance properly
            current_series['awake'] = current_series['awake'].shift(-1).ffill()
            # if this is not done using a pandaas series to assign the value fills with nans
            awake_arr = current_series['awake'].to_numpy()
            # update the data awake column with the current series awake column
            data.loc[data['series_id'] == id, 'awake'] = awake_arr

        logger.debug("------ Finished handling series without NaN")
        # After handling the series without NaN we handle the weird cases
        # and add 2s for the awake labels
        for i, id in tqdm(enumerate(weird_series_encoded)):
            # get the events with the current series id
            current_events = self.events[self.events["series_id"] == id]
            # get the last item of the current events
            last_event = current_events.tail(1)
            # set awake of current series to 2 for all rows after last_event
            data.loc[(data['series_id'] == id) & (data['step'] >
                                                  last_event['step'].values[0]), 'awake'] = 2

        logger.debug("------ Finished handling weird series")
        # magic code i copied from EDA-Hugo to do the NaN stuff
        df_filled = self.events.copy()
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
            current_series = self.get_nan_train_series(
                current_series, nan_events, id)
            # For some reason the array is longer than the series so only take a slice same size as the series
            data.loc[data['series_id'] == id, 'NaN'] = current_series['NaN'].to_numpy()[
                                                       :data.loc[data['series_id'] == id].shape[0]]
            current_series = self.get_train_series(current_series, self.events, id)
            # after getting the awake column apply shift and fill to
            # set the event occurance properly
            current_series['awake'] = current_series['awake'].shift(-1).ffill()
            current_series.loc[current_series['NaN'] == 1, 'awake'] = 2
            # if this is not done using a pandaas series to assign the value fills with nans
            awake_arr = current_series['awake'].to_numpy()
            # For some reason the array is longer than the series so only take a slice same size as the series
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
