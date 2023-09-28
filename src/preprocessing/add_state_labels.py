'''This class is used to add the state labels like pseudo-NaN, asleep and dont predict to the data'''

from src.preprocessing.pp import PP
import pandas as pd
import numpy as np


class AddStateLabels(PP):

    def preprocess(self, data):
        # initialize the columns
        data['NaN'] = 0
        data['awake'] = 0
        # Read the events dataframe
        events = pd.read_csv('data/raw/train_events.csv')
        events_copy = events.copy()
        events_copy.dropna(inplace=True)
        # This part does some needed pp for getting the NaN series
        series_has_NaN = events.groupby('series_id')['step'].apply(lambda x: x.isnull().any())
        series_has_NaN.value_counts()
        df_has_NaN = series_has_NaN.to_frame()
        df_has_NaN.reset_index(inplace=True)

        # this finds the series ids without NaN
        notNaN = df_has_NaN.loc[df_has_NaN.step == 0]["series_id"].to_list()
        weird_series = ["0cfc06c129cc", "31011ade7c0a", "55a47ff9dc8a", "a596ad0b82aa", "a9a2f7fac455"]
        # Firstly we loop with the series without NaN

        for i, id in enumerate(notNaN):
            # Get the current series
            # Save the current series to the data
            current_series = self.get_train_series(data, events_copy, id)
            # this is needed
            # idk why but without it it doesnt work
            current_series = current_series.set_index('step')
            data = data.set_index('step')
            # update the data awake column with the current series awake column
            data.loc[data['series_id'] == id, 'awake'] = current_series['awake']

        # after handling the series without NaN we handle the weird cases
        # and add 2s for the awake labels
        for i, id in enumerate(weird_series):
            # get the events with the current series id
            current_events = events[events["series_id"] == id]
            # get the last item of the current events
            last_event = current_events.tail(1)
            # set awake of current series to 2 for all rows after last_event
            data.loc[(data['series_id'] == id) & (data['step'] > last_event['step'].values[0]), 'awake'] = 2

        # magic code i copied from EDA-Hugo to do the NaN stuff
        df_filled = events.copy()
        onset_mask = df_filled['event'] == 'onset'
        wakeup_mask = df_filled['event'] == 'wakeup'
        df_filled.loc[onset_mask, 'step'] = df_filled.groupby('series_id')['step'].ffill().bfill()
        df_filled.loc[wakeup_mask, 'step'] = df_filled.groupby('series_id')['step'].bfill().ffill()
        nan_events = df_filled[pd.isnull(df_filled['timestamp'])].copy()
        nan_series = nan_events['series_id'].unique()
        # now loop with the series with NaN
        for i, id in enumerate(nan_series):
            # Get the current series
            current_series = data[data['series_id'] == id]
            current_series = self.get_nan_train_series(current_series, nan_events, id)

            data.loc[data['series_id'] == id, 'NaN'] = current_series['NaN']
            print(i/len(events['series_id'].unique())*100, '% done')
            current_series = self.get_train_series(current_series, events, id)
            current_series.loc[current_series['NaN'] == 1, 'awake'] = 2

            data.loc[data['series_id'] == id, 'awake'] = current_series['awake']

        if 'NaN' in data.columns:
            data.drop(columns=['NaN'], inplace=True)
        return data

    # This is copied over from EDA-Hugo
    def get_nan_train_series(self, current_series, train_events, series):
        current_events = train_events[train_events["series_id"] == series].copy()

        current_events["pseudo-NaN"] = current_events["event"].replace({"onset": 1, "wakeup": 0})
        train = pd.merge(current_series, current_events[['step', 'pseudo-NaN']], on='step', how='left')

        # Set before filling so setup pseudo-NaN / sleeps
        train["pseudo-NaN"] = train["pseudo-NaN"].bfill(axis='rows')
        # Set future NaNs to 1 if step > last pseudo-NaN record
        train["NaN"] = np.where(train["pseudo-NaN"].isnull(), "unlabelled", "labelled")
        # Set NaNs to 1 if asleep
        train["NaN"] = np.where(train["pseudo-NaN"] == 0, "NaN", train["NaN"])
        # Fill final pseudo-NaNs
        train['pseudo-NaN'] = train['pseudo-NaN'].fillna(1)  # pseudo-NaN
        train["pseudo-NaN"] = train["pseudo-NaN"].astype("int")
        mask = (train["NaN"] == "unlabelled") | (train["NaN"] == "NaN")
        train.loc[mask, "NaN"] = 1
        train.loc[~mask, "NaN"] = 0
        train["NaN"] = train["NaN"].astype("int")
        return (train)

    # This is copied over from EDA-Hugo
    def get_train_series(self, train_series, train_events, series):
        current_series = train_series[train_series["series_id"] == series]
        current_events = train_events[train_events["series_id"] == series]

        # cleaning etc.
        current_events = current_events.dropna()
        current_events["step"] = current_events["step"].astype("int")
        current_events["awake"] = current_events["event"].replace({"onset": 1, "wakeup": 0})

        train = pd.merge(current_series, current_events[['step', 'awake']], on='step', how='left')
        if 'awake_y' in train.columns:
            train.rename(columns={'awake_y': 'awake'}, inplace=True)
        if 'awake_x' in train.columns:
            train.drop(columns=['awake_x'], inplace=True)
        train["awake"] = train["awake"].bfill(axis='rows')

        train['awake'] = train['awake'].fillna(1)  # awake
        train["awake"] = train["awake"].astype("int")

        return (train)
