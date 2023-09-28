'''This class is used to add the state labels like pseudo-NaN, asleep and dont predict to the data'''

from src.preprocessing.pp import PP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class AddStateLabels(PP):

    def preprocess(self, data):
        # the data will get a Nan column
        # the code for the series without nan and with nan is not the same
        # so we have 2 separate loops
        data['NaN'] = 0
        # Read the events dataframe
        events = pd.read_csv('data/raw/train_events.csv')
        # This part does someneeded pp for getting the NaN series
        df_filled = events.copy()
        onset_mask = df_filled['event'] == 'onset'
        wakeup_mask = df_filled['event'] == 'wakeup'
        df_filled.loc[onset_mask, 'step'] = df_filled.groupby('series_id')['step'].ffill().bfill()
        df_filled.loc[wakeup_mask, 'step'] = df_filled.groupby('series_id')['step'].bfill().ffill()
        nan_events = df_filled[pd.isnull(df_filled['timestamp'])].copy()
        nan_series = nan_events['series_id'].unique()

        # This part figures out the series witouth Nan
        series_has_NaN = events.groupby('series_id')['step'].apply(lambda x: x.isnull().any())
        series_has_NaN.value_counts()
        df_has_NaN = series_has_NaN.to_frame()
        df_has_NaN.reset_index(inplace=True)
        notNaN = df_has_NaN.loc[df_has_NaN.step == 0]["series_id"].to_list()

        # Firstly we loop with the series without NaN
        for i, id in enumerate(notNaN):
            # Get the current series
            print(id)
            current_series = data[data['series_id'] == id]
            # Save the current series to the data
            data.loc[data['series_id'] == id, 'NaN'] = current_series['NaN']
            print(i/len(events['series_id'].unique())*100, '% done')

            # after the mask is applied to the series 
            # do the asleep awake thing
            # then apply the mask to the data

            current_series = self.get_train_series(current_series, events, id)
            # now apply the mask to the data
            data.loc[data['series_id'] == id, 'awake'] = current_series['awake']
            plt.figure()
            plt.title("Series ID" + str(id))
            sns.lineplot(data=current_series, x="step", y="anglez", hue="awake", linewidth=0.5)
            plt.show()

        # now loop with the series with NaN
        for i, id in enumerate(nan_series):
            # Get the current series
            current_series = data[data['series_id'] == id]
            current_series = self.get_nan_train_series(current_series, nan_events, id)
            # Set the NaN column to be an int
            # Save the current series to the data
            data.loc[data['series_id'] == id, 'NaN'] = current_series['NaN']
            print(i/len(events['series_id'].unique())*100, '% done')

            # after the mask is applied to the series 
            # do the asleep awake thing
            # then apply the mask to the data

            current_series = self.get_train_series(current_series, events, id)
            # now apply the mask to the data
            current_series.loc[current_series['NaN'] == 1, 'awake'] = 2
            plt.figure()
            plt.title("Series ID" + str(id))
            sns.lineplot(data=current_series, x="step", y="anglez", hue="awake", linewidth=0.5)
            plt.show()
        return data

    def get_nan_train_series(self, current_series, train_events, series):
        current_events = train_events[train_events["series_id"] == series].copy()

        # cleaning etc.
        # current_events = current_events.dropna()
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
        plt.figure()
        sns.lineplot(data=train, x="step", y="anglez", hue="NaN", linewidth=0.5)
        plt.show()
        return (train)

    def get_train_series(self, train_series, train_events, series):
        current_series = train_series[train_series["series_id"] == series]
        current_events = train_events[train_events["series_id"] == series]

        # cleaning etc.
        current_events = current_events.dropna()
        current_events["step"] = current_events["step"].astype("int")
        current_events["awake"] = current_events["event"].replace({"onset": 1, "wakeup": 0})

        train = pd.merge(current_series, current_events[['step', 'awake']], on='step', how='left')
        train["awake"] = train["awake"].bfill(axis='rows')
        # final section:
        # train_events.groupby('series_id').tail(1)["event"].unique()
        # Result: the last event is always a "wakeup"
        train['awake'] = train['awake'].fillna(1)  # awake
        train["awake"] = train["awake"].astype("int")
        
        return train
