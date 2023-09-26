'''This class is used to add the state labels like awake, asleep and dont predict to the data'''

from src.preprocessing.pp import PP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class AddStateLabels(PP):
    def __init__(self):
        pass

    def preprocess(self, data):
        # Add state labels to the data
        # data is the series dataframe so we need to read
        # the events dataframe

        # Read the events dataframe
        events = pd.read_csv('data/raw/train_events.csv')
        for i, id in enumerate(events['series_id'].unique()):
            # Get the current series
            current_series = data[data['series_id'] == id]
            current_series = self.get_nan_train_series(current_series, events, id)
            # Set the NaN column to be an int
            # Save the current series to the data
            print(current_series.tail())
            data[data['series_id'] == id] = current_series
            print(i/len(events['series_id'].unique())*100, '% done')
        return data
    
    def get_nan_train_series(self, train_series, train_events, series):
        current_series = train_series[train_series["series_id"] == series]
        current_events = train_events[train_events["series_id"] == series].copy()

        # cleaning etc.
        # current_events = current_events.dropna()
        current_events["step"] = current_events["step"]
        current_events["awake"] = current_events["event"].replace({"onset": 1, "wakeup": 0})
        train = pd.merge(current_series, current_events[['step', 'awake']], on='step', how='left')

        # Set before filling so setup awake / sleeps
        train["awake"] = train["awake"].bfill(axis='rows')

        # Set future NaNs to 1 if step > last awake record
        train["NaN"] = np.where(train["awake"].isnull(), "unlabelled", "labelled")
        # Set NaNs to 1 if asleep
        train["NaN"] = np.where(train["awake"] == 0, "NaN", train["NaN"])
        train['awake'] = train['awake']

        # Fill final awakes
        train['awake'] = train['awake'].fillna(1)  # awake
        train["awake"] = train["awake"].astype("int")
        train.loc[(train["NaN"] == "NaN") | (train["NaN"] == "unlabelled"), "awake"] = 2
        train = train.drop("NaN", axis=1)
        train = train.rename(columns={"awake": "state"})
        plt.figure()
        plt.title("Series ID" + str(series))
        sns.lineplot(data=train, x="step", y="anglez", hue="state", linewidth=0.5)
        plt.show()
        return (train)
