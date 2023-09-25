# TODO this garbage slow code is placeholder will be made effcient later
from src.preprocessing.pp import PP
import pandas as pd


class AddHour(PP):
    def __init__(self):
        pass

    def preprocess(self, data):
        # the code below should create a new dataframe with the following columns:
        # timestamp, hour, date, time
        data.rename(columns={"timestamp": "timestampOld"}, inplace=True)
        print('mark1')
        data['date'] = data["timestampOld"].str.split('T', expand=True)[0]
        print('mark2')
        data['time'] = data['timestampOld'].str.split('T', expand=True)[1].str.split('-', expand=True)[0]
        print('mark3')
        data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'])
        print('mark4')
        data['hour'] = data['timestamp'].dt.hour.astype("Int8")
        print('mark5')
        return data
