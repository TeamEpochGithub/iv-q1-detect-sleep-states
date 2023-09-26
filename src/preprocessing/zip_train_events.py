from src.preprocessing.pp import PP
import pandas as pd


class ZipTrainEvents(PP):
    """Preprocessing step that zips the training labels to the data

    It retrieves the training labels, matches on 'series_id', 'timestamp', 'step' with the dataset,
    and zips the event column to it.
    The values for each step are the strings "onset" or "wakeup" or None if there is no event
    """
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        train_events = pd.read_csv("./data/raw/train_events.csv")  # This is BAD, but I don't know how else to do it...

        zipped_df = df.merge(train_events, on=['series_id', 'timestamp', 'step'], how='left')
        return zipped_df[['series_id', 'step', 'timestamp', 'anglez', 'enmo', 'event']]
