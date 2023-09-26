from src.preprocessing.pp import PP
import pandas as pd


class Truncate(PP):
    """Preprocessing step that truncates the unlabelled end of the data

    After adding the "event" column with ZipTrainEvents,
    this will look at the last onset/wakeup event and truncate all data after that.
    """
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        last_index = df["event"].last_valid_index()
        return df.truncate(after=last_index)
