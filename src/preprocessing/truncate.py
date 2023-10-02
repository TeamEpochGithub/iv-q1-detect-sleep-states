import pandas as pd
from src.preprocessing.pp import PP, PPException


class Truncate(PP):
    """Preprocessing step that truncates the unlabelled end of the data

    After adding the "awake" column with AddStateLabels,
    this will look at the last time the participant is either awake or asleep and truncate all data after that.
    """
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Truncates the unlabelled end of the data

        :param data: The labeled dataframe to truncate
        :return: The dataframe without the unlabeled end
        :raises PPException: If AddStateLabels wasn't used before
        """
        if "awake" not in data.columns:
            raise PPException("No awake column. Did you run AddStateLabels before?")

        # Truncate does not work with windowing yet
        if "window" in data.columns:
            raise NotImplementedError()

        last_index: int = data[(data["awake"] != 2)].last_valid_index()
        return data.truncate(after=last_index)
