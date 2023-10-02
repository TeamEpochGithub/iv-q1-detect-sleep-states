import pandas as pd
from src.preprocessing.pp import PP, PPException


class RemoveUnlabeled(PP):
    """Preprocessing step that removes the unlabelled data

    After adding the "awake" column with AddStateLabels, this will only keep the labeled data
    by dropping all the columns where "awake" is 2.
    If the "window" column in present, only drop the windows where all the "awake" values are 2.
    """

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Removes all the data points where there is no labeled data

        :param data: The labeled dataframe to remove the unlabeled data from
        :return: The dataframe without the unlabeled data
        :raises PPException: If AddStateLabels wasn't used before
        """
        if "awake" not in data.columns:
            raise PPException("No awake column. Did you run AddStateLabels before?")

        if "window" not in data.columns:
            return data[(data["awake"] != 2)]

        return data.groupby(["window"]).filter(lambda x: (x['awake'] != 2).any())
