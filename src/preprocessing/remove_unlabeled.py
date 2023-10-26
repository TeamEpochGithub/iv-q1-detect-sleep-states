import pandas as pd

from ..logger.logger import logger
from ..preprocessing.pp import PP, PPException


class RemoveUnlabeled(PP):
    """Preprocessing step that removes the unlabelled data

    After adding the "awake" column with AddStateLabels, this will only keep the labeled data
    by dropping all the rows where "awake" is 2.
    If the "window" column in present, only drop the windows where all the "awake" values are 2.
    """

    def __init__(self, **kwargs: dict) -> None:
        """Initialize the RemoveUnlabeled class"""
        super().__init__(**kwargs | {"kind": "remove_unlabeled"})

    def __repr__(self) -> str:
        """Return a string representation of a RemoveUnlabeled object"""
        return f"{self.__class__.__name__}()"

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Removes all the data points where there is no labeled data

        :param data: The labeled dataframe to remove the unlabeled data from
        :return: The dataframe without the unlabeled data
        :raises PPException: If AddStateLabels wasn't used before
        """
        # TODO Do this check with the config checker instead #190
        if "awake" not in data.columns:
            logger.critical("No awake column. Did you run AddStateLabels before?")
            raise PPException("No awake column. Did you run AddStateLabels before?")

        if "window" in data.columns:
            logger.info("------ Removing unlabeled data with windowing")
            return data.groupby(["window"]).filter(lambda x: (x['awake'] != 2).any()).reset_index(drop=True)

        logger.info("------ Removing unlabeled data without windowing")
        return data[(data["awake"] != 2)].reset_index(drop=True)
