import pandas as pd
from src.preprocessing.pp import PP, PPException
from ..logger.logger import logger


class Truncate(PP):
    """Preprocessing step that truncates the unlabelled end of the data

    After adding the "awake" column with AddStateLabels,
    this will look at the last time the participant is either awake or asleep and truncate all data after that.
    """

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Truncates the unlabeled end of the data

        :param data: The labeled dataframe to truncate
        :return: The dataframe without the unlabeled end
        :raises PPException: If AddStateLabels wasn't used before
        """
        if "awake" not in data.columns:
            logger.critical("No awake column. Did you run AddStateLabels before?")
            raise PPException("No awake column. Did you run AddStateLabels before?")

        # Truncate does not work with windowing yet
        if "window" in data.columns:
            logger.critical("Truncate does not work with windowing yet")
            raise NotImplementedError()

        logger.info("------ Truncating data without windowing")
        return data.groupby("series_id").apply(lambda x: x.truncate(after=x[(x["awake"] != 2)].last_valid_index()))
