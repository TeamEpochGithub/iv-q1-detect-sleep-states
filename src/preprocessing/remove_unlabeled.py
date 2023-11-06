import pandas as pd

from ..logger.logger import logger
from ..preprocessing.pp import PP, PPException


class RemoveUnlabeled(PP):
    """Preprocessing step that removes the NaN and unlabeled data

    After adding the "awake" column with AddStateLabels, this will only keep the labeled data
    by dropping all the rows where "awake" is 3 (no more event data) and optionally 2 (don't make a prediction).
    If the "window" column in present, only drop the windows where all the "awake" values are 3 or 2.
    """

    def __init__(self, remove_partially_unlabeled_windows: bool, remove_nan: bool, remove_entire_series: bool,
                 **kwargs: dict) -> None:
        """Initialize the RemoveUnlabeled class

        :param remove_partially_unlabeled_windows: Only applies when windowing is used.
            If True, remove all windows that contain unlabeled data.
            If False, only remove the fully unlabeled windows.
        :param remove_nan: If false, remove the NaN data too. If True, keep the NaN data.
        :param remove_entire_series: If True, remove all series that contain unlabeled data. Ignores windowing.
        """
        super().__init__(**kwargs | {"kind": "remove_unlabeled"})
        self.remove_partially_unlabeled_windows = remove_partially_unlabeled_windows
        self.remove_nan = remove_nan
        self.remove_entire_series = remove_entire_series

    def __repr__(self) -> str:
        """Return a string representation of a RemoveUnlabeled object"""
        return (f"{self.__class__.__name__}(partially_unlabeled_windows={self.remove_partially_unlabeled_windows}, "
                f"remove_nan={self.remove_nan}, remove_entire_series={self.remove_entire_series})")

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

        logger.info(f"------ Data shape before: {data.shape}")

        if self.remove_entire_series:
            data = data.groupby(["series_id"]).filter(lambda x: (x['awake'] != 3).all()).reset_index(drop=True)
            if self.remove_nan:
                data = data.groupby(["series_id"]).filter(lambda x: (x['awake'] != 2).all()).reset_index(drop=True)

            logger.info(f"------ Data shape after: {data.shape}")
            return data

        if "window" in data.columns:
            logger.info("------ Removing unlabeled data with windowing")
            if self.remove_partially_unlabeled_windows:
                data = data.groupby(["series_id", "window"]).filter(lambda x: not (x['awake'] == 3).any()).reset_index(
                    drop=True)
            else:
                data = data.groupby(["series_id", "window"]).filter(lambda x: (x['awake'] != 3).any()).reset_index(
                    drop=True)

            if self.remove_nan:
                if self.remove_partially_unlabeled_windows:
                    data = data.groupby(["series_id", "window"]).filter(
                        lambda x: not (x['awake'] == 2).any()).reset_index(drop=True)
                else:
                    data = data.groupby(["series_id", "window"]).filter(lambda x: (x['awake'] != 2).any()).reset_index(
                        drop=True)

            logger.info(f"------ Data shape after: {data.shape}")
            return self.reset_windows_indices(data)

        logger.info("------ Removing unlabeled data without windowing")
        data = data[(data["awake"] != 3)].reset_index(drop=True)
        if self.remove_nan:
            data = data[(data["awake"] != 2)].reset_index(drop=True)

        logger.info(f"------ Data shape after: {data.shape}")
        return data

    @staticmethod
    def reset_windows_indices(data: pd.DataFrame) -> pd.DataFrame:
        """Reset the window number after removing unlabeled data

        :param data: The dataframe to reset the window number with columns "series_id" and "window"
        """
        data["window"] = (data.groupby("series_id")["window"].rank(method="dense").sub(1).astype(int)
                          .reset_index(drop=True))
        return data
