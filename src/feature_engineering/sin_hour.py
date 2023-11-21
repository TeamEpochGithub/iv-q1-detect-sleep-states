import pandas as pd

from .feature_engineering import FE
from ..logger.logger import logger
import numpy as np
from dataclasses import dataclass


@dataclass
class SinHour(FE):
    """

    # This step will take the hour from the column with the datetime
    and map the hours between 0-2*pi and take the sin of it
    Unless this is done the hour features spectrogram will have harmonics
    """

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the data. This method should be overritten by the child class.

        :param data: the data to process
        :return: the processed data
        """
        # assert that the data has a timestamp column
        assert "timestamp" in data.columns, "dataframe has no timestamp column"

        # get the hour from the datetime column
        hour = data['timestamp'].dt.hour

        # map the hour to a value between 0-2*pi
        hour = hour.map(lambda x: x / 24 * 2 * np.pi)
        logger.debug('------ Mapped hour to radians')
        sin_hour = np.sin(hour)
        logger.debug('------ Took the sin of the hour')
        data['f_sin_hour'] = sin_hour
        logger.debug('------ Added sin hour to dataframe')
        return data
