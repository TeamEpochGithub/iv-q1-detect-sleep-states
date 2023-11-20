from dataclasses import dataclass

import pandas as pd
import gc
from tqdm import tqdm

from .feature_engineering import FE, FEException
from .. import data_info
from ..logger.logger import logger
from ..util.suncalc import get_position

_SUN_FEATURES: list[str] = ["azimuth", "altitude"]


@dataclass
class Sun(FE):
    """Add sun features to the data

    The following sun-related features can be added: "azimuth", "altitude".

    :param sun_features: the sun features to add
    """
    sun_features: list[str]

    def __post_init__(self) -> None:
        """Check if the sun features are supported"""
        if any(sun_feature not in _SUN_FEATURES for sun_feature in self.sun_features):
            logger.critical(f"Unknown sun features: {self.sun_features}")
            raise FEException(f"Unknown sun features: {self.sun_features}")

    def feature_engineering(self, data: dict) -> dict:
        """Add the selected sun columns.

        :param data: the original data with at least the "series_id", "window", and "timestamp" columns
        :return: the data with the selected sun data added as extra columns
        """

        # Group by series_id and window and if 4 in data['utc'] replace 0 with 4
        times = {}
        sun_data = {}
        for sid in tqdm(data.keys()):
            data[sid] = data[sid].groupby('window').apply(lambda x: self.fill_padding(x)).reset_index(drop=True)
            times[sid] = data[sid]['timestamp'] + pd.to_timedelta(data[sid]['utc'], unit='h')
            sun_data[sid] = pd.DataFrame(get_position(times[sid], data_info.longitude, data_info.latitude))
            sun_data[sid].columns = ['azimuth', 'altitude']
            gc.collect()
            # Concat features if they are in self.sun_features and rename them to f_
            if 'azimuth' in self.sun_features:
                data[sid]['azimuth'] = sun_data[sid]['azimuth']
                data[sid] = data[sid].rename(columns={'azimuth': 'f_azimuth'})
                del sun_data[sid]['azimuth']
                gc.collect()
            if 'altitude' in self.sun_features:
                data[sid]['altitude'] = sun_data[sid]['altitude']
                data[sid] = data[sid].rename(columns={'altitude': 'f_altitude'})
                del sun_data[sid]['altitude']
                gc.collect()

        return data

    @staticmethod
    def fill_padding(x: pd.DataFrame) -> pd.DataFrame:
        """Fill padding for UTC

        :param x: the data with at least the "utc" column
        :return: data with correctly padded utc
        """
        if 4 in x['utc']:
            x['utc'] = x['utc'].replace(0, 4)

        if 5 in x['utc']:
            x['utc'] = x['utc'].replace(0, 5)

        return x
