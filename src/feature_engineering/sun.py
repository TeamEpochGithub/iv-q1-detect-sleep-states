import gc
from dataclasses import dataclass
from typing import Final, Literal, Iterable

import pandas as pd
from tqdm import tqdm

from .feature_engineering import FE
from .. import data_info
from ..util.suncalc import get_position

_SUN_FEATURES: Final[set[Literal['azimuth', 'altitude']]] = {'azimuth', 'altitude'}


@dataclass
class Sun(FE):
    """Add sun features to the data

    The following sun-related features can be added: 'azimuth', 'altitude'.

    :param sun_features: the sun features to add
    """
    sun_features: Iterable[Literal['azimuth', 'altitude']]

    def __post_init__(self) -> None:
        """Check if the sun features are supported"""
        assert all(sun_feature in _SUN_FEATURES for sun_feature in self.sun_features), \
            f"Unknown sun features: {self.sun_features}"

    def feature_engineering(self, data: dict) -> dict:
        """Add the selected sun columns.

        :param data: the original data with at least the 'series_id', 'window', and 'timestamp' columns
        :return: the data with the selected sun data added as extra columns
        """
        sun_data: dict[str, pd.DataFrame] = {}

        for sid in tqdm(data.keys()):
            sun_data[sid] = pd.DataFrame(get_position(data[sid]['timestamp'], data_info.longitude, data_info.latitude))
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
