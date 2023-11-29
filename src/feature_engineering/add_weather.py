from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TypeVar

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.feature_engineering.feature_engineering import FE

WEATHER_CONDITION_CODES: Final[list[str]] = [
    'null', 'Clear', 'Fair', 'Cloudy', 'Overcast', 'Fog', 'Freezing Fog', 'Light Rain', 'Rain', 'Heavy Rain',
    'Freezing Rain', 'Heavy Freezing Rain', 'Sleet', 'Heavy Sleet', 'Light Snowfall', 'Snowfall', 'Heavy Snowfall',
    'Rain Shower', 'Heavy Rain Shower', 'Sleet Shower', 'Heavy Sleet Shower', 'Snow Shower', 'Heavy Snow Shower',
    'Lightning', 'Hail', 'Thunderstorm', 'Heavy Thunderstorm', 'Storm'
]

WEATHER_FEATURES_INPUT: Final[set[str]] = {'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres',
                                           'tsun', 'coco'}

WEATHER_DATA_PATH: Final[Path] = Path(__file__).resolve().parent.parent / "external" / "weather.csv"

# Type of the key of the data dictionary
T = TypeVar('T')


@dataclass
class AddWeather(FE):
    """Add weather data to the data.

    Make sure to download the weather data first using weather_data_downloader/main.py,
    since we cannot retrieve that data on Kaggle.

    The following features are supported: 'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco'
    For more information about the weather features, see https://dev.meteostat.net/python/hourly.html#data-structure

    The path to the weather data is hardcoded to ../../external/weather.csv because of issues with submitting to Kaggle.

    :param weather_features: the weather features to add
    """
    weather_features: Iterable[str]

    _weather_features_out: list[str] = field(init=False, default_factory=list, repr=False, compare=False)
    _weather_data: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check if the weather features are supported and if the weather.csv file exists."""
        assert all(weather_feature in WEATHER_FEATURES_INPUT for weather_feature in self.weather_features), \
            f"Unknown {self.weather_features = }"

        assert WEATHER_DATA_PATH.exists(), (f"Weather data file {WEATHER_DATA_PATH} does not exist. "
                                            f"Make sure to download the weather data first using weather_data_downloader/main.py. "
                                            f"If you're running this on Kaggle, make sure to add the weather data as a dataset. ")

        self._weather_features_out: list[str] = [f"f_{weather_feature}" for weather_feature in self.weather_features]
        if 'f_coco' in self._weather_features_out:
            self._weather_features_out.remove('f_coco')
            self._weather_features_out += [f'f_coco_{weather_condition_code}' for weather_condition_code in
                                           WEATHER_CONDITION_CODES]

    def run(self, data: MutableMapping[T, pd.DataFrame]) -> MutableMapping[T, pd.DataFrame]:
        """Read the weather data, then run the actual feature engineering step.

        :param data: the data to process with timestamp columns
        :return: the processed data with the columns of the desired weather features
        """
        self._weather_data = pd.read_csv(WEATHER_DATA_PATH)
        self._weather_data['timestamp'] = pd.to_datetime(self._weather_data['time'], utc=True, errors='raise')
        self._weather_data.drop(columns=['time'], inplace=True)

        assert self._weather_data['timestamp'].dtype == pd.DatetimeTZDtype(tz='UTC'), \
            f"The weather data timestamp column (dtype={self._weather_data['timestamp'].dtype}) is not of dtype 'datetime64[ns, UTC]'!"

        # Set all float columns to float32
        for column in self._weather_data.columns:
            if self._weather_data[column].dtype == np.float64:
                self._weather_data[column] = self._weather_data[column].astype(np.float32)

        return self.feature_engineering(data)

    def feature_engineering(self, data: MutableMapping[T, pd.DataFrame]) -> MutableMapping[T, pd.DataFrame]:
        """Add the weather data to the data.

        Since the weather data is hourly, we use the nearest timestamp for each row.

        :param data: the data to process with timestamp columns
        :return: the processed data with the columns of the desired weather features
        """
        assert 'timestamp' in next(iter(data.values())).columns, "The timestamp column is missing!"

        for sid in tqdm(data.keys()):
            data[sid] = pd.merge_asof(data[sid], self._weather_data[
                self._weather_data.columns.intersection(set(['timestamp'] + self._weather_features_out))],
                                      on='timestamp', direction='nearest')

        return data
