from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd
from tqdm import tqdm

from src.feature_engineering.feature_engineering import FE

_WEATHER_FEATURES: Final[set[str]] = {'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt',
                                      'pres', 'tsun', 'coco'}


@dataclass
class AddWeather(FE):
    """Add weather data to the data.

    Make sure to download the weather data first using weather_data_downloader/main.py,
    since we cannot retrieve that data on Kaggle.

    The following features are supported: 'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco'
    For more information about the weather features, see https://dev.meteostat.net/python/hourly.html#data-structure

    :param weather_data_path: the path to the weather data
    :param weather_features: the weather features to add
    """
    weather_data_path: str
    weather_features: list[str]

    _weather_data: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check if the weather features are supported"""
        assert all(weather_feature in _WEATHER_FEATURES for weather_feature in self.weather_features), \
            f"Unknown {self.weather_features = }"

    def run(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Read the weather data

        :param data: the data to process with timestamp columns
        :return: the processed data with the columns in weather_features
        """
        path: Path = Path(self.weather_data_path)
        assert path.exists(), (f"Weather data file {path} does not exist. "
                               f"Make sure to download the weather data first using weather_data_downloader/main.py.")

        self._weather_data = pd.read_csv(path)
        self._weather_data['timestamp'] = pd.to_datetime(self._weather_data['time'], utc=True, errors='raise')
        self._weather_data.drop(columns=['time'], inplace=True)

        assert self._weather_data['timestamp'].dtype == pd.DatetimeTZDtype(tz='UTC'), \
            f"The weather data timestamp column (dtype={self._weather_data['timestamp'].dtype}) is not of dtype 'datetime64[ns, UTC]'!"
        return self.feature_engineering(data)

    def feature_engineering(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Add the weather data to the data.

        Since the weather data is hourly, we use the nearest timestamp for each row.

        :param data: the data to process with timestamp columns
        :return: the processed data with the columns in weather_features
        """
        assert 'timestamp' in next(iter(data.values())).columns, "The timestamp column is missing"

        for sid, _ in tqdm(data.items()):
            data[sid] = pd.merge_asof(data[sid], self._weather_data[['timestamp'] + self.weather_features],
                                      on='timestamp', direction='nearest')
            data[sid].rename(columns={weather_feature: f'f_{weather_feature}' for weather_feature in self.weather_features}, inplace=True)

        return data
