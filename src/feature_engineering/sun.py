import pandas as pd
from tqdm import tqdm

from .feature_engineering import FE, FEException
from .. import data_info
from ..util.suncalc import get_position

_SUN_FEATURES: list[str] = ["azimuth", "altitude"]


class Sun(FE):
    """Add sun features to the data

    The following sun-related features can be added: "azimuth", "altitude".
    """

    def __init__(self, sun_features: str | list[str], **kwargs: dict) -> None:
        """Initialize the Time class

        :param sun_features: the time features to add
        """
        super().__init__(**kwargs | {"kind": "sun"})

        if isinstance(sun_features, list):
            self.sun_features = sun_features
        else:
            self.sun_features = [sun_features]

        if any(sun_feature not in _SUN_FEATURES for sun_feature in self.sun_features):
            raise FEException(f"Unknown sun features: {sun_features}")

    def __repr__(self) -> str:
        """Return a string representation of a Time object"""
        return f"{self.__class__.__name__}(sun_features={self.sun_features})"

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add the selected time columns.

        :param data: the original data
        :return: the data with the selected time data added
        """

        # Group by series_id and window and if 4 in data['utc'] replace 0 with 4
        tqdm.pandas()
        data = data.groupby(['series_id', 'window']).progress_apply(lambda x: self.fill_padding(x)).reset_index(drop=True)

        times = data['timestamp'] + pd.to_timedelta(data['utc'], unit='h')

        sunData = pd.DataFrame(get_position(times, data_info.longitude, data_info.latitude))
        sunData.columns = ['azimuth', 'altitude']
        # Concat features if they are in self.sun_features and rename them to f_
        if 'azimuth' in self.sun_features:
            data = pd.concat([data, sunData['azimuth']], axis=1)
            data = data.rename(columns={'azimuth': 'f_azimuth'})
        if 'altitude' in self.sun_features:
            data = pd.concat([data, sunData['altitude']], axis=1)
            data = data.rename(columns={'altitude': 'f_altitude'})

        return data

    def fill_padding(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Fill padding for utc, in a future commit
        :param x: series_id and window
        :return: data with correctly padded utc
        """
        if 4 in x['utc']:
            x['utc'] = x['utc'].replace(0, 4)

        if 5 in x['utc']:
            x['utc'] = x['utc'].replace(0, 5)

        return x
