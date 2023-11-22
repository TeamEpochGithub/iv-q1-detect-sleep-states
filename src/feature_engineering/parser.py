import gc
from dataclasses import field

from scipy.signal import savgol_filter
import numpy as np
import pandas as pd

from .feature_engineering import FE


class Parser(FE):
    """Parser class for feature engineering. Computes multiple features based on strings. Example:
    anglez_diff_abs_clip_10_rolling_median_100.
    It progressively builds up the feature and stores intermediate steps for reuse.
    When done, it will only add the features as specified to the dataframe, and not the intermediates.
    This class will be able to reproduce some of the same features as the original classes.

    Feature options:
    - <prev>_diff: compute the diff (slope) of the feature
    - <prev>_abs: compute the absolute value of the feature
    - <prev>_<stat>_<size>: compute the rolling statistic of the feature (either a pandas or numpy function)
    - <prev>_clip_<size>: clip the feature between -size and size
    - <prev>_savgol_<size>: compute the savgol filter of the feature with size, see scipy.signal.savgol_filter

    <prev> can be an original feature, or it will be recursively computed with the same scheme.
    """

    window_sizes: list[int] = field(default_factory=lambda: [100])

    def __init__(self, feats: list[str]):
        super().__init__()
        self.feats = feats
        self.available_lookup = dict()

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for the parser class. This will add the features to the dataframe.
        """

        # add the existing columns to the available lookup
        for col in data.columns:
            self.available_lookup[col] = data[col]

        # loop over the features to add each one
        for feat in self.feats:
            self.add_feature(feat)

        # only keep the features we want
        for feat in self.feats:
            data[feat] = self.available_lookup[feat]

        # clean up after itself
        del self.available_lookup
        gc.collect()

        return data

    def add_feature_and_save(self, feat) -> pd.Series:
        """Wrapper around add_feature, that saves the result in the lookup dict"""
        res = self.add_feature(feat)
        self.available_lookup[feat] = res
        return res

    def add_feature(self, feat) -> pd.Series:
        """Compute the feature recursively by parsing the name"""

        # already available
        if feat in self.available_lookup:
            return self.available_lookup[feat]

        # compute diff (slope)
        if feat.endswith("_diff"):
            prev = self.add_feature_and_save(feat[:-5])
            return prev.diff().bfill()

        # compute absolute value
        if feat.endswith("_abs"):
            prev = self.add_feature_and_save(feat[:-4])
            return prev.abs()

        # we now know it should be a feature as <prev>_<func>_<size>
        splits = feat.split("_")
        func = splits[-2]
        size = int(splits[-1])
        prev = self.add_feature_and_save("_".join(splits[:-2]))

        # use a pandas rolling statistic function
        if func in ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt']:
            return prev.rolling(size, center=True).agg(func).bfill().ffill()

        if func == 'range':
            return prev.rolling(size, center=True).agg(np.ptp).bfill().ffill().ffill()

        if func == 'clip':
            return prev.clip(-size, size)

        if func == "savgol":
            return pd.Series(savgol_filter(prev, size, 3))

        raise ValueError(f"Unknown feature: {feat}")