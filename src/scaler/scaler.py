from __future__ import annotations

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler, PowerTransformer, \
    QuantileTransformer, Normalizer

from ..logger.logger import logger


class Scaler:
    """This is a wrapper class for sklearn scalers

    It can be used to fit and transform data with a scaler.
    """

    def __init__(self, kind: str, **kwargs: dict) -> None:
        """Initialize the scaler.

        :param kind: the kind of scaler to use
        :param kwargs: the parameters of the scaler
        """
        self.kind = kind

        match kind:
            case "standard":
                self.scaler = StandardScaler(**kwargs)
            case "minmax":
                self.scaler = MinMaxScaler(**kwargs)
            case "maxabs":
                self.scaler = MaxAbsScaler(**kwargs)
            case "robust":
                self.scaler = RobustScaler(**kwargs)
            case "powertransformer":
                self.scaler = PowerTransformer(**kwargs)
            case "quantiletransformer":
                self.scaler = QuantileTransformer(**kwargs)
            case "normalizer":
                self.scaler = Normalizer(**kwargs)
            case "none":
                self.scaler = None
            case _:
                logger.critical("Unknown scaler %s", kind)
                raise ScalerException("Unknown scaler %s", kind)

    def save(self, path: str) -> None:
        """Save the scaler to a file.

        It uses joblib to save the scaler to a file.
        If the scaler is None, do nothing.

        :param path: the path to save the scaler to
        """
        if self.scaler is None:
            logger.info("--- No scaler used.")
            return

        joblib.dump(self.scaler, path)
        logger.info("--- Scaler saved to: " + path)

    def load(self, path: str) -> None:
        """Load the scaler from a file.

        It uses joblib to load the scaler from a file.
        If the scaler is None, do nothing.

        :param path: the path to load the scaler from
        """
        if self.scaler is None:
            logger.info("--- No scaler used.")
            return

        self.scaler: BaseEstimator = joblib.load(path)
        logger.info("--- Scaler loaded from: " + path)

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the scaler to the data.

        :param df: the data to fit the scaler to
        """
        if self.scaler is None:
            return

        feature_cols = [col for col in df.columns if col.startswith('f_')]
        data_to_standardize = df[['enmo', 'anglez'] + feature_cols]
        self.scaler.fit(data_to_standardize)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data with the scaler.

        :param df: the data to transform
        :return: the transformed data
        """
        if self.scaler is None:
            return df

        feature_cols = [col for col in df.columns if col.startswith('f_')]
        data_to_standardize = df[['enmo', 'anglez'] + feature_cols]
        standardized_data = self.scaler.transform(data_to_standardize)

        # Create a copy of the DataFrame before modifying it
        df = df.copy()
        df.loc[:, list(data_to_standardize.columns)] = standardized_data

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the scaler to the data and transform the data with the scaler.

        :param df: the data to fit the scaler to and transform
        :return: the transformed data
        """
        if self.scaler is None:
            return df

        feature_cols = [col for col in df.columns if col.startswith('f_')]
        data_to_standardize = df[['enmo', 'anglez'] + feature_cols]
        standardized_data = self.scaler.fit_transform(data_to_standardize)

        # Create a copy of the DataFrame before modifying it
        df = df.copy()
        df.loc[:, list(data_to_standardize.columns)] = standardized_data

        return df


class ScalerException(Exception):
    """
    Exception class for scalers.
    """
    pass
