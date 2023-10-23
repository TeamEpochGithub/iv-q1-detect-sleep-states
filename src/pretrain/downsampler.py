import numpy as np
import pandas as pd

from src.logger.logger import logger

_METHODS: dict[str, callable] = {
    "mean": np.mean,
    "median": np.median,
    "max": np.max,
    "min": np.min,
    "std": np.std,
    "var": np.var,
}


class Downsampler:
    """
    Downsampler class to downsample the data.
    """

    def __init__(self, factor: int, features: list[str], methods: list[str], standard: str):
        """Initialize the downsampler.

        :param factor: the factor to downsample by
        :param features: the features to downsample
        :param methods: the methods to downsample with
        :param standard: standard downsampling method
        """
        self.factor = factor
        self.features = features
        self.methods = methods
        self.standard = standard

    def downsampleY(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Downsample the y data that contains the labels and always use median for this, since it works best.
        :param y: the y data to downsample
        :return: the downsampled y data
        """

        y_names = y.columns

        # Convert to numpy array
        y = y.to_numpy()

        # Check if the data can be downsampled and see if it is divisible by the factor
        if y.shape[0] % self.factor != 0:
            logger.critical("Data cannot be downsampled by factor %s", str(self.factor))
            raise Exception("Data cannot be downsampled by factor " + str(self.factor))

        Y_new = []

        for i in self.y.shape[1]:
            y_downsampled = np.median(y[:, i].reshape(-1, self.factor), axis=1)
            Y_new.append(y_downsampled)

        y = np.array(Y_new).T

        # Convert to a dataframe with y_names
        y = pd.DataFrame(y, columns=y_names)
        return y

    def downsampleX(self, X: pd.DataFrame) -> pd.DataFrame:
        """Downsample the X data that contains the features

        :param X: the X data to downsample
        :return: the downsampled X data
        """

        # Get integers for the features and check if they are in X
        features_iloc = []
        for feature in self.features:
            if feature not in X.columns:
                logger.critical("Feature %s not in X", feature)
                raise Exception("Feature " + feature + " not in X")
            features_iloc.append(X.columns.get_loc(feature))

        # Convert to numpy array
        X = X.to_numpy()

        # Check if the data can be downsampled and see if it is divisible by the factor
        if X.shape[0] % self.factor != 0:
            logger.critical("Data cannot be downsampled by factor %s", str(self.factor))
            raise Exception("Data cannot be downsampled by factor " + str(self.factor))
        # Print the shape of the data

        logger.info("Shape of X before downsampling: " + str(X.shape))

        new_X = []
        new_X_names = []
        # Downsample the data and add names
        for i in features_iloc:
            for method in self.methods:
                try:
                    f_downsampled = _METHODS[method](X[:, i].reshape(-1, self.factor), axis=1)
                    new_X.append(f_downsampled)
                    new_X_names.append(X.columns[i] + "_" + method)
                except:
                    logger.critical("Unknown downsampling method %s", method)
                    raise Exception("Unknown downsampling method " + method)

        # Downsample all other features with the standard method
        for i in range(X.shape[1]):
            if i not in features_iloc:
                try:
                    f_downsampled = _METHODS[method](X[:, i].reshape(-1, self.factor), axis=1)
                    new_X.append(f_downsampled)
                    new_X_names.append(X.columns[i] + "_" + method)
                except:
                    logger.critical("Unknown downsampling method %s", self.standard)
                    raise Exception("Unknown downsampling method " + self.standard)

        # Convert to numpy array
        new_X = np.array(new_X).T

        # Convert to a dataframe with new_X_names
        new_X = pd.DataFrame(new_X, columns=new_X_names)
        return new_X
