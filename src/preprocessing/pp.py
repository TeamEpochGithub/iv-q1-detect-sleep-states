import pandas as pd

from ..logger.logger import logger


class PP:
    """
    Base class for preprocessing (PP) steps.
    All child classes should implement the preprocess method.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the preprocessing step.
        :param use_pandas: whether to use Pandas or Polars dataframes. Default is True.
        """
        self.use_pandas = kwargs['use_pandas'] if 'use_pandas' in kwargs else True

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the preprocessing step.
        :param data: the data to preprocess
        :return: the preprocessed data
        """
        return self.preprocess(data)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data. This method should be overridden by the child class.
        :param data: the data to preprocess
        :return: the preprocessed data
        """
        logger.critical("Preprocess method not implemented. Did you forget to override it?")
        raise PPException("Preprocess method not implemented. Did you forget to override it?")


class PPException(Exception):
    """
    Exception class for preprocessing steps.
    """
    pass
