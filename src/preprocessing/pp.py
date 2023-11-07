from __future__ import annotations

import pandas as pd

from ..logger.logger import logger


class PP:
    """
    Base class for preprocessing (PP) steps.
    All child classes should implement the preprocess method.
    """

    def __init__(self, kind: str, use_pandas: bool = True, **kwargs) -> None:
        """
        Initialize the preprocessing step.

        :param kind: the kind of the preprocessing step
        :param use_pandas: whether to use Pandas or Polars dataframes. Default is True.
        """
        self.kind = kind
        self.use_pandas = use_pandas

    @staticmethod
    def from_config_single(config: dict) -> PP:
        """Parse the config of a single preprocessing step and return the preprocessing step.

        :param config: the config of a single preprocessing step
        :return: the specific preprocessing step
        """
        kind: str = config["kind"]

        match kind:
            case "mem_reduce":
                from .mem_reduce import MemReduce
                return MemReduce(**config)
            case "add_noise":
                from .add_noise import AddNoise
                return AddNoise(**config)
            case "similarity_nan":
                from .similarity_nan import SimilarityNan
                return SimilarityNan(**config)
            case "add_regression_labels":
                from .add_regression_labels import AddRegressionLabels
                return AddRegressionLabels(**config)
            case "add_segmentation_labels":
                from .add_segmentation_labels import AddSegmentationLabels
                return AddSegmentationLabels(**config)
            case "add_event_labels":
                from .add_event_labels import AddEventLabels
                return AddEventLabels(**config)
            case "add_state_labels":
                from .add_state_labels import AddStateLabels
                return AddStateLabels(**config)
            case "remove_unlabeled":
                from .remove_unlabeled import RemoveUnlabeled
                return RemoveUnlabeled(**config)
            case "split_windows":
                from .split_windows import SplitWindows
                return SplitWindows(**config)
            case _:
                logger.critical(f"Unknown preprocessing step: {kind}")
                raise PPException(f"Unknown preprocessing step: {kind}")

    @staticmethod
    def from_config(config_list: list[dict], training: bool) -> list[PP]:
        """Parse the config list of preprocessing steps and return the preprocessing steps.

        It also filters out the steps that are only for training.

        :param config_list: the list of preprocessing steps
        :param training: whether the pp steps are for training or not
        :return: the list of preprocessing steps
        """
        training_only: list[str] = ["add_state_labels", "remove_unlabeled", "truncate",
                                    "add_event_labels", "add_regression_labels",
                                    "add_segmentation_labels"]
        pp_steps: list[dict] = [pp_step for pp_step in config_list if
                                training or pp_step["kind"] not in training_only]

        return [PP.from_config_single(pp_step) for pp_step in pp_steps]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the preprocessing step.

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
