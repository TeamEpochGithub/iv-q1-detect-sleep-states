from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..logger.logger import logger


class PP(ABC):
    """
    Base class for preprocessing (PP) steps.
    All child classes should implement the preprocess method.
    """

    @staticmethod
    def from_config_single(config: dict) -> PP:
        """Parse the config of a single preprocessing step and return the preprocessing step.

        :param config: the config of a single preprocessing step
        :return: the specific preprocessing step
        """
        kind: str = config.pop("kind", None)

        if kind is None:
            logger.critical("No kind specified for preprocessing step")
            raise PPException("No kind specified for preprocessing step")

        # Import the preprocessing steps here to avoid circular imports
        from .mem_reduce import MemReduce
        from .add_noise import AddNoise
        from .similarity_nan import SimilarityNan
        from .add_regression_labels import AddRegressionLabels
        from .add_segmentation_labels import AddSegmentationLabels
        from .add_event_labels import AddEventLabels
        from .add_state_labels import AddStateLabels
        from .remove_unlabeled import RemoveUnlabeled
        from .split_windows import SplitWindows

        _PREPROCESSING_KINDS: dict[str] = {
            "mem_reduce": MemReduce,
            "add_noise": AddNoise,
            "similarity_nan": SimilarityNan,
            "add_regression_labels": AddRegressionLabels,
            "add_segmentation_labels": AddSegmentationLabels,
            "add_event_labels": AddEventLabels,
            "add_state_labels": AddStateLabels,
            "remove_unlabeled": RemoveUnlabeled,
            "split_windows": SplitWindows
        }

        try:
            return _PREPROCESSING_KINDS[kind](**config)
        except KeyError:
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

    @abstractmethod
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
