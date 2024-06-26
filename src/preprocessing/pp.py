from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Final, TypeAlias

import pandas as pd

_TRAINING_ONLY: Final[list[str]] = ['add_state_labels', 'remove_unlabeled', 'truncate', 'add_event_labels',
                                    'add_regression_labels', 'add_segmentation_labels']

JSON: TypeAlias = None | int | str | bool | list['JSON'] | dict[str, 'JSON']


class PP(ABC):
    """
    Base class for preprocessing (PP) steps.
    All child classes should implement the preprocess method.
    """

    @staticmethod
    def from_config_single(config: dict[str, JSON]) -> PP:
        """Parse the config of a single preprocessing step and return the preprocessing step.

        :param config: the config of a single preprocessing step
        :return: the specific preprocessing step
        """
        config_copy = copy.deepcopy(config)
        kind = config_copy.pop('kind', None)

        assert kind is not None, "No kind specified for preprocessing step"

        # Import the preprocessing steps here to avoid circular imports
        from .mem_reduce import MemReduce
        from .add_noise import AddNoise
        from .similarity_nan import SimilarityNan
        from .add_segmentation_labels import AddSegmentationLabels
        from .add_event_labels import AddEventLabels
        from .add_state_labels import AddStateLabels
        from .split_windows import SplitWindows

        _PREPROCESSING_KINDS: Final[dict[str | type[PP]]] = {
            'mem_reduce': MemReduce,
            'add_noise': AddNoise,
            'similarity_nan': SimilarityNan,
            'add_segmentation_labels': AddSegmentationLabels,
            'add_event_labels': AddEventLabels,
            'add_state_labels': AddStateLabels,
            'split_windows': SplitWindows
        }

        assert kind in _PREPROCESSING_KINDS, f"Unknown preprocessing step: {kind=}"
        return _PREPROCESSING_KINDS[kind](**config_copy)

    @staticmethod
    def from_config(config_list: list[dict[str, JSON]], training: bool) -> list[PP]:
        """Parse the config list of preprocessing steps and return the preprocessing steps.

        It also filters out the steps that are only for training.

        :param config_list: the list of preprocessing steps
        :param training: whether the pp steps are for training or not
        :return: the list of preprocessing steps
        """
        pp_steps: list[dict[str, JSON]] = [pp_step for pp_step in config_list if
                                           training or pp_step['kind'] not in _TRAINING_ONLY]

        return [PP.from_config_single(pp_step) for pp_step in pp_steps]

    def run(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Run the preprocessing step.

        :param data: the data to preprocess
        :return: the preprocessed data
        """
        return self.preprocess(data)

    @abstractmethod
    def preprocess(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Preprocess the data. This method should be overridden by the child class.
        :param data: the data to preprocess
        :return: the preprocessed data
        """
        pass


class PPException(Exception):
    """
    Exception class for preprocessing steps.
    """
    pass
