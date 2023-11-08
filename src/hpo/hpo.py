from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from sklearn.base import BaseEstimator

from ..logger.logger import logger


class HPO(ABC):
    """Base class for hyperparameter optimization (HPO) methods."""

    @staticmethod
    def from_config(config: dict) -> HPO | None:
        """Parse the config of a single hyperparameter optimization method and return the hyperparameter optimization method.

        :param config: the config of a single hyperparameter optimization method
        :return: the specific hyperparameter optimization method
        """
        kind: str | None = config.pop("kind", None)
        apply: bool = config.pop("apply", True)

        if kind is None or apply is False:
            return None

        from .wandb_sweeps import WandBSweeps
        _HPO_KINDS: dict[str, type(HPO)] = {
            "wandb_sweeps": WandBSweeps,
        }

        try:
            return _HPO_KINDS[kind](**config)
        except KeyError:
            logger.critical(f"Unknown HPO method: {kind}")
            raise HPOException(f"Unknown HPO method: {kind}")

    @abstractmethod
    def optimize(self, to_optimize: Callable | BaseEstimator) -> None:
        """Optimize the hyperparameters for a single model.

        :param to_optimize: the function or pipeline that runs the preprocessing, feature engineering, pretrain, training, and cross validation.
        """
        pass


class HPOException(Exception):
    """Exception for unknown HPO method."""
    pass
