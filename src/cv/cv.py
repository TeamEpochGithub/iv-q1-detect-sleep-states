from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, GroupShuffleSplit, LeaveOneGroupOut, \
    LeavePGroupsOut, PredefinedSplit, KFold, LeaveOneOut, LeavePOut, RepeatedKFold, RepeatedStratifiedKFold, \
    ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit

from src.models.event_model import EventModel
from .. import data_info
from ..logger.logger import logger
from ..score.compute_score import compute_score_full, compute_score_clean, from_numpy_to_submission_format

_SPLITTERS: dict[str] = {
    "group_k_fold": GroupKFold,
    "group_shuffle_split": GroupShuffleSplit,
    "k_fold": KFold,
    "leave_one_group_out": LeaveOneGroupOut,
    "leave_p_group_out": LeavePGroupsOut,
    "leave_one_out": LeaveOneOut,
    "leave_p_out": LeavePOut,
    "predefined_split": PredefinedSplit,
    "repeated_k_fold": RepeatedKFold,
    "repeated_stratified_k_fold": RepeatedStratifiedKFold,
    "shuffle_split": ShuffleSplit,
    "stratified_k_fold": StratifiedKFold,
    "stratified_shuffle_split": StratifiedShuffleSplit,
    "stratified_group_k_fold": StratifiedGroupKFold,
    "time_series_split": TimeSeriesSplit
}

_SCORERS: dict[str, Callable[[...], float]] = {
    "score_full": lambda info, y_pred: compute_score_full(
        *(from_numpy_to_submission_format(info, y_pred))),
    "score_clean": lambda info, y_pred: compute_score_clean(
        *(from_numpy_to_submission_format(info, y_pred)))
}


def _get_scoring(scoring: str | Callable[[...], float] | list[str | Callable[[...], float]]) \
        -> Callable[[...], float] | list[Callable[[...], float]]:
    """Get the scoring method(s)

    The input can be either a string, a callable, or a list of strings and/or callables.

    This method makes it easy to configure the scoring method form the config,
    but with the ability to temporarily replace it with a custom method for testing purposes.

    All scoring methods must have the signature `scoring(y_true: np.ndarray, y_pred: np.ndarray, **kwargs: dict) -> float`.

    :param scoring: the scoring methods as a string, callable, or list of those
    :return: the scoring methods as a callable or list of callables
    """
    match scoring:
        case str():
            try:
                return _SCORERS[scoring]
            except KeyError:
                logger.critical("Unknown scoring method %s", scoring)
                raise CVException("Unknown scoring method %s", scoring)
        case list():
            return [_get_scoring(s) for s in scoring]
        case None:
            logger.critical("Scoring method not specified")
            raise CVException("Scoring method not specified")
        case Callable():
            return scoring


class CV:
    def __init__(self, splitter: str, splitter_params: dict,
                 scoring: str | Callable[[...], float] | list[str | Callable[[...], float]]) -> None:
        """Initialize the CV object

        :param splitter: the splitter used to split the data. See [README.md](./README.md) for all options.
        :param splitter_params: parameters for the splitters.
        See the [sklearn documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) for the parameters that each splitter needs.
        """
        try:
            self.splitter = _SPLITTERS[splitter](**splitter_params)
        except KeyError:
            logger.critical("Unknown CV splitter %s", splitter)
            raise CVException("Unknown CV splitter %s", splitter)
        self.scoring = _get_scoring(scoring)

    def cross_validate(self, model: EventModel, data: np.ndarray, labels: np.ndarray, train_window_info: pd.DataFrame,
                       groups: np.ndarray = None, ) -> np.ndarray:
        """Evaluate the model using the CV method

        Run the cross-validation as specified in the config.
        The data is split into train and val sets using the splitter.
        The model is trained on the train set and evaluated on the val set.
        The scores of all folds for each scorer is returned.

        :param model: the model to evaluate with methods `train` and `pred`
        :param data: the data to fit of shape (X_train[0], window_size, n_features)
        :param labels: the labels of shape (X_train[0], window_size, features)
        :param train_window_info: the window info for the training split
        :param groups: the group labels used while splitting the data of shape (size, ) or None for no grouping
        :return: the scores of all folds of shape (n_splits, n_scorers)
        """
        scores = []

        # Split the data in folds with train and validation sets
        for i, (train_idx, validate_idx) in enumerate(self.splitter.split(data, labels, groups)):

            # Set substage to the current fold
            data_info.substage = "Fold " + str(i)
            logger.info("Fold %d", i)

            X_train, X_validate = data[train_idx], data[validate_idx]
            y_train, y_validate = labels[train_idx], labels[validate_idx]

            # Get the window info for scoring
            data_info.validate_window_info = train_window_info.iloc[validate_idx]

            model.train(X_train, X_validate, y_train, y_validate)
            y_pred: np.array = model.pred(X_validate, pred_with_cpu=False)

            # Reset weights, optimizer and scheduler for next fold
            model.reset_weights()
            model.reset_optimizer()
            model.reset_scheduler()

            # Compute the score for each scorer
            if isinstance(self.scoring, list):
                score = [scorer(data_info.validate_window_info, y_pred)
                         for scorer in self.scoring]
            else:
                score = self.scoring(data_info.validate_window_info, y_pred)

            scores.append(score)

            # If we are doing HPO and the score_full is lower than 0.1, stop the HPO
            if data_info.hpo and score[0] < 0.1:
                break

        return np.array(scores)


class CVException(Exception):
    """Exception raised for errors in the CV."""
    pass
