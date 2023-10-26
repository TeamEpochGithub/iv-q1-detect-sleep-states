from collections.abc import Callable

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, GroupShuffleSplit, LeaveOneGroupOut, \
    LeavePGroupsOut, PredefinedSplit, KFold, LeaveOneOut, LeavePOut

from ..logger.logger import logger
from ..models.model import Model
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
    "stratified_group_k_fold": StratifiedGroupKFold,
}

_SCORERS: dict[str, Callable[[...], float]] = {
    "score_full": lambda y_true, y_pred, **kwargs: compute_score_full(
        *(from_numpy_to_submission_format(y_true, y_pred, **kwargs))),
    "score_clean": lambda y_true, y_pred, **kwargs: compute_score_clean(
        *(from_numpy_to_submission_format(y_true, y_pred, **kwargs)))
}


def _get_scoring(scoring: str | Callable[[...], float] | list[str | Callable[[...], float]]) \
        -> Callable[[...], float] | list[Callable[[...], float]]:
    """Get the scoring method(s)

    The input can be either a string, a callable, or a list of strings and/or callables.
    If it's a string, it will look up the scoring method in the _SCORERS dictionary.
    If it's a callable, it will return that scoring method directly.
    If it's a list, it will recursively call this function of each scoring method.

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

        :param splitter: the splitter used to split the data. See [README.md](../README.md) for all options.
        :param splitter_params: parameters for the splitters.
        See the [sklearn documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) for the parameters that each splitter needs.
        """
        try:
            self.splitter = _SPLITTERS[splitter](**splitter_params)
        except KeyError:
            logger.critical("Unknown CV splitter %s", splitter)
            raise CVException("Unknown CV splitter %s", splitter)

        self.scoring = _get_scoring(scoring)

    def cross_validate(self, model: Model, data: np.ndarray, labels: np.ndarray, groups: np.ndarray = None,
                       scoring_params: dict = {}) -> np.ndarray:
        """Evaluate the model using the CV method

        Run the cross-validation as specified in the config.
        The data is split into train and test sets using the splitter.
        The model is trained on the train set and evaluated on the test set.
        The scores of all folds for each scorer is returned.

        param model: the model to evaluate with methods `train` and `pred`
        :param data: the data to fit of shape (X_train_test[0], window_size, n_features)
        :param labels: the labels of shape (X_train_test[0], window_size, features)
        :param groups: the group labels used while splitting the data of shape (size, ) or None for no grouping
        :param scoring_params: the parameters for the scoring function(s)
        :return: the scores of all folds of shape (n_splits, n_scorers)
        """
        scores = []

        # Split the data in folds with train and test sets
        for i, (train_idx, test_idx) in enumerate(self.splitter.split(data, labels, groups)):
            model.reset_optimizer()

            X_train, X_test = data[train_idx], data[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            model.train(X_train, X_test, y_train, y_test)
            y_pred: np.array = model.pred(X_test)

            # Compute the score for each scorer
            if isinstance(self.scoring, list):
                score = [scorer(y_test, y_pred, test_idx_cv=test_idx, **scoring_params) for scorer in self.scoring]
            else:
                score = self.scoring(y_test, y_pred, test_idx_cv=test_idx, **scoring_params)
            scores.append(score)

        return np.array(scores)


class CVException(Exception):
    """Exception raised for errors in the CV."""
    pass
