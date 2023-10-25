from collections.abc import Callable

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, GroupShuffleSplit, LeaveOneGroupOut, \
    LeavePGroupsOut, PredefinedSplit, KFold, LeaveOneOut, LeavePOut

from ..logger.logger import logger
from ..models.model import Model
from ..score.compute_score import compute_score_full, compute_score_clean

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

_SCORERS: dict[str, Callable] = {
    "score_full": compute_score_full,
    "score_clean": compute_score_clean,
}


def _get_scoring(scoring: str | Callable | list[str | Callable]) -> Callable | list[Callable]:
    match scoring:
        # case Callable():
        #     return scoring
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
        case _:
            return scoring


class CV:
    def __init__(self, pred_with_cpu: bool, splitter: str, splitter_params: dict, **kwargs: dict) -> None:
        """Initialize the CV object

        :param splitter: the splitter used to split the data
        :param kwargs: the arguments for the splitter
        """
        self.pred_with_cpu = pred_with_cpu

        try:
            self.splitter = _SPLITTERS[splitter](**splitter_params)
        except KeyError:
            logger.critical("Unknown CV splitter %s", splitter)
            raise CVException("Unknown CV splitter %s", splitter)

    def cross_validate(self, model: Model, data: np.array, labels: np.array, groups: np.array = None, scoring: str | Callable | list[str | Callable] = None) -> np.array:
        """Evaluate the model using the CV method

        Run the cross-validation as specified in the config.
        The data is split into train and test sets using the splitter.
        The model is trained on the train set and evaluated on the test set.
        The average score of all folds is returned.

        :param data: the data to fit of shape (size, window_size, features)
        :param labels: the labels of shape (size, window_size, features)
        :param model: the model to evaluate
        :param groups: the groups labels used while splitting the data of shape (size, ) or None for no grouping
        :return: the scores of shape (n_folds, n_metrics)
        """
        scores = []
        scoring_func = _get_scoring(scoring)

        for i, (train_idx_cv, test_idx_cv) in enumerate(self.splitter.split(data, labels, groups)):
            model.reset_optimizer()

            X_train_cv, X_test_cv = data[train_idx_cv], data[test_idx_cv]
            y_train_cv, y_test_cv = labels[train_idx_cv], labels[test_idx_cv]

            model.train(X_train_cv, X_test_cv, y_train_cv, y_test_cv)
            y_pred_cv: np.array = model.pred(X_test_cv, with_cpu=self.pred_with_cpu)

            if isinstance(scoring_func, list):
                score = [scorer(y_test_cv, y_pred_cv, test_idx_cv=test_idx_cv) for scorer in scoring]
            else:
                score = scoring_func(y_test_cv, y_pred_cv, test_idx_cv=test_idx_cv)
            scores.append(score)

        return np.array(scores)


class CVException(Exception):
    """Exception raised for errors in the CV."""
    pass
