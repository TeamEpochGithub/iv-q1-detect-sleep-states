import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, KFold

from ..logger.logger import logger
from ..models.model import Model
from ..score.doscoring import compute_scores

_SPLITTERS: dict[str | KFold] = {
    "group_k_fold": GroupKFold,
    "stratified_group_k_fold": StratifiedGroupKFold,
}

_SCORERS: dict[str, callable] = {
    "score_full": lambda x, y: compute_scores(x, y)[0],
    "score_clean": lambda x, y: compute_scores(x, y)[1],
}


def _get_scoring(scoring: str | callable | list[str | callable]) -> callable | list[callable]:
    match scoring:
        case str():
            try:
                return _SCORERS[scoring]
            except KeyError:
                logger.critical("Unknown scoring method %s", scoring)
                raise CVException("Unknown scoring method %s", scoring)
        case callable():
            return scoring
        case list():
            return [_get_scoring(s) for s in scoring]


class CV:
    def __init__(self, splitter: str, scoring: str | callable | list[str | callable], **kwargs: dict) -> None:
        """Initialize the CV object

        :param splitter: the splitter used to split the data
        :param kwargs: the arguments for the splitter
        """

        try:
            self.splitter = _SPLITTERS[splitter](**kwargs)
        except KeyError:
            logger.critical("Unknown CV method %s", splitter)
            raise CVException("Unknown CV method %s", splitter)

        self.scoring = _get_scoring(scoring)

    def cross_validate(self, model: Model, data: pd.DataFrame, labels: pd.Series) -> float | list[float]:
        """Evaluate the model using the CV method

        Run the cross-validation as specified in the config.
        The data is split into train and test sets using the splitter.
        The model is trained on the train set and evaluated on the test set.
        The average score of all folds is returned.

        :param data: the data to fit
        :param labels: the labels
        :param model: the model to evaluate
        :return: the average score
        """
        scores: list = []
        for train_idx, test_idx in self.splitter.split(data, labels, self.data['series_id']):
            model.reset_optimizer()

            X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
            y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

            model.train(X_train, X_test, y_train, y_test)

            if isinstance(self.scoring, list):
                score = [scoring(X_test, y_test) for scoring in self.scoring]
            else:
                score = self.scoring(X_test, y_test)
            scores.append(score)

        # TODO Check axes
        return np.mean(scores)


class CVException(Exception):
    """Exception raised for errors in the CV."""
    pass
