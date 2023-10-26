from collections.abc import Callable
from unittest import TestCase

from src.cv.cv import _get_scoring, CV
from src.score.compute_score import compute_score_full, compute_score_clean


class Test(TestCase):
    def test_get_scoring(self):
        scorer = _get_scoring(lambda x: x)
        self.assertIsInstance(scorer, Callable)

        scorer = _get_scoring("score_full")
        self.assertIsInstance(scorer, Callable)

        scorer = _get_scoring(["score_full", "score_clean"])
        self.assertIsInstance(scorer, list)

    def test_cv_init(self):
        cv = CV(splitter="group_k_fold", scoring="score_full", splitter_params={"n_splits": 5})
        self.assertEqual(cv.splitter.n_splits, 5)
        self.assertEqual(cv.splitter.__class__.__name__, "GroupKFold")

        cv = CV(splitter="group_shuffle_split", scoring=["score_full", "score_clean"],
                splitter_params={"n_splits": 5})
        self.assertEqual(cv.splitter.n_splits, 5)
        self.assertEqual(cv.splitter.__class__.__name__, "GroupShuffleSplit")
