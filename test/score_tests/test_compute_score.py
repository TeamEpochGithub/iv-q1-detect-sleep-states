from unittest import TestCase

import pandas as pd

from src.score.compute_score import verify_submission, ScoringException


class Test(TestCase):
    def test_verify_submission(self):
        submission = pd.DataFrame({
            'series_id': [0, 1, 1, 1, 1],
            'event': ['onset', 'onset', 'wakeup', 'onset', 'wakeup'],
        })
        verify_submission(submission)

    def test_verify_submission_fail(self):
        submission = pd.DataFrame({
            'series_id': [0, 1, 1, 1, 1],
            'event': ['onset', 'onset', 'wakeup', 'onset', 'onset'],
        })
        self.assertRaises(ScoringException, verify_submission, submission)

    # def test_compute_score_full(self):
    #     submission = pd.DataFrame({
    #         'series_id': ['a', 'a', 'a'],
    #         'event': ['wakeup', 'onset', 'wakeup'],
    #         'score': [1.0, 1.0, 1.0],
    #         'step': [10, 20, 40],
    #     })
    #     solution = pd.DataFrame({
    #         'series_id': ['a', 'a', 'a', 'a'],
    #         'event': ['onset', 'wakeup', 'onset', 'wakeup'],
    #         'step': [0, 10, 20, 30],
    #     })
    #
    #     result = compute_score_full(submission, solution)
    #
    #     # TODO Finish tests. I can't do it because I don't know how the scoring works.
    #     self.assertEqual(result, 0.6666666666666666)
