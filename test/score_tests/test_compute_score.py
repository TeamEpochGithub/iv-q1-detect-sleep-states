from unittest import TestCase

import pandas as pd

from src.score.compute_score import ScoringException, verify_cv


class Test(TestCase):
    def test_verify_submission(self):
        submission = pd.DataFrame({
            'series_id': [0, 1, 1, 1, 1],
            'event': ['onset', 'onset', 'wakeup', 'onset', 'wakeup'],
        })
        solution = pd.DataFrame({
            'series_id': [0, 1, 1, 1, 1],
            'event': ['onset', 'onset', 'wakeup', 'onset', 'wakeup'],
        })
        verify_cv(submission, solution)

    def test_verify_submission_fail(self):
        submission = pd.DataFrame({
            'series_id': [0, 1, 1, 1, 1],
            'event': ['onset', 'onset', 'wakeup', 'onset', 'onset'],
        })
        solution = pd.DataFrame({
            'series_id': [0, 1, 1, 1, 1],
            'event': ['onset', 'onset', 'wakeup', 'onset', 'wakeup'],
        })
        self.assertRaises(ScoringException, verify_cv, submission, solution)
