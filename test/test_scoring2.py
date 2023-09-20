from unittest import TestCase
from src.scoring import score


class Test(TestCase):
    def test_score(self):
        import pandas as pd

        column_names = {
            'series_id_column_name': 'series_id',
            'time_column_name': 'time',
            'event_column_name': 'event',
            'score_column_name': 'score',
        }

        tolerances = {'pass': [1.0]}
        solution = pd.DataFrame({
            'series_id': ['a', 'a', 'a', 'a'],
            'event': ['start', 'pass', 'pass', 'end'],
            'time': [0, 10, 20, 30],
        })
        submission = pd.DataFrame({
            'series_id': ['a', 'a', 'a'],
            'event': ['pass', 'pass', 'pass'],
            'score': [1.0, 1.0, 1.0],
            'time': [10, 20, 40],
        })

        result = score(solution, submission, tolerances, **column_names, use_scoring_intervals=False,
                       plot_precision_recall=False)

        self.assertEqual(result, 0.6666666666666666)
