from unittest import TestCase

import numpy as np
import pandas as pd

from src.preprocessing.add_state_labels import AddStateLabels


class TestAddStateLabels(TestCase):
    def test_repr(self) -> None:
        pp = AddStateLabels('./dummy_event_path', './dummy_id_encoding_path',
                            use_similarity_nan=True, fill_limit=10)
        self.assertEqual(
            "AddStateLabels(events_path='./dummy_event_path', id_encoding_path='./dummy_id_encoding_path', use_similarity_nan=True, fill_limit=10, nan_tolerance_window=1)",
            pp.__repr__())

    # def test_preprocess_one_night(self):
    #     # make a test data frame with two series of 10 steps
    #     data = pd.DataFrame({
    #         'series_id': [0] * 10 + [1] * 10,
    #     })

    #     # make a test events data frame with two events per series
    #     events = pd.DataFrame({
    #         'series_id': ['a', 'a', 'b', 'b'],
    #         'event': ['onset', 'wakeup', 'onset', 'wakeup'],
    #         'step': [2, 8, 3, 7],
    #     })

    #     # make a test id_encoding dictionary
    #     id_encoding = {'a': 0, 'b': 1}

    #     # run the function
    #     pp = AddStateLabels('./dummy_event_path', './dummy_id_encoding_path',
    #                         use_similarity_nan=False, fill_limit=2)
    #     pp.events = events
    #     pp.id_encoding = id_encoding

    #     result = pp.preprocess({0: data})

    #     # check that the result is as expected
    #     expected = pd.DataFrame({
    #         'series_id': [0] * 10 + [1] * 10,
    #         'awake': [1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
    #                   1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    #     })

    #     self.assertEquals(result[0].to_dict(), expected.to_dict())

    # def test_preprocess_unlabeled(self):
    #     # make a test data frame with one series of 10 steps
    #     data = pd.DataFrame({
    #         'series_id': [0] * 10,
    #     })

    #     # make a test events data frame with two events per series
    #     events = pd.DataFrame({
    #         'series_id': [],
    #         'event': [],
    #         'step': [],
    #     })

    #     # make a test id_encoding dictionary
    #     id_encoding = {'a': 0}

    #     # run the function
    #     pp = AddStateLabels('./dummy_event_path', './dummy_id_encoding_path',
    #                         use_similarity_nan=False, fill_limit=2)
    #     pp.events = events
    #     pp.id_encoding = id_encoding

    #     result = pp.preprocess(data)

    #     # check that the result is as expected
    #     expected = pd.DataFrame({
    #         'series_id': [0] * 10,
    #         'awake': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    #     })

    #     self.assertEquals(result.to_dict(), expected.to_dict())

    # def test_preprocess_nan(self):
    #     # make a test data frame with one series of 10 steps
    #     data = pd.DataFrame({
    #         'series_id': [0] * 10,
    #     })

    #     # make a test events data frame with two events per series
    #     events = pd.DataFrame({
    #         'series_id': ['a', 'a'],
    #         'event': ['wakeup', 'onset'],
    #         'step': [3, np.nan],
    #     })

    #     # make a test id_encoding dictionary
    #     id_encoding = {'a': 0}

    #     # run the function
    #     pp = AddStateLabels('./dummy_event_path', './dummy_id_encoding_path',
    #                         use_similarity_nan=False, fill_limit=2)
    #     pp.events = events
    #     pp.id_encoding = id_encoding

    #     result = pp.preprocess(data)

    #     # check that the result is as expected
    #     expected = pd.DataFrame({
    #         'series_id': [0] * 10,
    #         'awake': [0, 0, 0, 2, 2, 2, 2, 2, 2, 2],
    #     })

    #     self.assertEquals(result.to_dict(), expected.to_dict())

    # def test_preprocess_similarity_nan_normal(self):
    #     # make a test data frame with two series of 10 steps
    #     data = pd.DataFrame({
    #         'series_id': [0] * 10 + [1] * 10,
    #         'similarity_nan': [42] * 20,
    #     })

    #     # make a test events data frame with two events per series
    #     events = pd.DataFrame({
    #         'series_id': ['a', 'a', 'b', 'b'],
    #         'event': ['onset', 'wakeup', 'onset', 'wakeup'],
    #         'step': [2, 8, 3, 7],
    #     })

    #     # make a test id_encoding dictionary
    #     id_encoding = {'a': 0, 'b': 1}

    #     # run the function
    #     pp = AddStateLabels('./dummy_event_path', './dummy_id_encoding_path',
    #                         use_similarity_nan=True, fill_limit=10)
    #     pp.events = events
    #     pp.id_encoding = id_encoding

    #     result = pp.preprocess(data)

    #     # check that the result is as expected
    #     expected = pd.DataFrame({
    #         'series_id': [0] * 10 + [1] * 10,
    #         'awake': [1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
    #                   1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    #         'similarity_nan': [42] * 20,
    #     })

    #     self.assertEquals(result.to_dict(), expected.to_dict())

    # def test_preprocess_similarity_nan_with_nan(self):
    #     # make a test data frame with two series of 10 steps
    #     data = pd.DataFrame({
    #         'series_id': [0] * 20,
    #         'similarity_nan': [42] * 15 + [0] * 5,
    #     })

    #     # make a test events data frame with two events per series
    #     events = pd.DataFrame({
    #         'series_id': ['a', 'a', 'a', 'a'],
    #         'event': ['onset', 'wakeup', 'onset', 'wakeup'],
    #         'step': [2, 8, np.nan, np.nan],
    #     })

    #     # make a test id_encoding dictionary
    #     id_encoding = {'a': 0, 'b': 1}

    #     # run the function
    #     pp = AddStateLabels('./dummy_event_path', './dummy_id_encoding_path',
    #                         use_similarity_nan=True, fill_limit=2)
    #     pp.events = events
    #     pp.id_encoding = id_encoding

    #     result = pp.preprocess(data)

    #     # check that the result is as expected
    #     expected = pd.DataFrame({
    #         'series_id': [0] * 20,
    #         'similarity_nan': [42] * 15 + [0] * 5,
    #         'awake': [1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
    #                   3, 3, 3, 3, 3, 2, 2, 2, 2, 2],
    #     })

    #     self.assertEquals(result.to_dict(), expected.to_dict())

    # def test_preprocess_similarity_nan_with_nan2(self):
    #     # make a test data frame with two series of 10 steps
    #     data = pd.DataFrame({
    #         'series_id': [0] * 20,
    #         'similarity_nan': [42] * 15 + [0] * 5,
    #     })

    #     # make a test events data frame with two events per series
    #     events = pd.DataFrame({
    #         'series_id': ['a', 'a', 'a', 'a'],
    #         'event': ['onset', 'wakeup', 'onset', 'wakeup'],
    #         'step': [2, 8, np.nan, np.nan],
    #     })

    #     # make a test id_encoding dictionary
    #     id_encoding = {'a': 0, 'b': 1}

    #     # run the function
    #     pp = AddStateLabels('./dummy_event_path', './dummy_id_encoding_path',
    #                         use_similarity_nan=True, fill_limit=10)
    #     pp.events = events
    #     pp.id_encoding = id_encoding

    #     result = pp.preprocess(data)

    #     # check that the result is as expected
    #     expected = pd.DataFrame({
    #         'series_id': [0] * 20,
    #         'similarity_nan': [42] * 15 + [0] * 5,
    #         'awake': [1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
    #                   1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    #     })

    #     self.assertEquals(result.to_dict(), expected.to_dict())
