from unittest import TestCase

import numpy as np
import pandas as pd

from src.preprocessing.add_state_labels import AddStateLabels


class TestAddStateLabels(TestCase):

    def test_preprocess_one_night(self):
        # make a test data frame with two series of 10 steps
        data = pd.DataFrame({
            'series_id': [0] * 10 + [1] * 10,
        })

        # make a test events data frame with two events per series
        events = pd.DataFrame({
            'series_id': ['a', 'a', 'b', 'b'],
            'event': ['onset', 'wakeup', 'onset', 'wakeup'],
            'step': [2, 8, 3, 7],
        })

        # make a test id_encoding dictionary
        id_encoding = {'a': 0, 'b': 1}

        # run the function
        pp = AddStateLabels('./dummy_event_path')
        pp.events = events
        pp.id_encoding = id_encoding

        result = pp.preprocess(data)

        # check that the result is as expected
        expected = pd.DataFrame({
            'series_id': [0] * 10 + [1] * 10,
            'awake': [1, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                      1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        })

        self.assertEquals(result.to_dict(), expected.to_dict())

    def test_preprocess_unlabeled(self):
        # make a test data frame with one series of 10 steps
        data = pd.DataFrame({
            'series_id': [0] * 10,
        })

        # make a test events data frame with two events per series
        events = pd.DataFrame({
            'series_id': [],
            'event': [],
            'step': [],
        })

        # make a test id_encoding dictionary
        id_encoding = {'a': 0}

        # run the function
        pp = AddStateLabels('./dummy_event_path')
        pp.events = events
        pp.id_encoding = id_encoding

        result = pp.preprocess(data)

        # check that the result is as expected
        expected = pd.DataFrame({
            'series_id': [0] * 10,
            'awake': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        })

        self.assertEquals(result.to_dict(), expected.to_dict())

    def test_preprocess_nan(self):
        # make a test data frame with one series of 10 steps
        data = pd.DataFrame({
            'series_id': [0] * 10,
        })

        # make a test events data frame with two events per series
        events = pd.DataFrame({
            'series_id': ['a', 'a'],
            'event': ['wakeup', 'onset'],
            'step': [3, np.nan],
        })

        # make a test id_encoding dictionary
        id_encoding = {'a': 0}

        # run the function
        pp = AddStateLabels('./dummy_event_path')
        pp.events = events
        pp.id_encoding = id_encoding

        result = pp.preprocess(data)

        # check that the result is as expected
        expected = pd.DataFrame({
            'series_id': [0] * 10,
            'awake': [0, 0, 0, 2, 2, 2, 2, 2, 2, 2],
        })

        self.assertEquals(result.to_dict(), expected.to_dict())