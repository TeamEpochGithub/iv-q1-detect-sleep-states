from unittest import TestCase
import pandas as pd
from src.feature_engineering.parser import Parser


class TestParser(TestCase):

    def test_feature_engineering_single_diff(self):
        series = pd.DataFrame({
            'step': range(10),
            'anglez': [0, 0, 0, 1, 2, 3, 4, 4, 3, 2]})

        expected_feature = pd.Series([0, 0, 0, 1, 1, 1, 1, 0, -1, -1], dtype=float)
        expected = pd.DataFrame({
            'step': series.step,
            'anglez': series.anglez,
            'f_anglez_diff': expected_feature})

        parser = Parser(['anglez_diff'])
        result = parser.feature_engineering_single(series)

        pd.testing.assert_frame_equal(expected, result)

    def test_feature_engineering_single_abs(self):
        series = pd.DataFrame({
            'step': range(10),
            'anglez': [0.0, 0, 0, -10, 2, 3, 4, 4, 3, 2]})

        feature_name = 'anglez_abs'
        expected_feature = pd.Series([0, 0, 0, 10, 2, 3, 4, 4, 3, 2], dtype=float)
        expected = pd.DataFrame({
            'step': series.step,
            'anglez': series.anglez,
            'f_' + feature_name: expected_feature})

        parser = Parser([feature_name])
        result = parser.feature_engineering_single(series)

        pd.testing.assert_frame_equal(expected, result)

    def test_feature_engineering_abs_diff(self):
        series = pd.DataFrame({
            'step': range(10),
            'anglez': [0.0, 0, 0, -10, 2, 3, 4, 4, 3, 2]})

        feature_name = 'anglez_diff_abs'
        expected_feature = pd.Series([0, 0, 0, 10, 12, 1, 1, 0, 1, 1], dtype=float)
        expected = pd.DataFrame({
            'step': series.step,
            'anglez': series.anglez,
            'f_' + feature_name: expected_feature})

        parser = Parser([feature_name])
        result = parser.feature_engineering_single(series)

        pd.testing.assert_frame_equal(expected, result)

    def test_feature_engineering_mean(self):
        series = pd.DataFrame({
            'step': range(10),
            'anglez': [0.0, 0, 0, -10, 2, 3, 4, 4, 3, 2]})

        feature_name = 'anglez_mean_3'
        expected_feature = series.anglez.rolling(3, center=True).mean().bfill().ffill()
        expected = pd.DataFrame({
            'step': series.step,
            'anglez': series.anglez,
            'f_' + feature_name: expected_feature})

        parser = Parser([feature_name])
        result = parser.feature_engineering_single(series)

        pd.testing.assert_frame_equal(expected, result)

    def test_feature_engineering_rotation(self):
        series = pd.DataFrame({
            'step': range(10),
            'anglez': [0.0, 0, 0, -10, 2, 3, 4, 4, 3, 2]})

        feature_name = 'anglez_diff_abs_clip_10_median_3'
        expected_feature = (series.anglez
                            .diff()
                            .abs()
                            .bfill()
                            .clip(upper=10)
                            .rolling(window=3, center=True)
                            .median()
                            .ffill()
                            .bfill())

        expected = pd.DataFrame({
            'step': series.step,
            'anglez': series.anglez,
            'f_' + feature_name: expected_feature})

        parser = Parser([feature_name])
        result = parser.feature_engineering_single(series)

        pd.testing.assert_frame_equal(expected, result)

    def test_feature_engineering_savgol(self):
        series = pd.DataFrame({
            'step': range(10),
            'anglez': [0.0, 0, 0, -10, 2, 3, 4, 4, 3, 2]})

        feature_name = 'anglez_diff_abs_clip_10_savgol_5'
        expected_feature = (series.anglez
                            .diff()
                            .abs()
                            .bfill()
                            .clip(upper=10))
        from scipy.signal import savgol_filter
        expected_feature = savgol_filter(expected_feature, 5, 3)

        expected = pd.DataFrame({
            'step': series.step,
            'anglez': series.anglez,
            'f_' + feature_name: expected_feature})

        parser = Parser([feature_name])
        result = parser.feature_engineering_single(series)

        pd.testing.assert_frame_equal(expected, result)
