from unittest import TestCase

import pandas as pd

from src.pretrain.downsampler import Downsampler


class TestDownsampler(TestCase):
    """
    Test the downsampler class.
    """

    downsampler = Downsampler(2, ['x', 'y', 'z'], ['mean', 'median', 'max', 'min', 'std', 'var', 'sum'], 'mean')

    def test_downsamplerX(self):
        """
        Test the downsampler for the X data.
        """
        # Create a dummy dataframe
        dummy_df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8], 'y': [1, 2, 3, 4, 5, 6, 7, 8], 'z': [1, 2, 3, 4, 5, 6, 7, 8]})
        # Downsample the dummy dataframe
        dummy_df_downsampled = self.downsampler.downsampleX(dummy_df)
        # Check if the shape is correct
        self.assertEqual(dummy_df_downsampled.shape, (4, 3 * 7))
        # Check if the values are correct
        self.assertEqual(dummy_df_downsampled['x_mean'].values.tolist(), [1.5, 3.5, 5.5, 7.5])
        self.assertEqual(dummy_df_downsampled['y_mean'].values.tolist(), [1.5, 3.5, 5.5, 7.5])
        self.assertEqual(dummy_df_downsampled['z_mean'].values.tolist(), [1.5, 3.5, 5.5, 7.5])
        self.assertEqual(dummy_df_downsampled['x_median'].values.tolist(), [1.5, 3.5, 5.5, 7.5])
        self.assertEqual(dummy_df_downsampled['y_median'].values.tolist(), [1.5, 3.5, 5.5, 7.5])
        self.assertEqual(dummy_df_downsampled['z_median'].values.tolist(), [1.5, 3.5, 5.5, 7.5])
        self.assertEqual(dummy_df_downsampled['x_max'].values.tolist(), [2, 4, 6, 8])
        self.assertEqual(dummy_df_downsampled['y_max'].values.tolist(), [2, 4, 6, 8])
        self.assertEqual(dummy_df_downsampled['z_max'].values.tolist(), [2, 4, 6, 8])
        self.assertEqual(dummy_df_downsampled['x_min'].values.tolist(), [1, 3, 5, 7])
        self.assertEqual(dummy_df_downsampled['y_min'].values.tolist(), [1, 3, 5, 7])
        self.assertEqual(dummy_df_downsampled['z_min'].values.tolist(), [1, 3, 5, 7])
        self.assertEqual(dummy_df_downsampled['x_std'].values.tolist(), [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(dummy_df_downsampled['y_std'].values.tolist(), [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(dummy_df_downsampled['z_std'].values.tolist(), [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(dummy_df_downsampled['x_var'].values.tolist(), [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(dummy_df_downsampled['y_var'].values.tolist(), [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(dummy_df_downsampled['z_var'].values.tolist(), [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(dummy_df_downsampled['x_sum'].values.tolist(), [3, 7, 11, 15])
        self.assertEqual(dummy_df_downsampled['y_sum'].values.tolist(), [3, 7, 11, 15])
        self.assertEqual(dummy_df_downsampled['z_sum'].values.tolist(), [3, 7, 11, 15])

    def test_downsamplerY(self):
        """
        Test the downsampler for the y data.
        """
        # Create a dummy dataframe
        dummy_df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8], 'y': [1, 2, 3, 4, 5, 6, 7, 8]})
        # Downsample the dummy dataframe
        dummy_df_downsampled = self.downsampler.downsampleY(dummy_df)
        # Check if the shape is correct
        self.assertEqual(dummy_df_downsampled.shape, (4, 2))
        # Check if the values are correct
        self.assertEqual(dummy_df_downsampled['x'].values.tolist(), [1.5, 3.5, 5.5, 7.5])
        self.assertEqual(dummy_df_downsampled['y'].values.tolist(), [1.5, 3.5, 5.5, 7.5])
