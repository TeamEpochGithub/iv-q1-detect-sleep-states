from unittest import TestCase

import torch
from src.loss.regression_loss import RegressionLoss


class RegressionTest(TestCase):

    def test_regression_loss(self):
        test_pred = [[123, 444, 0.3, 0.2], [4123, 2235, 0.5, 0.2]]
        test_true = [[100, 400, 0, 0], [-1, -1, 1, 1]]

        test_pred = torch.tensor(test_pred)
        test_true = torch.tensor(test_true)
        loss = RegressionLoss()

        # Create mask from true values
        # If index 2 is 1 then index 0 is 0 else 1
        # If index 3 is 1 then index 1 is 0 else 1
        # Index 2 and 3 always have 1
        mask = torch.ones_like(test_true)
        mask[:, 0] = 1 - test_true[:, 2]
        mask[:, 1] = 1 - test_true[:, 3]

        test_loss = loss(test_pred, test_true, mask)
        self.assertEqual(test_loss, 308.2525)
