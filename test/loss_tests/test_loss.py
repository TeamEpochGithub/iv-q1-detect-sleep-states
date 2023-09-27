from unittest import TestCase

from src.loss.loss import Loss, LossException


class TestLoss(TestCase):
    """
    This class tests the loss class.
    """

    def test_get_loss_mse(self):
        loss_text = "mse-torch"
        loss = Loss.get_loss(loss_text)
        self.assertEqual("mseloss", loss.__class__.__name__.lower())

    def test_get_loss_crossentropy(self):
        loss_text = "crossentropy-torch"
        loss = Loss.get_loss(loss_text)
        self.assertEqual("crossentropyloss", loss.__class__.__name__.lower())

    def test_get_loss_error(self):
        loss_text = "thisisnotalossfunction"
        with self.assertRaises(LossException) as context:
            Loss.get_loss(loss_text)
        self.assertEqual("Loss function not found: " + loss_text, str(context.exception))
