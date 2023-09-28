from unittest import TestCase

from src.models.example_model import SimpleModel
from src.optimizer.optimizer import Optimizer


class TestOptimizer(TestCase):
    """
    This class tests the optimizer class.
    """
    def setUp(self):
        self.model = SimpleModel(2, 10, 1, None)
        self.lr = 0.01

    def test_get_optimizer_adam(self):
        optimizer_text = "adam-torch"
        optimizer = Optimizer.get_optimizer(optimizer_text, self.lr, self.model)
        self.assertEqual(optimizer_text[:4], optimizer.__class__.__name__.lower())
        self.assertEqual(self.lr, optimizer.defaults["lr"])

    def test_get_optimizer_sgd(self):
        optimizer_text = "sgd-torch"
        optimizer = Optimizer.get_optimizer(optimizer_text, self.lr, self.model)
        self.assertEqual(optimizer_text[:3], optimizer.__class__.__name__.lower())
        self.assertEqual(self.lr, optimizer.defaults["lr"])

    def test_get_optimizer_error(self):
        optimizer_text = "thisisnotanoptimizerfunction"
        with self.assertRaises(Exception) as context:
            Optimizer.get_optimizer(optimizer_text, self.lr, self.model)
        self.assertEqual("Optimizer function not found: " + optimizer_text, str(context.exception))
