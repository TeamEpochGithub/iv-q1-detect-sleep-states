from unittest import TestCase

from src.models.example_model import SimpleModel
from src.optimiser.optimiser import Optimiser


class TestOptimiser(TestCase):
    """
    This class tests the optimiser class.
    """

    def setUp(self):
        self.model = SimpleModel(2, 10, 1, None)
        self.lr = 0.01
        self.weight_decay = 0.01

    def test_get_optimiser_adam(self):
        optimiser_text = "adam-torch"
        optimiser = Optimiser.get_optimiser(optimiser_text, self.lr, self.weight_decay, self.model)
        self.assertEqual(optimiser_text[:4], optimiser.__class__.__name__.lower())
        self.assertEqual(self.lr, optimiser.defaults["lr"])

    def test_get_optimiser_sgd(self):
        optimiser_text = "sgd-torch"
        optimiser = Optimiser.get_optimiser(optimiser_text, self.lr, self.weight_decay, self.model)
        self.assertEqual(optimiser_text[:3], optimiser.__class__.__name__.lower())
        self.assertEqual(self.lr, optimiser.defaults["lr"])

    def test_get_optimiser_error(self):
        optimiser_text = "thisisnotanoptimiserfunction"
        with self.assertRaises(Exception) as context:
            Optimiser.get_optimiser(optimiser_text, self.lr, self.weight_decay, self.model)
        self.assertEqual("Optimiser function not found: " + optimiser_text, str(context.exception))
