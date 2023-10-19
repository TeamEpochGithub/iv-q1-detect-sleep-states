from unittest import TestCase

from src.preprocessing.add_noise import AddNoise
from src.preprocessing.add_regression_labels import AddRegressionLabels
from src.preprocessing.add_segmentation_labels import AddSegmentationLabels
from src.preprocessing.add_state_labels import AddStateLabels
from src.preprocessing.mem_reduce import MemReduce
from src.preprocessing.pp import PP, PPException
from src.preprocessing.remove_unlabeled import RemoveUnlabeled
from src.preprocessing.split_windows import SplitWindows
from src.preprocessing.truncate import Truncate


class TestPP(TestCase):
    def test_from_config_single(self):
        self.assertIsInstance(PP.from_config_single({"kind": "mem_reduce", "id_encoding_path": "a.json"}), MemReduce)
        self.assertIsInstance(PP.from_config_single({"kind": "add_noise"}), AddNoise)
        self.assertIsInstance(PP.from_config_single({"kind": "add_regression_labels"}), AddRegressionLabels)
        self.assertIsInstance(PP.from_config_single({"kind": "add_segmentation_labels"}), AddSegmentationLabels)
        self.assertIsInstance(PP.from_config_single({"kind": "add_state_labels",
                                                     "id_encoding_path": "e.json",
                                                     "events_path": "e.csv"}), AddStateLabels)
        self.assertIsInstance(PP.from_config_single({"kind": "remove_unlabeled"}), RemoveUnlabeled)
        self.assertIsInstance(PP.from_config_single({"kind": "split_windows"}), SplitWindows)
        self.assertIsInstance(PP.from_config_single({"kind": "truncate"}), Truncate)
        self.assertRaises(PPException, PP.from_config_single, {"kind": "e"})

    def test_from_config(self):
        config: dict = {
            "preprocessing": [
                {
                    "kind": "mem_reduce",
                    "id_encoding_path": "a.json"
                },
                {
                    "kind": "add_state_labels",
                    "id_encoding_path": "b.json",
                    "events_path": "c.csv"
                },
                {
                    "kind": "split_windows",
                    "start_hour": 1,
                    "window_size": 2
                }
            ],
        }

        # Test parsing with training=True and training=False
        pp_steps = PP.from_config(config["preprocessing"], training=True)

        self.assertIsInstance(pp_steps[0], MemReduce)
        self.assertEqual(pp_steps[0].id_encoding_path, "a.json")
        self.assertIsInstance(pp_steps[1], AddStateLabels)
        self.assertEqual(pp_steps[1].id_encoding_path, "b.json")
        self.assertEqual(pp_steps[1].events_path, "c.csv")
        self.assertIsInstance(pp_steps[2], SplitWindows)
        self.assertEqual(pp_steps[2].start_hour, 1)
        self.assertEqual(pp_steps[2].window_size, 2)

        pp_steps = PP.from_config(config["preprocessing"], training=False)

        self.assertIsInstance(pp_steps[0], MemReduce)
        self.assertEqual(pp_steps[0].id_encoding_path, "a.json")
        self.assertIsInstance(pp_steps[1], SplitWindows)
        self.assertEqual(pp_steps[1].start_hour, 1)
        self.assertEqual(pp_steps[1].window_size, 2)