from unittest import TestCase

from src.preprocessing.add_noise import AddNoise
from src.preprocessing.add_segmentation_labels import AddSegmentationLabels
from src.preprocessing.add_state_labels import AddStateLabels
from src.preprocessing.mem_reduce import MemReduce
from src.preprocessing.pp import PP, PPException
from src.preprocessing.split_windows import SplitWindows


class TestPP(TestCase):
    def test_from_config_single(self):
        self.assertIsInstance(PP.from_config_single({"kind": "mem_reduce"}), MemReduce)
        self.assertIsInstance(PP.from_config_single({"kind": "add_noise"}), AddNoise)

        self.assertIsInstance(PP.from_config_single({"kind": "add_segmentation_labels"}), AddSegmentationLabels)
        self.assertIsInstance(PP.from_config_single({"kind": "add_state_labels",
                                                     "events_path": "e.csv",
                                                     "use_similarity_nan": False}), AddStateLabels)

        self.assertIsInstance(PP.from_config_single({"kind": "split_windows"}), SplitWindows)
        self.assertRaises(AssertionError, PP.from_config_single, {"kind": "e"})

    def test_from_config(self):
        config: dict = {
            "preprocessing": [
                {
                    "kind": "mem_reduce"
                },
                {
                    "kind": "add_state_labels",
                    "events_path": "c.csv",
                    "use_similarity_nan": False
                },
                {
                    "kind": "split_windows",
                    "start_hour": 1
                }
            ],
        }

        # Test parsing with training=True and training=False
        pp_steps = PP.from_config(config["preprocessing"], training=True)

        self.assertIsInstance(pp_steps[0], MemReduce)
        self.assertIsInstance(pp_steps[1], AddStateLabels)
        self.assertEqual(pp_steps[1].events_path, "c.csv")
        self.assertIsInstance(pp_steps[2], SplitWindows)
        self.assertEqual(pp_steps[2].start_hour, 1)

        config_2: dict = {
            "preprocessing": [
                {
                    "kind": "mem_reduce"
                },
                {
                    "kind": "add_state_labels",
                    "events_path": "c.csv",
                    "use_similarity_nan": False
                },
                {
                    "kind": "split_windows",
                    "start_hour": 1
                }
            ],
        }

        pp_steps = PP.from_config(config_2["preprocessing"], training=False)

        self.assertIsInstance(pp_steps[0], MemReduce)
        self.assertIsInstance(pp_steps[1], SplitWindows)
        self.assertEqual(pp_steps[1].start_hour, 1)
