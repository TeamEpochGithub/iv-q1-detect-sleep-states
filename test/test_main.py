from unittest import TestCase
from src.main import to_test


class Test(TestCase):
    def test_run_correct(self):
        result = to_test(5)
        self.assertEqual(result, 10)

    # def test_run_file_fail(self):
    #     result = to_test(5)
    #     self.assertEqual(result, 5)
