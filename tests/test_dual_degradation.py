import unittest

from experiments.dual_degradation_sweep import classify


class DualDegradationTests(unittest.TestCase):
    def test_classifies_correct_ambiguous_state(self):
        self.assertEqual(classify("stair", 1.0, 0.4), "ambiguous_correct")

    def test_classifies_sparse_wrong_state(self):
        self.assertEqual(classify("stair", 0.0, 0.1), "wrong_or_sparse")


if __name__ == "__main__":
    unittest.main()

