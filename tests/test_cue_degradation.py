import unittest

from experiments.cue_degradation_sweep import classify


class CueDegradationTests(unittest.TestCase):
    def test_classifies_committed_correct_state(self):
        self.assertEqual(classify("stair", 1.0, 0.01), "committed")

    def test_classifies_flip(self):
        self.assertEqual(classify("ladder", 1.0, 0.01), "flipped")


if __name__ == "__main__":
    unittest.main()

