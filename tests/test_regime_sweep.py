import unittest

from experiments.regime_sweep import classify_distractor, classify_shuffle


class RegimeSweepTests(unittest.TestCase):
    def test_shuffle_classifier_names_robust_temporal(self):
        self.assertEqual(classify_shuffle(0.95, 0.25, 0.2), "robust_temporal")

    def test_distractor_classifier_names_interference(self):
        self.assertEqual(classify_distractor(0.25, 0.75, 0.2), "interference")


if __name__ == "__main__":
    unittest.main()

