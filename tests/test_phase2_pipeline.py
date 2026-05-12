import unittest

from energy_memory.phase2.corpus import build_vocabulary, encode_texts, make_windows
from energy_memory.phase2.encoding import mask_positions, masked_window
from energy_memory.phase2.metrics import summarize_binary_outcomes


class Phase2PipelineTests(unittest.TestCase):
    def test_mask_position_semantics_match_spec_examples(self):
        self.assertEqual(mask_positions(8, 1, "center"), [4])
        self.assertEqual(mask_positions(8, 2, "center"), [3, 4])
        self.assertEqual(mask_positions(16, 3, "end"), [13, 14, 15])
        self.assertEqual(mask_positions(32, 2, "edge"), [0, 1])

    def test_masked_window_replaces_requested_positions(self):
        self.assertEqual(masked_window([1, 2, 3, 4], [1, 3], 99), [1, 99, 3, 99])

    def test_text_pipeline_builds_windows(self):
        vocab = build_vocabulary(["Alpha beta gamma delta epsilon zeta"])
        token_ids = encode_texts(["Alpha beta gamma delta epsilon zeta"], vocab)
        windows = make_windows(token_ids, 2)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0], tuple(token_ids[:2]))

    def test_summary_includes_cap_coverage_and_metastable(self):
        summary = summarize_binary_outcomes(
            outcomes=[1, 0, 1],
            gaps=[0.2, 0.1, 0.4],
            entropies=[0.1, 0.3, 0.2],
            energies=[-0.5, -0.3, -0.4],
            top_scores=[0.99, 0.45, 0.8],
        )
        self.assertAlmostEqual(summary.accuracy, 2 / 3)
        self.assertGreater(summary.cap_coverage_error[0.5], 0.0)
        self.assertGreater(summary.metastable_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
