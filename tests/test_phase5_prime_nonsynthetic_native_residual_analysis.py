"""Tests for Report 090 residual-analysis helpers."""

from __future__ import annotations

import unittest


class TestNonsyntheticNativeResidualAnalysis(unittest.TestCase):

    def test_summary_separates_exact_wrong_role_and_frequency_enrichment(self):
        from scripts.phase5_prime_nonsynthetic_native_residual_analysis import (
            _summary_for_condition,
        )

        rows = [
            {
                "correct": True,
                "scene_hit": True,
                "same_scene_unbind_matches_target": True,
                "retrieved_unbind_matches_target": True,
                "target_in_retrieved_scene_any_role": True,
                "target_seed_frequency": 15,
                "target_is_special": False,
                "target_is_top16_vocab": True,
                "target_is_top64_vocab": True,
                "role_delta": 1,
                "target_atom": 2,
                "target_token": "the",
            },
            {
                "correct": True,
                "scene_hit": False,
                "same_scene_unbind_matches_target": False,
                "retrieved_unbind_matches_target": True,
                "target_in_retrieved_scene_any_role": True,
                "target_seed_frequency": 8,
                "target_is_special": False,
                "target_is_top16_vocab": False,
                "target_is_top64_vocab": True,
                "role_delta": 3,
                "target_atom": 24,
                "target_token": "memory",
            },
            {
                "correct": False,
                "scene_hit": True,
                "same_scene_unbind_matches_target": False,
                "retrieved_unbind_matches_target": False,
                "target_in_retrieved_scene_any_role": False,
                "target_seed_frequency": 1,
                "target_is_special": False,
                "target_is_top16_vocab": False,
                "target_is_top64_vocab": False,
                "role_delta": 1,
                "target_atom": 100,
                "target_token": "rare",
            },
        ]

        summary = _summary_for_condition(rows)
        self.assertEqual(summary["n_total"], 3)
        self.assertEqual(summary["n_correct"], 2)
        self.assertAlmostEqual(summary["hit_retrieved_unbind_match_rate"], 1.0)
        self.assertAlmostEqual(summary["hit_same_scene_unbind_match_rate"], 0.5)
        self.assertGreater(
            summary["hit_target_frequency"]["mean"],
            summary["all_target_frequency"]["mean"],
        )
        self.assertEqual(summary["hit_role_delta_counts"], {"1": 1, "3": 1})


if __name__ == "__main__":
    unittest.main()
