"""Tests for the Phase-3 basin-membership readout + Selectivity-Δ two-floor rule."""

import unittest

import torch

from energy_memory.phase3.basin_readout import (
    newcombe_diff_ci,
    selectivity_delta,
    top_index_hits,
)


class TopIndexHitsTest(unittest.TestCase):
    def test_counts_matches(self):
        top = torch.tensor([0, 1, 2, 3, 1])
        tgt = torch.tensor([0, 1, 9, 3, 7])
        self.assertEqual(top_index_hits(top, tgt), 3)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            top_index_hits(torch.tensor([0, 1]), torch.tensor([0]))

    def test_all_or_none(self):
        t = torch.arange(5)
        self.assertEqual(top_index_hits(t, t), 5)
        self.assertEqual(top_index_hits(t, t + 100), 0)


class NewcombeDiffCITest(unittest.TestCase):
    def test_brackets_point_estimate(self):
        lo, hi = newcombe_diff_ci(80, 100, 30, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertLess(lo, hi)

    def test_zero_difference_interval_spans_zero(self):
        lo, hi = newcombe_diff_ci(50, 100, 50, 100)
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_positive_total_required(self):
        with self.assertRaises(ValueError):
            newcombe_diff_ci(0, 0, 1, 10)


class SelectivityDeltaTest(unittest.TestCase):
    def test_clean_pass(self):
        # true arm well above chance (0.25), shuffled near chance, large n.
        sd = selectivity_delta(
            true_hits=90, true_n=100, shuffled_hits=27, shuffled_n=100, chance=0.25
        )
        self.assertTrue(sd.two_floor_pass)
        self.assertFalse(sd.anti_overlap_flag)
        self.assertGreater(sd.delta_lo, 0.0)
        self.assertGreater(sd.true_lo, 0.25)

    def test_true_arm_at_chance_fails_first_floor(self):
        # true arm == chance: first floor (true_lo > chance) must fail even if
        # the shuffled arm is lower.
        sd = selectivity_delta(
            true_hits=25, true_n=100, shuffled_hits=10, shuffled_n=100, chance=0.25
        )
        self.assertFalse(sd.two_floor_pass)

    def test_subchance_shuffled_arm_rejected(self):
        # Δ>0 but driven by a significantly sub-chance shuffled arm -> rejected
        # as the anti-overlap signature, not a pass.
        sd = selectivity_delta(
            true_hits=40, true_n=200, shuffled_hits=2, shuffled_n=200, chance=0.25
        )
        self.assertTrue(sd.anti_overlap_flag)
        self.assertFalse(sd.two_floor_pass)

    def test_delta_ci_must_exclude_zero(self):
        # true arm above chance but a small Δ at tiny n -> the Newcombe Δ CI
        # spans 0 -> the second floor fails even though the first floor passes.
        sd = selectivity_delta(
            true_hits=3, true_n=4, shuffled_hits=2, shuffled_n=4, chance=0.25
        )
        self.assertGreater(sd.true_lo, 0.25)  # first floor passes
        self.assertLess(sd.delta_lo, 0.0)  # but Δ CI includes 0
        self.assertFalse(sd.two_floor_pass)

    def test_dict_roundtrip(self):
        sd = selectivity_delta(
            true_hits=90, true_n=100, shuffled_hits=27, shuffled_n=100, chance=0.25
        )
        d = sd.as_dict()
        self.assertIn("delta_ci", d)
        self.assertEqual(d["two_floor_pass"], True)


if __name__ == "__main__":
    unittest.main()
