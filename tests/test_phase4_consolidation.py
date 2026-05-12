"""Tests for Benna-Fusi multi-timescale consolidation."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestConsolidation(unittest.TestCase):

    def setUp(self):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        self.config = ConsolidationConfig(m=4, alpha=0.25)
        self.state = ConsolidationState(self.config, device="cpu")

    def test_add_pattern_initializes_u1(self):
        idx = self.state.add_pattern(novelty_strength=1.0)
        self.assertEqual(idx, 0)
        self.assertEqual(self.state.n_patterns, 1)
        self.assertAlmostEqual(self.state.u[0, 0].item(), 1.0)
        # u_2...u_m should start at zero
        for k in range(1, self.config.m):
            self.assertAlmostEqual(self.state.u[0, k].item(), 0.0)

    def test_reinforce_only_affects_u1(self):
        self.state.add_pattern(novelty_strength=1.0)
        self.state.reinforce(0, magnitude=0.5)
        self.assertAlmostEqual(self.state.u[0, 0].item(), 1.5)
        for k in range(1, self.config.m):
            self.assertAlmostEqual(self.state.u[0, k].item(), 0.0)

    def test_step_dynamics_propagates_to_slow_variables(self):
        """Input at u_1 should propagate forward through the chain."""
        self.state.add_pattern(novelty_strength=1.0)
        initial_u2 = self.state.u[0, 1].item()
        for _ in range(20):
            self.state.step_dynamics()
        # u_2 should have grown from zero due to coupling
        self.assertGreater(self.state.u[0, 1].item(), initial_u2)

    def test_step_dynamics_decays_u1_without_reinforcement(self):
        self.state.add_pattern(novelty_strength=1.0)
        initial_u1 = self.state.u[0, 0].item()
        for _ in range(10):
            self.state.step_dynamics()
        self.assertLess(self.state.u[0, 0].item(), initial_u1)

    def test_bidirectional_coupling(self):
        """A pulse at u_2 should propagate back to u_1 (bidirectional)."""
        self.state.add_pattern(novelty_strength=0.0)
        self.state.u[0, 1] = 1.0
        initial_u1 = self.state.u[0, 0].item()
        for _ in range(5):
            self.state.step_dynamics()
        # u_1 should have received some signal from u_2
        self.assertGreater(self.state.u[0, 0].item(), initial_u1)

    def test_effective_strength_weighted_sum(self):
        self.state.add_pattern(novelty_strength=1.0)
        s = self.state.effective_strength()
        self.assertEqual(s.shape[0], 1)
        # With default weights (2^(1-k)), strength = u_1 * 1 + u_2 * 0.5 + ...
        expected = sum(
            self.state.u[0, k].item() * (2.0 ** (-k))
            for k in range(self.config.m)
        )
        self.assertAlmostEqual(s[0].item(), expected, places=5)

    def test_death_after_window(self):
        """A pattern with no reinforcement should eventually be marked dead."""
        cfg = self.state.config
        # Use small death_window for fast test
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        fast_cfg = ConsolidationConfig(
            m=4, alpha=0.25, death_threshold=0.001, death_window=5,
        )
        fast_state = ConsolidationState(fast_cfg, device="cpu")
        fast_state.add_pattern(novelty_strength=0.005)  # already near threshold
        for _ in range(20):
            fast_state.step_dynamics()
        # Strength should have decayed; pattern should be in dead list
        dead = fast_state.dead_indices()
        self.assertIn(0, dead)

    def test_remove_pattern(self):
        self.state.add_pattern(novelty_strength=1.0)
        self.state.add_pattern(novelty_strength=2.0)
        self.assertEqual(self.state.n_patterns, 2)
        self.state.remove_pattern(0)
        self.assertEqual(self.state.n_patterns, 1)
        # The remaining pattern should be the one initialized to 2.0
        self.assertAlmostEqual(self.state.u[0, 0].item(), 2.0)

    def test_multiple_patterns_independent_dynamics(self):
        """Sparse updates: reinforcing one pattern doesn't affect another."""
        self.state.add_pattern(novelty_strength=1.0)
        self.state.add_pattern(novelty_strength=1.0)
        before = self.state.u[1].clone()
        self.state.reinforce(0, magnitude=10.0)
        # Pattern 1's u should be unchanged
        self.assertTrue(torch.allclose(self.state.u[1], before))

    def test_input_vector_to_step_dynamics(self):
        self.state.add_pattern(novelty_strength=0.0)
        self.state.add_pattern(novelty_strength=0.0)
        inp = torch.tensor([1.0, 2.0])
        self.state.step_dynamics(input_vector=inp)
        # Both patterns received their respective inputs to u_1
        self.assertGreater(self.state.u[0, 0].item(), 0.0)
        self.assertGreater(self.state.u[1, 0].item(), self.state.u[0, 0].item())

    def test_invalid_m_raises(self):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        with self.assertRaises(ValueError):
            ConsolidationState(ConsolidationConfig(m=1), device="cpu")

    def test_invalid_strength_weights_raises(self):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        with self.assertRaises(ValueError):
            ConsolidationState(
                ConsolidationConfig(m=4, strength_weights=[1.0, 1.0]),  # wrong length
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
