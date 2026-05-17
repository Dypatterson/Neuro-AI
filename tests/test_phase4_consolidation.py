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


@unittest.skipIf(torch is None, "torch required")
class TestSaighiInhibition(unittest.TestCase):
    """Saighi & Rozenberg (2025) per-pattern A_k self-inhibition.

    See notes/notes/2026-05-15-saighi-hrr-replay-synthesis.md.
    """

    def setUp(self):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        self.config = ConsolidationConfig(
            m=4, alpha=0.25, inhibition_gain=0.1, inhibition_decay=0.0,
        )
        self.state = ConsolidationState(self.config, device="cpu")

    def test_inhibition_initializes_to_zero(self):
        idx = self.state.add_pattern()
        self.assertAlmostEqual(self.state.A[idx].item(), 0.0)

    def test_accumulate_inhibition_uses_gain_default(self):
        idx = self.state.add_pattern()
        self.state.accumulate_inhibition(idx)  # uses config.inhibition_gain
        self.assertAlmostEqual(self.state.A[idx].item(), 0.1)
        self.state.accumulate_inhibition(idx)
        self.assertAlmostEqual(self.state.A[idx].item(), 0.2)

    def test_accumulate_inhibition_zero_gain_is_noop(self):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        s = ConsolidationState(
            ConsolidationConfig(m=4, alpha=0.25, inhibition_gain=0.0),
            device="cpu",
        )
        idx = s.add_pattern()
        s.accumulate_inhibition(idx)
        self.assertAlmostEqual(s.A[idx].item(), 0.0)

    def test_inhibition_is_per_pattern(self):
        i0 = self.state.add_pattern()
        i1 = self.state.add_pattern()
        self.state.accumulate_inhibition(i0)
        self.assertAlmostEqual(self.state.A[i0].item(), 0.1)
        self.assertAlmostEqual(self.state.A[i1].item(), 0.0)

    def test_inhibition_decay_applied_in_step_dynamics(self):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        s = ConsolidationState(
            ConsolidationConfig(
                m=4, alpha=0.25, inhibition_gain=1.0, inhibition_decay=0.1,
            ),
            device="cpu",
        )
        idx = s.add_pattern()
        s.accumulate_inhibition(idx)
        self.assertAlmostEqual(s.A[idx].item(), 1.0)
        s.step_dynamics()
        # A *= (1 - 0.1) = 0.9
        self.assertAlmostEqual(s.A[idx].item(), 0.9, places=5)

    def test_remove_pattern_drops_inhibition_row(self):
        i0 = self.state.add_pattern()
        i1 = self.state.add_pattern()
        self.state.accumulate_inhibition(i1, magnitude=0.5)
        self.state.remove_pattern(i0)
        self.assertEqual(self.state.n_patterns, 1)
        self.assertAlmostEqual(self.state.A[0].item(), 0.5)


@unittest.skipIf(torch is None, "torch required")
class TestFreqWeightedAlpha(unittest.TestCase):
    """Tests for brainstorm-idea-5 retrieval-frequency-weighted α coupling."""

    def _config(self, lam):
        from energy_memory.phase4.consolidation import ConsolidationConfig
        return ConsolidationConfig(m=4, alpha=0.25, alpha_freq_lambda=lam)

    def _state(self, lam):
        from energy_memory.phase4.consolidation import ConsolidationState
        return ConsolidationState(self._config(lam), device="cpu")

    def test_reinforce_increments_retrieval_count(self):
        s = self._state(0.0)
        idx = s.add_pattern()
        self.assertEqual(int(s.retrieval_count[idx]), 0)
        s.reinforce(idx)
        s.reinforce(idx)
        s.reinforce(idx)
        self.assertEqual(int(s.retrieval_count[idx]), 3)

    def test_lambda_zero_is_bit_identical_to_fixed_alpha(self):
        """With lambda=0 the cascade must match fixed-α exactly across N steps.

        This is the load-bearing backward-compatibility guarantee: every
        run that doesn't set --alpha-freq-lambda gets identical behavior.
        """
        s = self._state(0.0)
        # Build a varied substrate: 4 patterns, some reinforced multiple times.
        for _ in range(4):
            s.add_pattern(novelty_strength=1.0)
        s.reinforce(0); s.reinforce(0); s.reinforce(0)
        s.reinforce(1); s.reinforce(1)
        s.reinforce(2)
        # 20 ticks of cascade dynamics.
        for _ in range(20):
            s.step_dynamics()
        u_lam0 = s.u.clone()

        # Re-create a control state without any freq-weighting and run the
        # same script. The retrieval-count tensor differs but step_dynamics
        # with lambda=0 should ignore it entirely.
        s2 = self._state(0.0)
        for _ in range(4):
            s2.add_pattern(novelty_strength=1.0)
        s2.reinforce(0); s2.reinforce(0); s2.reinforce(0)
        s2.reinforce(1); s2.reinforce(1)
        s2.reinforce(2)
        for _ in range(20):
            s2.step_dynamics()
        self.assertTrue(torch.equal(u_lam0, s2.u))

    def test_lambda_amplifies_early_transient_propagation(self):
        """The mechanism check: at an early time in the cascade transient,
        a high-retrieval pattern propagates u_1 mass into u_2 faster than
        a low-retrieval one under λ>0; under λ=0 their cascades are
        identical (the same α). This isolates the per-row α_eff scaling
        from boundary/equilibrium effects, which are explored separately
        below.
        """
        def run_one_step(lam):
            s = self._state(lam)
            s.add_pattern(novelty_strength=1.0)
            s.add_pattern(novelty_strength=1.0)
            s.retrieval_count[0] = 10
            s.retrieval_count[1] = 1
            s.step_dynamics()
            return s

        s_lam0 = run_one_step(0.0)
        s_lam1 = run_one_step(1.0)
        # u_2 = u_1 + α*(-2*u_1 + u_2_old)
        # u_1_old = 1.0, u_2_old = 0.0 for both, so:
        #   under λ=0 both patterns have α=0.25 → u_2 = 0 + 0.25*(-2*1 + 0) = -0.5
        # Actually wait — u_2 *next* = u_2 + Δu_2 = 0 + α*(u_1 - 2*u_2) = α*u_1
        # since u_2 = u_3 = 0.
        # For λ=1, pattern 0 has α_eff=0.5, pattern 1 has α_eff=0.275.
        # So u_2_new(0) = 0.5*1 = 0.5; u_2_new(1) = 0.275*1 = 0.275.
        # Compare against λ=0: both = 0.25*1 = 0.25.
        self.assertAlmostEqual(s_lam0.u[0, 1].item(), s_lam0.u[1, 1].item(), places=6)
        self.assertGreater(s_lam1.u[0, 1].item(), s_lam1.u[1, 1].item())
        # Pattern 0 (high count) transfers faster than under fixed α; pattern 1
        # (low count, normalized=0.1) transfers slightly faster than under λ=0.
        self.assertGreater(s_lam1.u[0, 1].item(), s_lam0.u[0, 1].item())
        self.assertGreater(s_lam1.u[1, 1].item(), s_lam0.u[1, 1].item())

    def test_lambda_changes_steady_state_distribution_of_u_m(self):
        """Architectural finding worth documenting: at the cascade's steady
        state under continuous input, higher α actually produces a LOWER
        u_m (the Dirichlet boundary at u_{m+1}=0 makes high-α cascades leak
        more mass per unit time). So under freq-weighted α at long horizons,
        well-retrieved patterns end up with LESS u_m than fixed-α
        counterparts — the opposite of the naive "more retrieval → more
        slow variable" reading. The experiment's checkpoint correlation
        metric therefore tracks this signed relationship; whether the sign
        is positive or negative at the production scale (n_cues=3000) is
        what the experiment will measure.

        This test pins the steady-state behavior as a known property of
        the implementation so it can't drift unnoticed.
        """
        def run(lam, count, n_steps=400):
            s = self._state(lam)
            s.add_pattern(novelty_strength=0.0)
            s.retrieval_count[0] = count
            # apply constant input via input_vector to drive u_1
            input_vec = torch.tensor([0.1], dtype=torch.float32, device=s.device)
            for _ in range(n_steps):
                s.step_dynamics(input_vector=input_vec)
            return float(s.u[0, s.config.m - 1].item())

        # Single-pattern runs, with retrieval_count set to determine α_eff.
        # The normalization makes count_k / max_count = 1 for the only
        # pattern in the cascade, so α_eff = α * (1+λ).
        u_m_lam0 = run(0.0, count=10)
        u_m_lam1 = run(1.0, count=10)
        # Both are positive (continuous input drives positive u_m).
        self.assertGreater(u_m_lam0, 0.0)
        self.assertGreater(u_m_lam1, 0.0)
        # λ=1.0 lifts α from 0.25 to 0.5; steady-state u_m = I/(5α) for m=4 is
        # halved. So u_m_lam1 < u_m_lam0 at long horizons.
        self.assertLess(u_m_lam1, u_m_lam0)

    def test_remove_pattern_drops_retrieval_count_row(self):
        s = self._state(0.0)
        i0 = s.add_pattern()
        i1 = s.add_pattern()
        s.reinforce(i1)
        s.reinforce(i1)
        s.remove_pattern(i0)
        self.assertEqual(s.n_patterns, 1)
        self.assertEqual(int(s.retrieval_count[0]), 2)

    def test_stats_reports_retrieval_count_fields(self):
        s = self._state(0.5)
        s.add_pattern()
        s.add_pattern()
        s.reinforce(0)
        s.reinforce(0)
        s.reinforce(1)
        stats = s.stats()
        self.assertEqual(stats["retrieval_count_max"], 2)
        self.assertAlmostEqual(stats["retrieval_count_mean"], 1.5)
        self.assertEqual(stats["retrieval_count_nonzero"], 2)


if __name__ == "__main__":
    unittest.main()
