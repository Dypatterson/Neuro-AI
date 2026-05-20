"""Tests for the A+B death-mechanism dynamic-form re-expression.

A: continuous coverage-weighted reinforcement rate in
   ``energy_memory.phase4.consolidation``.
B: ``H_anti = -α · log(d_eff)`` repulsion field in
   ``energy_memory.substrate.torch_fhrr``.

Design note:
notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md
"""

from __future__ import annotations

import math
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestCoverageRedundancyInstantaneous(unittest.TestCase):
    """``_coverage_redundancy_instantaneous`` (Candidate A's r_i proxy)."""

    def _random_unit_phasors(self, n, d, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        phase = torch.rand((n, d), generator=gen) * (2.0 * math.pi)
        return torch.polar(torch.ones((n, d)), phase)

    def test_orthogonal_patterns_have_low_redundancy(self):
        from energy_memory.phase4.consolidation import (
            _coverage_redundancy_instantaneous,
        )
        patterns = self._random_unit_phasors(n=8, d=4096, seed=17)
        r = _coverage_redundancy_instantaneous(patterns)
        self.assertEqual(r.shape, (8,))
        # Random high-dim FHRR vectors: |G_ij|^2 ~ 1/D, so r ~ sqrt(1/D)
        # ~= 0.016 for D=4096. Should be far below 0.1.
        self.assertLess(float(r.max()), 0.1)

    def test_duplicate_patterns_have_high_redundancy(self):
        from energy_memory.phase4.consolidation import (
            _coverage_redundancy_instantaneous,
        )
        base = self._random_unit_phasors(n=1, d=4096, seed=11)
        patterns = base.repeat(5, 1)  # all identical
        r = _coverage_redundancy_instantaneous(patterns)
        # Identical atoms: every off-diagonal |G_ij| = 1, so r = 1.
        for i in range(5):
            self.assertAlmostEqual(float(r[i]), 1.0, places=4)

    def test_single_pattern_returns_zero(self):
        from energy_memory.phase4.consolidation import (
            _coverage_redundancy_instantaneous,
        )
        patterns = self._random_unit_phasors(n=1, d=4096, seed=3)
        r = _coverage_redundancy_instantaneous(patterns)
        self.assertEqual(r.shape, (1,))
        self.assertEqual(float(r[0]), 0.0)


@unittest.skipIf(torch is None, "torch required")
class TestCandidateAReinforcement(unittest.TestCase):
    """Coverage-weighted reinforcement modulation in ``reinforce()``."""

    def test_coverage_lambda_zero_preserves_baseline_reinforce(self):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(m=4, alpha=0.25, coverage_lambda=0.0)
        state = ConsolidationState(cfg, device="cpu")
        state.add_pattern(novelty_strength=1.0)
        # Manually set r_ema high — coverage_lambda=0 should ignore it
        state.r_ema[0] = 0.95
        before = state.u[0, 0].item()
        state.reinforce(0, magnitude=0.5)
        self.assertAlmostEqual(state.u[0, 0].item(), before + 0.5, places=6)

    def test_high_r_ema_scales_reinforcement_down(self):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(m=4, alpha=0.25, coverage_lambda=1.0)
        state = ConsolidationState(cfg, device="cpu")
        state.add_pattern(novelty_strength=0.0)
        state.add_pattern(novelty_strength=0.0)
        state.r_ema[0] = 0.0  # novel
        state.r_ema[1] = 0.9  # redundant

        state.reinforce(0, magnitude=1.0)
        state.reinforce(1, magnitude=1.0)
        self.assertAlmostEqual(state.u[0, 0].item(), 1.0, places=6)
        # Reinforcement scaled by (1 - 1.0 * 0.9) = 0.1
        self.assertAlmostEqual(state.u[1, 0].item(), 0.1, places=6)

    def test_remove_pattern_keeps_r_ema_aligned(self):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(m=4, alpha=0.25)
        state = ConsolidationState(cfg, device="cpu")
        for i in range(4):
            state.add_pattern(novelty_strength=1.0)
            state.r_ema[i] = 0.1 * (i + 1)
        state.remove_pattern(1)
        self.assertEqual(state.n_patterns, 3)
        self.assertEqual(state.r_ema.shape, (3,))
        self.assertAlmostEqual(float(state.r_ema[0]), 0.1)
        self.assertAlmostEqual(float(state.r_ema[1]), 0.3)  # was idx 2
        self.assertAlmostEqual(float(state.r_ema[2]), 0.4)  # was idx 3

    def test_step_dynamics_ignores_pattern_matrix_when_lambda_zero(self):
        """At coverage_lambda=0, r_ema must stay at zero even if pattern_matrix
        is supplied — preserves the "off by default, bit-identical" guarantee.
        """
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(m=4, alpha=0.25, coverage_lambda=0.0)
        state = ConsolidationState(cfg, device="cpu")
        state.add_pattern(novelty_strength=1.0)
        state.add_pattern(novelty_strength=1.0)
        # Two identical patterns — would push r_ema to 1 if mechanism were on.
        gen = torch.Generator(device="cpu").manual_seed(5)
        phase = torch.rand((1, 4096), generator=gen) * (2.0 * math.pi)
        base = torch.polar(torch.ones((1, 4096)), phase)
        patterns = base.repeat(2, 1)
        for _ in range(50):
            state.step_dynamics(pattern_matrix=patterns)
        self.assertAlmostEqual(float(state.r_ema.max()), 0.0, places=6)

    def test_step_dynamics_drives_r_ema_toward_r_inst(self):
        """At coverage_lambda>0, r_ema should approach the instantaneous r_i
        with the prescribed EMA timescale.
        """
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
            _coverage_redundancy_instantaneous,
        )
        cfg = ConsolidationConfig(
            m=4, alpha=0.25,
            coverage_lambda=1.0, coverage_ema_rate=0.1,
        )
        state = ConsolidationState(cfg, device="cpu")
        # Two identical patterns: r_inst → 1 on both rows.
        gen = torch.Generator(device="cpu").manual_seed(7)
        phase = torch.rand((1, 4096), generator=gen) * (2.0 * math.pi)
        base = torch.polar(torch.ones((1, 4096)), phase)
        patterns = base.repeat(2, 1)
        state.add_pattern(novelty_strength=1.0)
        state.add_pattern(novelty_strength=1.0)
        r_inst = _coverage_redundancy_instantaneous(patterns)
        self.assertAlmostEqual(float(r_inst.max()), 1.0, places=4)
        for _ in range(200):
            state.step_dynamics(pattern_matrix=patterns)
        # After 200 steps at η=0.1, (1-0.1)^200 ≈ 7e-10 of the original
        # distance remains. r_ema should be very close to 1.
        self.assertGreater(float(state.r_ema.min()), 0.99)


@unittest.skipIf(torch is None, "torch required")
class TestCandidateBSubstrateEnergy(unittest.TestCase):
    """``substrate_energy_anti`` and ``repulsion_force`` on TorchFHRR."""

    def _substrate(self, alpha_anti=1.0, seed=17):
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        return TorchFHRR(dim=512, seed=seed, device="cpu", alpha_anti=alpha_anti)

    def _random_patterns(self, substrate, n, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        phase = torch.rand((n, substrate.dim), generator=gen) * (2.0 * math.pi)
        return torch.polar(torch.ones((n, substrate.dim)), phase).to(substrate.device)

    def test_alpha_anti_zero_returns_zero_energy(self):
        substrate = self._substrate(alpha_anti=0.0)
        patterns = self._random_patterns(substrate, n=8, seed=1)
        e = substrate.substrate_energy_anti(patterns)
        self.assertEqual(float(e), 0.0)
        # Force should also be zero — caller should be able to add it
        # unconditionally without producing NaN gradients.
        f = substrate.repulsion_force(patterns)
        self.assertEqual(f.shape, patterns.shape)
        self.assertEqual(float(f.abs().max()), 0.0)

    def test_d_eff_higher_means_lower_energy(self):
        """H_anti = -α log(d_eff): spread (high d_eff) is low-energy,
        collapse (low d_eff) is high-energy.
        """
        substrate = self._substrate(alpha_anti=1.0, seed=23)
        # Spread substrate: 16 independent random patterns
        spread = self._random_patterns(substrate, n=16, seed=23)
        # Collapsed substrate: 16 copies of a single random pattern
        base = self._random_patterns(substrate, n=1, seed=23)
        collapsed = base.repeat(16, 1)
        e_spread = float(substrate.substrate_energy_anti(spread))
        e_collapsed = float(substrate.substrate_energy_anti(collapsed))
        self.assertLess(e_spread, e_collapsed)
        # Sanity: spread d_eff >> collapsed d_eff
        d_spread = float(substrate.d_eff(spread))
        d_collapsed = float(substrate.d_eff(collapsed))
        self.assertGreater(d_spread, d_collapsed)

    def test_one_repulsion_step_increases_d_eff(self):
        """Gradient descent on H_anti = -α log(d_eff) must increase d_eff."""
        substrate = self._substrate(alpha_anti=1.0, seed=11)
        # Start truly low-d_eff: 8 random mixtures of 2 random bases all
        # lie in a 2-D subspace, so d_eff ≈ 1–2 (not 8).
        gen = torch.Generator(device="cpu").manual_seed(33)
        A = torch.polar(
            torch.ones((substrate.dim,)),
            torch.rand((substrate.dim,), generator=gen) * (2.0 * math.pi),
        )
        B = torch.polar(
            torch.ones((substrate.dim,)),
            torch.rand((substrate.dim,), generator=gen) * (2.0 * math.pi),
        )
        weights = torch.rand((8,), generator=gen)
        patterns = torch.stack([
            substrate.normalize(w * A + (1.0 - w) * B) for w in weights
        ]).to(substrate.device)
        d0 = float(substrate.d_eff(patterns))
        self.assertLess(d0, 3.0)  # confirm we started in a collapsed regime

        force = substrate.repulsion_force(patterns)
        new_patterns = substrate.normalize(patterns + 100.0 * force)
        d1 = float(substrate.d_eff(new_patterns))
        self.assertGreater(d1, d0)

    def test_repulsion_force_preserves_complex_dtype_and_shape(self):
        substrate = self._substrate(alpha_anti=1.0, seed=1)
        patterns = self._random_patterns(substrate, n=4, seed=2)
        force = substrate.repulsion_force(patterns)
        self.assertEqual(force.shape, patterns.shape)
        self.assertTrue(force.is_complex())


@unittest.skipIf(torch is None, "torch required")
class TestUnifiedReplayMemoryWiring(unittest.TestCase):
    """A and B activate together in ``run_replay_cycle``."""

    def _build(self, alpha_anti=0.0, coverage_lambda=0.0, repulsion_step=0.0):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        from energy_memory.phase4.replay_loop import (
            ReplayConfig, UnifiedReplayMemory,
        )
        from energy_memory.phase4.trajectory import TracedHopfieldMemory
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        substrate = TorchFHRR(
            dim=256, seed=17, device="cpu", alpha_anti=alpha_anti,
        )
        # TracedHopfieldMemory inherits from TorchHopfieldMemory — it IS the
        # storage layer, not a wrapper around one.
        memory = TracedHopfieldMemory(substrate)
        cons = ConsolidationState(
            ConsolidationConfig(
                m=4, alpha=0.25,
                coverage_lambda=coverage_lambda,
                coverage_ema_rate=0.5,
            ),
            device="cpu",
        )
        replay = UnifiedReplayMemory(
            substrate=substrate, memory=memory, consolidation=cons,
            config=ReplayConfig(repulsion_step_size=repulsion_step),
        )
        # 8 random mixtures of 2 bases → all live in a 2-D subspace, so
        # d_eff ≈ 1–2 to start (a real collapsed regime, not just "close
        # to a single point").
        gen = torch.Generator(device="cpu").manual_seed(101)
        A = torch.polar(
            torch.ones((substrate.dim,)),
            torch.rand((substrate.dim,), generator=gen) * (2.0 * math.pi),
        )
        B = torch.polar(
            torch.ones((substrate.dim,)),
            torch.rand((substrate.dim,), generator=gen) * (2.0 * math.pi),
        )
        for w in torch.rand((8,), generator=gen):
            memory.store(substrate.normalize(w * A + (1.0 - w) * B))
        replay.attach_initial_patterns()
        return substrate, memory, replay

    def test_default_config_does_not_mutate_patterns(self):
        substrate, memory, replay = self._build(
            alpha_anti=0.0, coverage_lambda=0.0, repulsion_step=0.0,
        )
        before = torch.stack(memory._patterns).clone()
        for _ in range(3):
            replay.run_replay_cycle()
        after = torch.stack(memory._patterns)
        self.assertTrue(torch.allclose(before, after))

    def test_ab_active_increases_d_eff_across_cycles(self):
        substrate, memory, replay = self._build(
            alpha_anti=1.0, coverage_lambda=1.0, repulsion_step=50.0,
        )
        d0 = float(substrate.d_eff(memory._pattern_matrix()))
        for _ in range(20):
            replay.run_replay_cycle()
        d1 = float(substrate.d_eff(memory._pattern_matrix()))
        self.assertGreater(d1, d0)

    def test_garbage_collect_noops_when_coverage_lambda_on(self):
        """A+B asymptotic decay replaces binary deletion. The existing
        ``garbage_collect()`` primitive must not fire when A+B is on,
        or the binary controller A+B replaces would run alongside it.
        """
        substrate, memory, replay = self._build(
            alpha_anti=0.0, coverage_lambda=1.0, repulsion_step=0.0,
        )
        # Force one pattern past the binary death threshold by zeroing
        # its u-chain and aging the below-threshold counter past death_window.
        cons = replay.consolidation
        cons.u[0, :] = 0.0
        cons.below_threshold_steps[0] = cons.config.death_window + 1
        # dead_indices still detects the pattern (it is a measurement)…
        self.assertIn(0, cons.dead_indices())
        # …but garbage_collect is suppressed when A+B is on.
        removed = replay.garbage_collect()
        self.assertEqual(removed, [])
        self.assertEqual(memory.stored_count, 8)

    def test_garbage_collect_still_fires_when_a_b_off(self):
        """Non-regression: with A+B off, binary death is unchanged."""
        substrate, memory, replay = self._build(
            alpha_anti=0.0, coverage_lambda=0.0, repulsion_step=0.0,
        )
        cons = replay.consolidation
        cons.u[0, :] = 0.0
        cons.below_threshold_steps[0] = cons.config.death_window + 1
        removed = replay.garbage_collect()
        self.assertEqual(removed, [0])
        self.assertEqual(memory.stored_count, 7)

    def test_a_active_updates_r_ema_through_replay_cycle(self):
        substrate, memory, replay = self._build(
            alpha_anti=0.0, coverage_lambda=1.0, repulsion_step=0.0,
        )
        for _ in range(5):
            replay.run_replay_cycle()
        # 8 mixtures of 2 bases → high pairwise |G_ij| → r_ema rises
        # from 0 toward saturation at rate 0.5 per cycle.
        self.assertGreater(float(replay.consolidation.r_ema.min()), 0.3)


if __name__ == "__main__":
    unittest.main()
