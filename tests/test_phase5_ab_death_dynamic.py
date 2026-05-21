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

    def test_sparse_duplicate_against_orthogonal_substrate(self):
        """A1' regression: a single duplicate among many orthogonal atoms
        should yield r_i ≈ 1 for the two duplicate atoms and r_i small
        for the rest. The earlier mean-RMS reduction gave r_i ≈ 0.03 for
        sparse duplicates (the report-048 failure mode).
        """
        from energy_memory.phase4.consolidation import (
            _coverage_redundancy_instantaneous,
        )
        # 30 random unit phasors + 1 duplicate of the first one
        orth = self._random_unit_phasors(n=30, d=4096, seed=23)
        patterns = torch.cat([orth, orth[:1]], dim=0)  # 31 atoms total
        r = _coverage_redundancy_instantaneous(patterns)
        self.assertEqual(r.shape, (31,))
        # Atom 0 and atom 30 are duplicates: both should have r ≈ 1.
        self.assertGreater(float(r[0]), 0.95)
        self.assertGreater(float(r[30]), 0.95)
        # Other atoms (1..29) are orthogonal to everyone; max similarity
        # to any neighbor is still small (~0.01-0.05 at D=4096).
        non_duplicate_r = torch.cat([r[1:30]])
        self.assertLess(float(non_duplicate_r.max()), 0.20)


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

    def test_step3_bias_is_zero_for_high_strength_atoms(self):
        """At E_i >> ε, the sigmoid saturates near 1 → bias ≈ 0."""
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(
            m=4, alpha=0.25, coverage_lambda=1.0,
            retrieval_weight_epsilon=0.05, retrieval_weight_tau=0.02,
        )
        state = ConsolidationState(cfg, device="cpu")
        state.add_pattern(novelty_strength=1.0)  # strong atom
        bias = state.retrieval_weight_bias()
        # |E_i| ≈ 1.0 >> ε=0.05; (ε - |E|)/τ = -47.5 → softplus ≈ 0
        self.assertLess(float(bias[0]), 1e-10)

    def test_step3_bias_grows_for_low_strength_atoms(self):
        """At E_i << ε, bias = (ε − E)/τ (linear regime of softplus)."""
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(
            m=4, alpha=0.25, coverage_lambda=1.0,
            retrieval_weight_epsilon=0.05, retrieval_weight_tau=0.02,
        )
        state = ConsolidationState(cfg, device="cpu")
        state.add_pattern(novelty_strength=0.0)  # dead atom (E=0)
        bias = state.retrieval_weight_bias()
        # (ε - 0)/τ = 2.5 → softplus(2.5) ≈ 2.578
        self.assertAlmostEqual(float(bias[0]), 2.578, places=2)

    def test_step3_bias_at_epsilon_equals_log2(self):
        """At E_i = ε exactly, sigmoid = 0.5 → bias = log(2) ≈ 0.693."""
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(
            m=4, alpha=0.25, coverage_lambda=1.0,
            retrieval_weight_epsilon=0.05, retrieval_weight_tau=0.02,
        )
        state = ConsolidationState(cfg, device="cpu")
        # u-chain weights: 2^(1-k) starting at 1.0 for k=1. With m=4:
        # effective_strength = u_1·1 + u_2·0.5 + u_3·0.25 + u_4·0.125
        # Setting u_1 = 0.05 (and others 0) gives effective_strength = 0.05.
        state.add_pattern(novelty_strength=0.0)
        state.u[0, 0] = 0.05
        bias = state.retrieval_weight_bias()
        self.assertAlmostEqual(float(bias[0]), math.log(2.0), places=4)

    def test_step3_softmax_weight_ratio(self):
        """Dead atom vs alive atom: softmax weight ratio after applying
        step-3 bias should be ≈ 12× (the design's target separation for
        ε=0.05, τ=0.02 at population median).
        """
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(
            m=4, alpha=0.25, coverage_lambda=1.0,
            retrieval_weight_epsilon=0.05, retrieval_weight_tau=0.02,
        )
        state = ConsolidationState(cfg, device="cpu")
        state.add_pattern(novelty_strength=0.0)
        state.add_pattern(novelty_strength=0.0)
        state.u[0, 0] = 0.0      # dead
        state.u[1, 0] = 0.1      # alive (2× epsilon)
        bias = state.retrieval_weight_bias()
        # Equal-similarity baseline: pretend both atoms have score = 0.
        # Effective log-prob differs by bias[1] − bias[0]; softmax ratio is
        # exp(bias[0] − bias[1]) (the dead atom is HEAVILY downweighted).
        log_ratio_alive_to_dead = float(bias[0] - bias[1])
        ratio = math.exp(log_ratio_alive_to_dead)
        # σ(2.5) / σ(-2.5) = 0.924 / 0.076 ≈ 12.2 → log ratio ≈ 2.5
        self.assertGreater(ratio, 8.0)
        self.assertLess(ratio, 20.0)

    def test_step3_replay_loop_passes_bias_when_coverage_on(self):
        """End-to-end: with coverage_lambda > 0, retrieve_and_observe()
        consults retrieval_weight_bias and a low-strength atom is
        suppressed in the retrieval result.
        """
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        from energy_memory.phase4.replay_loop import (
            ReplayConfig, UnifiedReplayMemory,
        )
        from energy_memory.phase4.trajectory import TracedHopfieldMemory
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        substrate = TorchFHRR(dim=512, seed=17, device="cpu")
        memory = TracedHopfieldMemory(substrate)
        # Two orthogonal patterns
        gen = torch.Generator(device="cpu").manual_seed(3)
        for _ in range(2):
            phase = torch.rand((512,), generator=gen) * (2.0 * math.pi)
            memory.store(torch.polar(torch.ones((512,)), phase))
        cfg = ConsolidationConfig(
            m=4, alpha=0.25, coverage_lambda=1.0,
            retrieval_weight_epsilon=0.05, retrieval_weight_tau=0.02,
        )
        cons = ConsolidationState(cfg, device="cpu")
        replay = UnifiedReplayMemory(
            substrate=substrate, memory=memory, consolidation=cons,
            config=ReplayConfig(),
        )
        replay.attach_initial_patterns()
        # Make atom 0 alive (E=0.5) and atom 1 dead (E=0)
        cons.u[0, 0] = 0.5
        cons.u[1, 0] = 0.0

        # Query equally similar to both (the mean) → without step-3 the
        # retrieval would split; with step-3 it should heavily prefer
        # atom 0.
        avg = substrate.normalize(
            memory._patterns[0] + memory._patterns[1]
        )
        result, _ = replay.retrieve_and_observe(avg, beta=10.0)
        # weights[0] should dominate weights[1] by an order of magnitude
        # after step-3 biasing.
        self.assertGreater(result.weights[0], 5.0 * result.weights[1])

    def test_step3_off_when_coverage_lambda_zero(self):
        """coverage_lambda=0 → no step-3 bias even if E_i are small."""
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        from energy_memory.phase4.replay_loop import (
            ReplayConfig, UnifiedReplayMemory,
        )
        from energy_memory.phase4.trajectory import TracedHopfieldMemory
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        substrate = TorchFHRR(dim=512, seed=17, device="cpu")
        memory = TracedHopfieldMemory(substrate)
        gen = torch.Generator(device="cpu").manual_seed(3)
        for _ in range(2):
            phase = torch.rand((512,), generator=gen) * (2.0 * math.pi)
            memory.store(torch.polar(torch.ones((512,)), phase))
        cfg = ConsolidationConfig(m=4, alpha=0.25, coverage_lambda=0.0)
        cons = ConsolidationState(cfg, device="cpu")
        replay = UnifiedReplayMemory(
            substrate=substrate, memory=memory, consolidation=cons,
            config=ReplayConfig(),
        )
        replay.attach_initial_patterns()
        cons.u[0, 0] = 0.0  # dead atom — would be suppressed if step 3 fired

        # _score_bias should return None
        self.assertIsNone(replay._score_bias())

    def test_a_active_updates_r_ema_through_replay_cycle(self):
        substrate, memory, replay = self._build(
            alpha_anti=0.0, coverage_lambda=1.0, repulsion_step=0.0,
        )
        for _ in range(5):
            replay.run_replay_cycle()
        # 8 mixtures of 2 bases → high pairwise |G_ij| → r_ema rises
        # from 0 toward saturation at rate 0.5 per cycle.
        self.assertGreater(float(replay.consolidation.r_ema.min()), 0.3)


class TestA1DiscoveryChannelInit(unittest.TestCase):
    """A1: ``r_ema`` initialized at the EMA's geometric equilibrium for
    each new atom at add-time. See
    notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md.
    """

    def _build_landscape(self, kind: str, *, coverage_lambda: float):
        """Build a replay system with one of two landscape shapes.

        kind="orthogonal": 8 random independent FHRR patterns (low
            pairwise |G_ij|, so r_inst per atom ≈ 0).
        kind="duplicates": 8 patterns that are all near-copies of a
            single base (high pairwise |G_ij|, so r_inst ≈ 1).
        """
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        from energy_memory.phase4.replay_loop import (
            ReplayConfig, UnifiedReplayMemory,
        )
        from energy_memory.phase4.trajectory import TracedHopfieldMemory
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        substrate = TorchFHRR(dim=256, seed=17, device="cpu")
        memory = TracedHopfieldMemory(substrate)
        cons = ConsolidationState(
            ConsolidationConfig(
                m=4, alpha=0.25,
                coverage_lambda=coverage_lambda,
                coverage_ema_rate=0.01,
            ),
            device="cpu",
        )
        replay = UnifiedReplayMemory(
            substrate=substrate, memory=memory, consolidation=cons,
            config=ReplayConfig(),
        )
        gen = torch.Generator(device="cpu").manual_seed(43)
        if kind == "orthogonal":
            for _ in range(8):
                phases = torch.rand((substrate.dim,), generator=gen) * (2.0 * math.pi)
                memory.store(torch.polar(torch.ones((substrate.dim,)), phases))
        elif kind == "duplicates":
            base_phases = torch.rand((substrate.dim,), generator=gen) * (2.0 * math.pi)
            base = torch.polar(torch.ones((substrate.dim,)), base_phases)
            for _ in range(8):
                # Tiny phase noise so patterns aren't FP-identical but
                # remain mutually nearly-collinear (|G_ij| → 1).
                noise = (torch.rand((substrate.dim,), generator=gen) - 0.5) * 0.02
                memory.store(substrate.normalize(
                    torch.polar(torch.ones((substrate.dim,)), base_phases + noise)
                ))
        else:
            raise ValueError(kind)
        return substrate, memory, replay

    def test_attach_initial_orthogonal_landscape_starts_at_zero(self):
        """Independent patterns ⇒ r_inst ≈ 0 per atom ⇒ A1 init ≈ 0."""
        substrate, memory, replay = self._build_landscape(
            "orthogonal", coverage_lambda=1.0,
        )
        replay.attach_initial_patterns()
        r_ema = replay.consolidation.r_ema
        self.assertEqual(r_ema.shape[0], 8)
        self.assertLess(float(r_ema.max()), 0.15)

    def test_attach_initial_duplicate_landscape_starts_near_one(self):
        """Near-collinear patterns ⇒ r_inst ≈ 1 per atom ⇒ A1 init ≈ 1.
        The pre-A1 init (zero) would have left every atom at 0 here.
        """
        substrate, memory, replay = self._build_landscape(
            "duplicates", coverage_lambda=1.0,
        )
        replay.attach_initial_patterns()
        r_ema = replay.consolidation.r_ema
        self.assertEqual(r_ema.shape[0], 8)
        self.assertGreater(float(r_ema.min()), 0.95)

    def test_coverage_lambda_zero_preserves_zero_init(self):
        """A1 is gated on coverage_lambda > 0. With A off, r_ema is
        still allocated as zeros (default-off behavior preserved).
        """
        substrate, memory, replay = self._build_landscape(
            "duplicates", coverage_lambda=0.0,
        )
        replay.attach_initial_patterns()
        r_ema = replay.consolidation.r_ema
        self.assertEqual(r_ema.shape[0], 8)
        self.assertEqual(float(r_ema.max()), 0.0)

    def test_discovery_channel_duplicate_atom_gets_high_r_ema_immediately(self):
        """End-to-end wiring: when the discovery channel adds a near-
        duplicate atom on a collapsed landscape, the new atom's r_ema
        is high *at add-time*, not zero waiting ~100 EMA steps to relax.
        This is the A+B+step3 substrate failure report 047 documents.
        """
        substrate, memory, replay = self._build_landscape(
            "duplicates", coverage_lambda=1.0,
        )
        replay.attach_initial_patterns()
        # Simulate the discovery channel: add one more near-duplicate
        # atom by mimicking what candidate_handler would do (store new
        # pattern into memory, then walk consolidation forward via the
        # A1-wired path).
        gen = torch.Generator(device="cpu").manual_seed(99)
        base_phases = torch.angle(memory._patterns[0])
        noise = (torch.rand((substrate.dim,), generator=gen) - 0.5) * 0.02
        duplicate = substrate.normalize(
            torch.polar(torch.ones((substrate.dim,)), base_phases + noise)
        )
        memory.store(duplicate)
        new_idx = memory.stored_count - 1
        # Replicate the discovery-channel A1 wiring (same code path as
        # replay_loop.run_replay_cycle when candidate_handler returns).
        r_inst = replay._compute_r_inst_for_init()
        while replay.consolidation.n_patterns <= new_idx:
            next_idx = replay.consolidation.n_patterns
            r_init = (
                None if r_inst is None
                else float(r_inst[next_idx].detach().cpu())
            )
            replay.consolidation.add_pattern(
                novelty_strength=replay.config.novelty_strength,
                r_ema_init=r_init,
            )
        new_r_ema = float(replay.consolidation.r_ema[new_idx])
        self.assertGreater(new_r_ema, 0.95)


if __name__ == "__main__":
    unittest.main()
