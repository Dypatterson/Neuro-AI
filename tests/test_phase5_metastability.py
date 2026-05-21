"""Tests for pair #4 — metastability ~ replay-prioritization.

Design note:
notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md

Mechanism:
  c_i = w_i · (1 − max_j w_j)                  (per-retrieval contribution)
  m_i ← (1 − μ_obs) · m_i + μ_obs · c_i        (slow EMA)
  m_i ← (1 − μ_rep) · m_i  on replay sampling  (pay-down)

Priority composition gains a (1 + κ · m_trace) factor where m_trace is
the metastability EMA of the trace's primary atom (i.e. the retrieval's
top index).
"""

from __future__ import annotations

import math
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestMetastabilityEMAUpdate(unittest.TestCase):
    """update_metastability(weights) applies c_i = w_i · (1 − max_w) into the EMA."""

    def _state(self, n_patterns, obs_rate):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(m=4, alpha=0.25, metastability_obs_rate=obs_rate)
        state = ConsolidationState(cfg, device="cpu")
        for _ in range(n_patterns):
            state.add_pattern(novelty_strength=1.0)
        return state

    def test_obs_rate_zero_is_noop(self):
        """The κ=0 control depends on m_i staying at zero when obs_rate=0."""
        state = self._state(n_patterns=4, obs_rate=0.0)
        # Anything but zero in weights would push m_i if the mechanism fired.
        weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
        state.update_metastability(weights)
        self.assertTrue(torch.all(state.metastability_ema == 0.0))

    def test_sharp_retrieval_produces_zero_metastability(self):
        """A near-one-hot weights vector gives c_i ≈ 0 for every atom."""
        state = self._state(n_patterns=4, obs_rate=0.5)
        weights = torch.tensor([0.99, 0.005, 0.003, 0.002])
        state.update_metastability(weights)
        # c_0 = 0.99 * (1 - 0.99) = 0.0099; c_1 = 0.005 * 0.01 = 5e-5; …
        # After one EMA step at μ_obs=0.5 from m_i=0: m_i = 0.5 * c_i.
        self.assertLess(float(state.metastability_ema[0]), 0.01)
        self.assertLess(float(state.metastability_ema[1]), 0.001)
        self.assertLess(float(state.metastability_ema[2]), 0.001)

    def test_diffuse_retrieval_produces_positive_metastability(self):
        """A tied retrieval gives c_i > 0 for every contributing atom."""
        state = self._state(n_patterns=4, obs_rate=0.5)
        weights = torch.tensor([0.30, 0.28, 0.22, 0.20])
        state.update_metastability(weights)
        # max_w = 0.30; c_0 = 0.30 * 0.70 = 0.21; c_3 = 0.20 * 0.70 = 0.14.
        # After one EMA step at μ_obs=0.5 from m_i=0: m_i = 0.5 * c_i.
        for i in range(4):
            self.assertGreater(float(state.metastability_ema[i]), 0.05)
        # Higher-weight atoms accumulate more metastability than lower ones.
        self.assertGreater(
            float(state.metastability_ema[0]),
            float(state.metastability_ema[3]),
        )

    def test_ema_blends_across_calls(self):
        """m_i follows the canonical EMA recurrence across multiple calls."""
        state = self._state(n_patterns=2, obs_rate=0.5)
        # Two identical retrievals; c_0 = 0.5 * (1 - 0.5) = 0.25 each call.
        weights = torch.tensor([0.5, 0.5])
        state.update_metastability(weights)
        # After 1 call: m_0 = 0.5 * 0.25 = 0.125
        self.assertAlmostEqual(float(state.metastability_ema[0]), 0.125, places=5)
        state.update_metastability(weights)
        # After 2 calls: m_0 = (1-0.5) * 0.125 + 0.5 * 0.25 = 0.1875
        self.assertAlmostEqual(float(state.metastability_ema[0]), 0.1875, places=5)

    def test_shape_mismatch_raises(self):
        state = self._state(n_patterns=4, obs_rate=0.5)
        with self.assertRaises(ValueError):
            state.update_metastability(torch.tensor([0.5, 0.5]))

    def test_new_atom_enters_with_zero_metastability(self):
        """Audit constraint #7: add_pattern() must not derive m_i from population stats."""
        state = self._state(n_patterns=3, obs_rate=0.5)
        # Drive m_i for the existing atoms to non-zero.
        state.update_metastability(torch.tensor([0.40, 0.35, 0.25]))
        # Now add a new atom — it must enter at m_i = 0.
        new_idx = state.add_pattern(novelty_strength=1.0)
        self.assertEqual(float(state.metastability_ema[new_idx]), 0.0)


@unittest.skipIf(torch is None, "torch required")
class TestMetastabilityPaybackOnSample(unittest.TestCase):
    """Sampling a trace pays down its primary atom's m_i by (1 − μ_rep)."""

    def _build(self, kappa=0.0, mu_rep=0.5):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        from energy_memory.phase4.replay_loop import ReplayStore
        from energy_memory.phase4.trajectory import TrajectoryTrace
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        substrate = TorchFHRR(dim=128, seed=7, device="cpu")
        cons = ConsolidationState(
            ConsolidationConfig(m=4, metastability_obs_rate=0.0),
            device="cpu",
        )
        for _ in range(4):
            cons.add_pattern(novelty_strength=1.0)
        # Seed m_i directly for atoms 0..3.
        cons.metastability_ema[:] = torch.tensor([0.20, 0.40, 0.10, 0.05])
        store = ReplayStore(
            capacity=10,
            consolidation=cons,
            metastability_gain=kappa,
            metastability_replay_decay=mu_rep,
        )
        # Build four traces, one per primary atom.
        for i in range(4):
            q = substrate.random_vector()
            trace = TrajectoryTrace(
                query=q, snapshots=[], final_state=q,
                final_top_score=1.0, final_top_index=i, converged=True,
            )
            store.add(trace, gate_signal=1.0, primary_atom_idx=i)
        return cons, store

    def test_pay_down_applied_on_sample(self):
        cons, store = self._build(kappa=0.0, mu_rep=0.5)
        before = cons.metastability_ema.clone()
        # Force sampling of trace 0 by giving its gate_signal much higher.
        store.gate_signals[0] = 1e6
        # Sample n=1; multinomial almost-certainly picks trace 0.
        gen = torch.Generator(device="cpu").manual_seed(11)
        sampled = store.sample(n=1, generator=gen)
        self.assertEqual(sampled, [0])
        # m_0 should have been multiplied by (1 - 0.5) = 0.5.
        self.assertAlmostEqual(
            float(cons.metastability_ema[0]), float(before[0]) * 0.5, places=5,
        )
        # Non-sampled atoms untouched.
        for i in (1, 2, 3):
            self.assertAlmostEqual(
                float(cons.metastability_ema[i]), float(before[i]), places=6,
            )

    def test_pay_down_disabled_at_mu_rep_zero(self):
        cons, store = self._build(kappa=0.0, mu_rep=0.0)
        before = cons.metastability_ema.clone()
        store.gate_signals[0] = 1e6
        gen = torch.Generator(device="cpu").manual_seed(11)
        store.sample(n=1, generator=gen)
        # mu_rep=0 → no pay-down code fires; m_i unchanged.
        self.assertTrue(torch.allclose(cons.metastability_ema, before))


@unittest.skipIf(torch is None, "torch required")
class TestKappaZeroPreservesPriority(unittest.TestCase):
    """The critical κ=0 non-regression test.

    With metastability_gain = 0.0, the priority composition must be
    bit-identical to the pre-pivot baseline — this is the load-bearing
    precondition that makes the κ=0 control valid as a falsification
    baseline. Audit constraint #6.
    """

    def _build(self, kappa):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        from energy_memory.phase4.replay_loop import ReplayStore
        from energy_memory.phase4.trajectory import TrajectoryTrace
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        substrate = TorchFHRR(dim=128, seed=23, device="cpu")
        cons = ConsolidationState(
            ConsolidationConfig(m=4, metastability_obs_rate=0.5),
            device="cpu",
        )
        for _ in range(4):
            cons.add_pattern(novelty_strength=1.0)
        # Make m_i non-uniform so that κ != 0 would actually change priorities.
        cons.metastability_ema[:] = torch.tensor([0.05, 0.25, 0.15, 0.40])
        store = ReplayStore(
            capacity=10,
            consolidation=cons,
            metastability_gain=kappa,
            metastability_replay_decay=0.0,
        )
        for i in range(4):
            q = substrate.random_vector()
            trace = TrajectoryTrace(
                query=q, snapshots=[], final_state=q,
                final_top_score=1.0, final_top_index=i, converged=True,
            )
            store.add(trace, gate_signal=0.5 + 0.1 * i, primary_atom_idx=i)
        return store

    def test_kappa_zero_priority_bit_identical_to_baseline(self):
        store_kappa0 = self._build(kappa=0.0)
        priorities_kappa0 = store_kappa0._priorities()
        # Baseline: gate × tag × suppression for each trace (no metastability factor).
        expected = [
            store_kappa0.gate_signals[i] * store_kappa0.tag_counts[i] * store_kappa0.suppression[i]
            for i in range(len(store_kappa0.traces))
        ]
        self.assertEqual(priorities_kappa0, expected)

    def test_kappa_positive_changes_priorities(self):
        """Sanity: with κ > 0 and non-uniform m_i, priorities must shift."""
        store_kappa_pos = self._build(kappa=2.0)
        priorities_kappa_pos = store_kappa_pos._priorities()
        baseline = [
            store_kappa_pos.gate_signals[i] * store_kappa_pos.tag_counts[i] * store_kappa_pos.suppression[i]
            for i in range(len(store_kappa_pos.traces))
        ]
        # Each priority is multiplied by (1 + κ · m_i). Verify directly.
        expected = [
            baseline[i] * (1.0 + 2.0 * float(store_kappa_pos._consolidation.metastability_ema[i]))
            for i in range(len(baseline))
        ]
        for got, want in zip(priorities_kappa_pos, expected):
            self.assertAlmostEqual(got, want, places=5)


@unittest.skipIf(torch is None, "torch required")
class TestMetastabilityWiredIntoRetrieveAndObserve(unittest.TestCase):
    """End-to-end: UnifiedReplayMemory.retrieve_and_observe updates m_i.

    Verifies the design note's load-bearing claim that c_i is computed
    from the same softmax weights retrieve() already produces — no
    re-evaluation pass.
    """

    def _build(self, obs_rate, gain):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        from energy_memory.phase4.replay_loop import (
            ReplayConfig, UnifiedReplayMemory,
        )
        from energy_memory.phase4.trajectory import TracedHopfieldMemory
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        substrate = TorchFHRR(dim=256, seed=42, device="cpu")
        memory = TracedHopfieldMemory(substrate)
        cons = ConsolidationState(
            ConsolidationConfig(m=4, metastability_obs_rate=obs_rate),
            device="cpu",
        )
        replay = UnifiedReplayMemory(
            substrate=substrate, memory=memory, consolidation=cons,
            config=ReplayConfig(
                metastability_gain=gain,
                metastability_replay_decay=0.5,
            ),
        )
        # Store 4 nearly-tied patterns so retrieval produces a diffuse softmax.
        gen = torch.Generator(device="cpu").manual_seed(101)
        base = torch.polar(
            torch.ones((substrate.dim,)),
            torch.rand((substrate.dim,), generator=gen) * (2.0 * math.pi),
        )
        for k in range(4):
            phase_jitter = torch.rand((substrate.dim,), generator=gen) * 0.02
            patt = substrate.normalize(base * torch.polar(torch.ones((substrate.dim,)), phase_jitter))
            memory.store(patt)
        replay.attach_initial_patterns()
        return cons, memory, replay, base

    def test_diffuse_retrieval_accumulates_metastability(self):
        cons, memory, replay, base = self._build(obs_rate=0.5, gain=0.0)
        # Query with the base pattern — all 4 stored patterns are near-duplicates,
        # so the softmax is diffuse and c_i is non-trivial.
        replay.retrieve_and_observe(query=base, beta=4.0, max_iter=6)
        # At least one atom should now have m_i > 0.
        self.assertGreater(float(cons.metastability_ema.max()), 0.01)

    def test_obs_rate_zero_keeps_metastability_at_zero(self):
        """The κ=0 control: with obs_rate=0, m_i never moves regardless of weights."""
        cons, memory, replay, base = self._build(obs_rate=0.0, gain=0.0)
        for _ in range(3):
            replay.retrieve_and_observe(query=base, beta=4.0, max_iter=6)
        self.assertTrue(torch.all(cons.metastability_ema == 0.0))


@unittest.skipIf(torch is None, "torch required")
class TestRunReplayCycleFiresMetastability(unittest.TestCase):
    """Regression: run_replay_cycle's replay retrievals must update m_i.

    The n=10 graduation retrain (2026-05-20) initially produced
    bit-identical κ=0 vs κ=2.0 results because the replay-cycle
    retrievals were not calling update_metastability. This test guards
    against re-introducing that bypass.
    """

    def _build(self, obs_rate):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        from energy_memory.phase4.replay_loop import (
            ReplayConfig, UnifiedReplayMemory,
        )
        from energy_memory.phase4.trajectory import TracedHopfieldMemory
        from energy_memory.substrate.torch_fhrr import TorchFHRR

        substrate = TorchFHRR(dim=256, seed=99, device="cpu")
        memory = TracedHopfieldMemory(substrate)
        cons = ConsolidationState(
            ConsolidationConfig(m=4, metastability_obs_rate=obs_rate),
            device="cpu",
        )
        replay = UnifiedReplayMemory(
            substrate=substrate, memory=memory, consolidation=cons,
            config=ReplayConfig(
                store_threshold=0.0,
                replay_every=1, replay_batch_size=2,
                metastability_gain=2.0,
                metastability_replay_decay=0.5,
            ),
        )
        gen = torch.Generator(device="cpu").manual_seed(7)
        base = torch.polar(
            torch.ones((substrate.dim,)),
            torch.rand((substrate.dim,), generator=gen) * (2.0 * math.pi),
        )
        for k in range(4):
            phase_jitter = torch.rand((substrate.dim,), generator=gen) * 0.03
            patt = substrate.normalize(base * torch.polar(torch.ones((substrate.dim,)), phase_jitter))
            memory.store(patt)
        replay.attach_initial_patterns()
        # Seed the replay store with one trace so run_replay_cycle has
        # something to retrieve.
        replay.retrieve_and_observe(query=base, beta=4.0, max_iter=6)
        return cons, replay

    def test_run_replay_cycle_updates_m_i(self):
        cons, replay = self._build(obs_rate=0.5)
        # Reset m_i to zero so we can verify run_replay_cycle moves it.
        cons.metastability_ema.zero_()
        replay.run_replay_cycle(beta=4.0, max_iter=6)
        # The replay retrieval should have pushed m_i above zero somewhere.
        self.assertGreater(float(cons.metastability_ema.max()), 0.0)

    def test_run_replay_cycle_noop_at_obs_rate_zero(self):
        cons, replay = self._build(obs_rate=0.0)
        cons.metastability_ema.zero_()
        replay.run_replay_cycle(beta=4.0, max_iter=6)
        self.assertTrue(torch.all(cons.metastability_ema == 0.0))


if __name__ == "__main__":
    unittest.main()
