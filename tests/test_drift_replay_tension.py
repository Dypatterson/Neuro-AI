"""C.2.5 — Drift -> Replay-Tension Energy.

Verifies the wiring across:

  - ConsolidationConfig.drift_ema_rate (mu_drift)
  - ConsolidationConfig.drift_replay_gain (kappa_drift)
  - ConsolidationState.drift_tension + _previous_codebook
  - ConsolidationState.snapshot_previous_codebook / update_drift_tension
  - ReplayConfig.drift_replay_gain
  - ReplayStore._priorities composes (1 + kappa_drift * Psi[primary])

Per the precommit at
notes/notes/2026-05-26-c25-drift-replay-tension-precommit.md the binding
watch-edges from the anti-homunculus reviewer require:

  - WE1 (A10): joint-factor entropy floor (kappa_meta * kappa_drift).
  - A8 (binding): feedback-loop stability across composed C.2.x mechanisms.
  - A9 (binding): composed-system smoothness across five statistics.
  - A7 (binding): single-factor kappa->inf entropy floor.

If any binding assertion fails the test FAILS — do NOT relax thresholds.
There is NO clamp on Psi_k (H23 binding); revise defaults, never clamp.
"""

from __future__ import annotations

import math
import statistics
import unittest
from typing import List, Tuple

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def _make_trace(dim: int = 8):
    # Minimal frozen TrajectoryTrace for ReplayStore tests.
    from energy_memory.phase4.trajectory import TrajectoryTrace
    return TrajectoryTrace(query=torch.zeros(dim, dtype=torch.complex64))


def _build_store_with_drift(
    n_traces: int,
    psi_values: List[float],
    kappa_drift: float,
    *,
    m_values: List[float] = None,
    kappa_meta: float = 0.0,
    mu_obs: float = 0.0,
    mu_drift: float = 0.0,
):
    """Wire ReplayStore + ConsolidationState so that:

      primary_atom[i] == i
      state.drift_tension[i] == psi_values[i]
      state.metastability_ema[i] == m_values[i] (if supplied)
    """
    from energy_memory.phase4.consolidation import (
        ConsolidationConfig, ConsolidationState,
    )
    from energy_memory.phase4.replay_loop import ReplayStore

    cfg = ConsolidationConfig(
        m=3,
        metastability_obs_rate=mu_obs,
        drift_ema_rate=mu_drift,
        drift_replay_gain=kappa_drift,
    )
    state = ConsolidationState(cfg, device="cpu")
    for _ in range(n_traces):
        state.add_pattern(novelty_strength=1.0)
    # Overwrite EMAs directly — we are testing the priority math, not the
    # update rule.
    state.drift_tension = torch.tensor(psi_values, dtype=torch.float32)
    if m_values is not None:
        state.metastability_ema = torch.tensor(m_values, dtype=torch.float32)

    store = ReplayStore(
        capacity=n_traces,
        consolidation=state,
        metastability_gain=kappa_meta,
        metastability_replay_decay=0.0,
        drift_replay_gain=kappa_drift,
    )
    for i in range(n_traces):
        store.add(_make_trace(), gate_signal=1.0, primary_atom_idx=i)
    return store, state


def _sample_counts(store, n_samples: int, seed: int = 0) -> List[int]:
    # Sample one trace at a time so any payback / suppression applies in order.
    gen = torch.Generator().manual_seed(seed)
    counts = [0] * len(store)
    for _ in range(n_samples):
        idx_list = store.sample(1, generator=gen)
        if not idx_list:
            continue
        counts[idx_list[0]] += 1
    return counts


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch required")
class UnitTests(unittest.TestCase):

    def test_mu_drift_zero_early_exits(self):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(m=3, drift_ema_rate=0.0)
        state = ConsolidationState(cfg, device="cpu")
        for _ in range(4):
            state.add_pattern()
        cb = torch.randn(4, 8, dtype=torch.complex64)
        state.snapshot_previous_codebook(cb)
        # mu_drift == 0 -> snapshot is a no-op (previous_codebook stays None).
        self.assertIsNone(state._previous_codebook)
        # Drift signal should NOT accumulate.
        cb_next = cb + 0.1
        state.update_drift_tension(cb_next)
        self.assertTrue(torch.all(state.drift_tension == 0.0))

    def test_kappa_drift_zero_early_exits_d_factor_build(self):
        # At kappa_drift = 0, _priorities() must take the else-branch and
        # never read drift_tension. All entries equal -> identical priorities.
        store, _ = _build_store_with_drift(
            n_traces=4,
            psi_values=[0.5, 0.1, 0.2, 0.3],
            kappa_drift=0.0,
        )
        priorities = store._priorities()
        self.assertTrue(all(p == priorities[0] for p in priorities))

    def test_drift_tension_is_nonnegative(self):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(m=3, drift_ema_rate=0.3)
        state = ConsolidationState(cfg, device="cpu")
        for _ in range(4):
            state.add_pattern()
        cb = torch.randn(4, 8, dtype=torch.complex64)
        state.snapshot_previous_codebook(cb)
        # Arbitrary perturbation including negative directions: drift_signal
        # is a NORM so it must remain >= 0.
        cb_next = cb + torch.randn_like(cb) * 0.5
        state.update_drift_tension(cb_next)
        self.assertTrue(torch.all(state.drift_tension >= 0.0))

    def test_snapshot_copies_current(self):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(m=3, drift_ema_rate=0.1)
        state = ConsolidationState(cfg, device="cpu")
        for _ in range(3):
            state.add_pattern()
        cb = torch.randn(3, 8, dtype=torch.complex64)
        state.snapshot_previous_codebook(cb)
        self.assertIsNotNone(state._previous_codebook)
        # It is a clone, not the same object.
        self.assertIsNot(state._previous_codebook, cb)
        # Values match.
        self.assertTrue(torch.equal(state._previous_codebook, cb))
        # Mutating the source must NOT change the snapshot.
        cb[0] = cb[0] + 1.0
        self.assertFalse(torch.equal(state._previous_codebook, cb))

    def test_update_drift_tension_known_delta(self):
        # Direct algebraic check: per-atom delta of 0.1 across all D coords.
        # ||delta_k|| = sqrt(D * 0.1^2) = 0.1 * sqrt(D) on FHRR-style tensors
        # where the per-coord magnitude is the |z| value. We use a real-valued
        # tensor here for arithmetic transparency; the magnitude formula is
        # identical (||x||_2). With mu_drift=0.2 the first EMA step gives
        # Psi = 0.2 * ||delta||.
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        d = 16
        cfg = ConsolidationConfig(m=3, drift_ema_rate=0.2)
        state = ConsolidationState(cfg, device="cpu")
        for _ in range(2):
            state.add_pattern()
        cb_prev = torch.zeros(2, d, dtype=torch.complex64)
        state.snapshot_previous_codebook(cb_prev)
        # Add a uniform 0.1 (real) per coordinate to atom 0; leave atom 1 alone.
        cb_curr = cb_prev.clone()
        cb_curr[0] = cb_curr[0] + 0.1
        state.update_drift_tension(cb_curr)
        expected_norm = 0.1 * math.sqrt(d)
        expected_psi0 = 0.2 * expected_norm
        self.assertAlmostEqual(
            float(state.drift_tension[0]), expected_psi0, places=5,
        )
        self.assertAlmostEqual(
            float(state.drift_tension[1]), 0.0, places=6,
        )

    def test_complex_fhrr_tensors_handled(self):
        # Complex tensor with a known per-coord magnitude: e^{i*pi/2} - 1 = i - 1
        # has |.| = sqrt(2). ||delta||_2 over D coords = sqrt(D) * sqrt(2).
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        d = 8
        cfg = ConsolidationConfig(m=3, drift_ema_rate=1.0)  # full replacement
        state = ConsolidationState(cfg, device="cpu")
        for _ in range(1):
            state.add_pattern()
        cb_prev = torch.ones(1, d, dtype=torch.complex64)  # value 1+0j
        state.snapshot_previous_codebook(cb_prev)
        cb_curr = torch.full((1, d), 1j, dtype=torch.complex64)  # 0+1j
        # delta per coord = (0+1j) - (1+0j) = -1+1j -> |.| = sqrt(2)
        state.update_drift_tension(cb_curr)
        expected = math.sqrt(d) * math.sqrt(2.0)
        self.assertAlmostEqual(
            float(state.drift_tension[0]), expected, places=4,
        )

    def test_shape_mismatch_skips_update(self):
        # After add_pattern (or prune) the previous-codebook snapshot may
        # have a stale shape. The update must skip gracefully — not raise.
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(m=3, drift_ema_rate=0.2)
        state = ConsolidationState(cfg, device="cpu")
        for _ in range(3):
            state.add_pattern()
        cb_prev = torch.zeros(3, 8, dtype=torch.complex64)
        state.snapshot_previous_codebook(cb_prev)
        # Simulate add_pattern between snapshot and update -> previous_codebook
        # is invalidated (set to None per the implementation).
        state.add_pattern()
        cb_now = torch.zeros(4, 8, dtype=torch.complex64)
        # Either previous is None or shape differs — either way, no raise.
        # And drift_tension stays at zero.
        try:
            state.update_drift_tension(cb_now)
        except Exception as e:  # pragma: no cover
            self.fail(f"shape mismatch raised: {e}")
        self.assertTrue(torch.all(state.drift_tension == 0.0))

    def test_priority_value_kappa_drift_2_psi_05(self):
        # Direct algebraic check: (1 + 2.0 * 0.5) == 2.0.
        store, _ = _build_store_with_drift(
            n_traces=1, psi_values=[0.5], kappa_drift=2.0,
        )
        self.assertAlmostEqual(store._priorities()[0], 2.0, places=6)


# ---------------------------------------------------------------------------
# Convergence-equivalence binding assertions
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch required")
class ConvergenceEquivalenceTests(unittest.TestCase):

    def test_A1_psi_accumulates_with_sustained_drift(self):
        # A1: perturb atom 0 by 0.1 per event for 20 events with mu_drift=0.1;
        # atom 1 stays stable. Assert Psi_0 > 0 and Psi_0 > Psi_1.
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        d = 16
        cfg = ConsolidationConfig(m=3, drift_ema_rate=0.1)
        state = ConsolidationState(cfg, device="cpu")
        for _ in range(2):
            state.add_pattern()
        cb = torch.zeros(2, d, dtype=torch.complex64)
        delta = 0.1
        for _ in range(20):
            state.snapshot_previous_codebook(cb)
            cb = cb.clone()
            cb[0] = cb[0] + delta
            state.update_drift_tension(cb)
        psi_0 = float(state.drift_tension[0])
        psi_1 = float(state.drift_tension[1])
        print(f"\n[A1] Psi_0={psi_0:.6f} Psi_1={psi_1:.6f}")
        self.assertGreater(psi_0, 0.0)
        self.assertGreater(psi_0, psi_1)
        self.assertAlmostEqual(psi_1, 0.0, places=6)

    def test_A2_high_psi_traces_sampled_more_often(self):
        # A2: psi_values[0]=high, others=low. With kappa_drift=2.0, trace 0
        # count > 2 * mean of other counts.
        # Analytic ratio with psi=[1.0, 0.05, 0.05, ...]: (1 + 2.0 * 1.0) /
        # (1 + 2.0 * 0.05) = 3.0 / 1.1 = 2.73 — gives headroom over the 2x
        # assertion (matches the methodology fix in C.2.4 A1).
        n = 8
        psi = [1.0] + [0.05] * (n - 1)
        store, _ = _build_store_with_drift(
            n_traces=n, psi_values=psi, kappa_drift=2.0,
        )
        counts = _sample_counts(store, n_samples=1000, seed=11)
        others_mean = sum(counts[1:]) / (n - 1)
        print(
            f"\n[A2] trace0={counts[0]} mean_others={others_mean:.2f} "
            f"ratio={counts[0] / max(others_mean, 1e-9):.2f} "
            f"(predicted ~2.73x at kappa_drift=2.0)"
        )
        self.assertGreater(
            counts[0], 2.0 * others_mean,
            f"trace0={counts[0]} not > 2x mean_others={others_mean:.2f}"
        )

    def test_A3_kappa_drift_zero_byte_identical(self):
        # A3: kappa_drift=0 must produce IDENTICAL counts whether
        # drift_tension is zero or pre-populated. This proves the
        # `if kappa_drift > 0.0` branch in _priorities() is the only place
        # drift touches priority.
        n = 8
        seed = 23

        store_zero, _ = _build_store_with_drift(
            n_traces=n, psi_values=[0.0] * n, kappa_drift=0.0,
        )
        counts_zero = _sample_counts(store_zero, n_samples=1000, seed=seed)

        store_full, _ = _build_store_with_drift(
            n_traces=n,
            psi_values=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01],
            kappa_drift=0.0,
        )
        counts_full = _sample_counts(store_full, n_samples=1000, seed=seed)

        self.assertEqual(
            counts_zero, counts_full,
            "kappa_drift=0 leaks drift_tension into priority",
        )
        print("\n[A3] kappa_drift=0 baseline counts identical")

    def test_A4_mu_drift_zero_byte_identical(self):
        # A4: mu_drift=0 -> drift_tension stays at 0 even after many events;
        # multiplier exactly 1.0 regardless of kappa_drift.
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        d = 16
        cfg = ConsolidationConfig(m=3, drift_ema_rate=0.0)  # off
        state = ConsolidationState(cfg, device="cpu")
        for _ in range(8):
            state.add_pattern()
        cb = torch.zeros(8, d, dtype=torch.complex64)
        for _ in range(20):
            state.snapshot_previous_codebook(cb)
            cb = cb + 0.1
            state.update_drift_tension(cb)
        self.assertTrue(torch.all(state.drift_tension == 0.0))

        # Sample counts with kappa_drift=2.0 must equal the kappa_drift=0
        # baseline because Psi is identically zero.
        from energy_memory.phase4.replay_loop import ReplayStore

        def _build_with_psi_zero(kappa):
            cfg = ConsolidationConfig(m=3, drift_ema_rate=0.0)
            st = ConsolidationState(cfg, device="cpu")
            for _ in range(8):
                st.add_pattern(novelty_strength=1.0)
            st.drift_tension = torch.zeros(8, dtype=torch.float32)
            store = ReplayStore(
                capacity=8, consolidation=st,
                metastability_gain=0.0, metastability_replay_decay=0.0,
                drift_replay_gain=kappa,
            )
            for i in range(8):
                store.add(_make_trace(), gate_signal=1.0, primary_atom_idx=i)
            return store

        counts_zero = _sample_counts(_build_with_psi_zero(0.0), 1000, seed=31)
        counts_k2 = _sample_counts(_build_with_psi_zero(2.0), 1000, seed=31)
        self.assertEqual(counts_zero, counts_k2,
                         "mu_drift=0 baseline broken: Psi=0 must give 1.0 multiplier")
        print("\n[A4] mu_drift=0 baseline identical across kappa_drift")

    def test_A5_trajectory_smoothness(self):
        # A5: run the full replay loop for 500 events with kappa_drift=1.0
        # and mu_drift=0.1. Sliding-window (w=20) on mean(drift_tension) over
        # events [200, 500] -> max/median ratio < 5.
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.phase2.encoding import (
            build_position_vectors, encode_window,
        )
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        from energy_memory.phase4.trajectory import TracedHopfieldMemory
        from energy_memory.phase4.replay_loop import (
            ReplayConfig, UnifiedReplayMemory,
        )

        substrate = TorchFHRR(dim=128, seed=5, device="cpu")
        positions = build_position_vectors(substrate, 3)
        codebook = substrate.random_vectors(8)
        memory = TracedHopfieldMemory(substrate, snapshot_k=4)
        windows = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)]
        for w in windows:
            memory.store(
                encode_window(substrate, positions, codebook, w), label=str(w),
            )

        cons_cfg = ConsolidationConfig(
            m=3, drift_ema_rate=0.1,
        )
        cons = ConsolidationState(cons_cfg, device="cpu")
        replay_cfg = ReplayConfig(
            store_threshold=0.0,
            store_capacity=64,
            resolve_threshold=0.999,  # disable candidate resolution
            replay_every=5,
            replay_batch_size=2,
            max_age=100,
            drift_replay_gain=1.0,
        )
        unified = UnifiedReplayMemory(
            substrate=substrate, memory=memory,
            consolidation=cons, config=replay_cfg,
        )
        unified.attach_initial_patterns()

        gen = torch.Generator().manual_seed(13)
        psi_means: List[float] = []
        # UnifiedReplayMemory uses a static memory (no consolidation
        # orchestrator), so we drive drift_tension by capturing and then
        # perturbing the consolidation codebook directly each event. This
        # exercises the smoothness of the Psi EMA under realistic per-event
        # drift magnitudes (Gaussian noise ~0.05 on each atom per event).
        perturb_codebook = memory._pattern_matrix().detach().clone()
        for _ in range(500):
            base = encode_window(
                substrate, positions, codebook,
                windows[int(torch.randint(0, len(windows), (1,), generator=gen))],
            )
            noisy = substrate.normalize(base + 0.2 * substrate.random_vector())
            cons.snapshot_previous_codebook(perturb_codebook)
            unified.retrieve_and_observe(noisy, beta=5.0)
            if unified.should_replay():
                unified.run_replay_cycle(beta=5.0)
            # Per-event Gaussian perturbation of the consolidation-side
            # codebook: ~0.05 std on real and imag parts. Drift signal per
            # atom is ||delta|| with delta drawn iid each event.
            noise = (
                torch.randn(perturb_codebook.shape, generator=gen) * 0.05
                + 1j * torch.randn(perturb_codebook.shape, generator=gen) * 0.05
            ).to(perturb_codebook.dtype)
            perturb_codebook = perturb_codebook + noise
            cons.update_drift_tension(perturb_codebook)
            psi_means.append(float(cons.drift_tension.mean()))

        warmup = 200
        smooth_window_size = 20
        window = psi_means[warmup:]
        smoothed: List[float] = []
        for i in range(len(window) - smooth_window_size + 1):
            chunk = window[i:i + smooth_window_size]
            smoothed.append(sum(chunk) / smooth_window_size)
        median_s = statistics.median(smoothed)
        max_s = max(smoothed)
        if median_s < 1e-12:
            ratio = 0.0
            self.assertLess(
                max_s, 1e-6,
                f"[A5] smoothed flat but max={max_s:.4e}"
            )
        else:
            ratio = max_s / median_s
            self.assertLess(
                ratio, 5.0,
                f"[A5] smoothed max/median={ratio:.2f} >= 5 — real finding."
            )
        print(
            f"\n[A5] sliding-window (w={smooth_window_size}) smoothed "
            f"mean(Psi): n={len(smoothed)} median={median_s:.4e} "
            f"max={max_s:.4e} ratio={ratio:.2f}"
        )

    def test_A6_actuator_does_not_consume_codebook_drift(self):
        # A6 (H6 binding): monkey-patch
        # energy_memory.phase34.reencoding.codebook_drift to raise. C.2.5
        # must compute drift directly from substrate state and run unchanged.
        import energy_memory.phase34.reencoding as reenc

        n = 8
        psi = [0.5] + [0.05] * (n - 1)
        store_normal, _ = _build_store_with_drift(
            n_traces=n, psi_values=list(psi), kappa_drift=2.0,
        )
        counts_normal = _sample_counts(store_normal, n_samples=500, seed=44)

        original = reenc.codebook_drift

        def boom(*args, **kwargs):
            raise RuntimeError(
                "C.2.5 actuator must not call reencoding.codebook_drift."
            )

        reenc.codebook_drift = boom
        try:
            # Both priority sampling AND state.update_drift_tension must run.
            store_patched, state_patched = _build_store_with_drift(
                n_traces=n, psi_values=list(psi), kappa_drift=2.0,
            )
            counts_patched = _sample_counts(
                store_patched, n_samples=500, seed=44,
            )
            # Also exercise update_drift_tension under the monkey-patch.
            from energy_memory.phase4.consolidation import (
                ConsolidationConfig, ConsolidationState,
            )
            cfg2 = ConsolidationConfig(m=3, drift_ema_rate=0.1)
            st2 = ConsolidationState(cfg2, device="cpu")
            for _ in range(3):
                st2.add_pattern()
            cb = torch.zeros(3, 8, dtype=torch.complex64)
            st2.snapshot_previous_codebook(cb)
            cb_new = cb + 0.1
            st2.update_drift_tension(cb_new)  # must not raise
        finally:
            reenc.codebook_drift = original

        self.assertEqual(
            counts_normal, counts_patched,
            "actuator priority depends on the codebook_drift primitive",
        )
        print("\n[A6] H6 verified — actuator does NOT consume codebook_drift")

    def test_A7_kappa_drift_infinity_entropy_floor(self):
        # A7 (binding): with kappa_drift=100 (extreme) and varied Psi values
        # in [0, 0.5], the sample-distribution entropy must remain
        # > 0.5 * log(N).
        n = 8
        gen = torch.Generator().manual_seed(101)
        psi_values = torch.rand(n, generator=gen).mul(0.5).tolist()
        store, _ = _build_store_with_drift(
            n_traces=n, psi_values=psi_values, kappa_drift=100.0,
        )
        counts = _sample_counts(store, n_samples=1000, seed=51)
        total = sum(counts)
        probs = [c / total for c in counts if c > 0]
        entropy = -sum(p * math.log(p) for p in probs)
        threshold = 0.5 * math.log(n)
        print(
            f"\n[A7] kappa_drift=100 entropy={entropy:.4f} "
            f"threshold=0.5*log({n})={threshold:.4f}"
        )
        self.assertGreater(
            entropy, threshold,
            f"sample entropy {entropy:.4f} <= {threshold:.4f} — "
            "the drift multiplier collapsed the distribution; real "
            "finding, do NOT relax."
        )

    def _build_full_c2x_loop(self, kappa_drift=1.0, mu_drift=0.1):
        # Helper: wire UnifiedReplayMemory + all five C.2.x mechanisms.
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.phase2.encoding import (
            build_position_vectors, encode_window,
        )
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        from energy_memory.phase4.trajectory import TracedHopfieldMemory
        from energy_memory.phase4.replay_loop import (
            ReplayConfig, UnifiedReplayMemory,
        )

        substrate = TorchFHRR(dim=128, seed=29, device="cpu")
        positions = build_position_vectors(substrate, 3)
        codebook = substrate.random_vectors(4)
        memory = TracedHopfieldMemory(substrate, snapshot_k=4)
        windows = [(0, 1, 2), (1, 2, 3), (2, 0, 1), (3, 1, 2)]
        for w in windows:
            memory.store(
                encode_window(substrate, positions, codebook, w), label=str(w),
            )

        cons_cfg = ConsolidationConfig(
            m=3,
            # C.2.1
            lambda_ac=0.5, epsilon_ac=1e-4,
            # C.2.2
            mu_T=0.1, tau_T=0.5, epsilon_T=1e-6,
            # C.2.3
            lambda_cc=0.5, theta_cc=0.5, tau_cc=0.1,
            # C.2.4
            metastability_obs_rate=0.1,
            # C.2.5
            drift_ema_rate=mu_drift,
            drift_replay_gain=kappa_drift,
            basin_trace_buffer_size=128,
            min_basin_for_signal=4,
        )
        cons = ConsolidationState(cons_cfg, device="cpu")
        replay_cfg = ReplayConfig(
            store_threshold=0.0,
            store_capacity=128,
            resolve_threshold=0.999,
            replay_every=5,
            replay_batch_size=2,
            max_age=100,
            metastability_gain=2.0,
            metastability_replay_decay=0.5,
            drift_replay_gain=kappa_drift,
        )
        unified = UnifiedReplayMemory(
            substrate=substrate, memory=memory,
            consolidation=cons, config=replay_cfg,
        )
        unified.attach_initial_patterns()
        return substrate, positions, codebook, memory, cons, unified, windows

    def test_A8_feedback_loop_stability(self):
        # A8 (binding): all five C.2.x mechanisms on. Run 300 events.
        # Assert post-warmup mean(Psi) trajectory is bounded:
        #   max(Psi_mean[100:300]) - min(Psi_mean[100:300]) < 2 * Psi_mean[100]
        (
            substrate, positions, codebook,
            memory, cons, unified, windows,
        ) = self._build_full_c2x_loop(kappa_drift=1.0, mu_drift=0.1)

        gen = torch.Generator().manual_seed(37)
        psi_means: List[float] = []
        # Drive controlled codebook drift each event so C.2.5's EMA is
        # actually exercised — without an OnlineCodebookUpdater orchestrator
        # the UnifiedReplayMemory keeps patterns static, so the drift
        # feedback loop has no source. With iid Gaussian per-event drift,
        # A8's binding is the EMA's bounded behavior under continuous
        # perturbation.
        perturb_codebook = memory._pattern_matrix().detach().clone()
        for step in range(300):
            wi = int(torch.randint(0, len(windows), (1,), generator=gen))
            base = encode_window_local(
                substrate, positions, codebook, windows[wi],
            )
            noise_mag = 0.15 + 0.25 * (wi == 0)
            noisy = substrate.normalize(base + noise_mag * substrate.random_vector())
            cons.snapshot_previous_codebook(perturb_codebook)
            unified.retrieve_and_observe(noisy, beta=5.0)
            # Drive splitting tension EMA (UnifiedReplayMemory does not).
            cons.update_splitting_tension()
            if unified.should_replay():
                unified.run_replay_cycle(beta=5.0)
            drift_noise = (
                torch.randn(perturb_codebook.shape, generator=gen) * 0.05
                + 1j * torch.randn(perturb_codebook.shape, generator=gen) * 0.05
            ).to(perturb_codebook.dtype)
            perturb_codebook = perturb_codebook + drift_noise
            cons.update_drift_tension(perturb_codebook)
            psi_means.append(float(cons.drift_tension.mean()))

        warmup_end = psi_means[100]
        late = psi_means[100:300]
        spread = max(late) - min(late)
        # Bound: spread < 2 * warmup_end. Handle warmup_end ~= 0 by adding a
        # small absolute floor — if Psi stays near zero throughout, spread
        # should also stay near zero.
        bound = max(2.0 * warmup_end, 2e-3)
        print(
            f"\n[A8] mean(Psi) @ event100={warmup_end:.6f} "
            f"spread[100:300]={spread:.6f} "
            f"ratio={spread / max(warmup_end, 1e-9):.2f} "
            f"bound={bound:.6f}"
        )
        self.assertLess(
            spread, bound,
            f"[A8] feedback runaway: spread={spread:.6f} >= {bound:.6f} — "
            "real finding, revise defaults (NOT clamp; H23 binding)."
        )

    def test_A9_composed_system_smoothness(self):
        # A9 (binding): all five C.2.x mechanisms. Run 300 events. Apply
        # sliding-window + floor gating to mean(m), mean(T), Σ tr(Σ),
        # mean(Psi), plus replay sample-distribution entropy. Each smooth.
        torch.manual_seed(43)
        (
            substrate, positions, codebook,
            memory, cons, unified, windows,
        ) = self._build_full_c2x_loop(kappa_drift=1.0, mu_drift=0.1)

        gen = torch.Generator().manual_seed(43)
        m_means: List[float] = []
        T_means: List[float] = []
        tr_sums: List[float] = []
        psi_means: List[float] = []
        sample_entropy: List[float] = []

        n_events = 300
        warmup = 100

        # Same controlled per-event codebook drift as A8 — exercises C.2.5
        # EMA under the composed dynamics.
        perturb_codebook = memory._pattern_matrix().detach().clone()
        for step in range(n_events):
            wi = int(torch.randint(0, len(windows), (1,), generator=gen))
            base = encode_window_local(
                substrate, positions, codebook, windows[wi],
            )
            noise_mag = 0.15 + 0.25 * (wi == 0)
            noisy = substrate.normalize(base + noise_mag * substrate.random_vector())
            cons.snapshot_previous_codebook(perturb_codebook)
            unified.retrieve_and_observe(noisy, beta=5.0)
            cons.update_splitting_tension()
            if unified.should_replay():
                unified.run_replay_cycle(beta=5.0)
            drift_noise = (
                torch.randn(perturb_codebook.shape, generator=gen) * 0.05
                + 1j * torch.randn(perturb_codebook.shape, generator=gen) * 0.05
            ).to(perturb_codebook.dtype)
            perturb_codebook = perturb_codebook + drift_noise
            cons.update_drift_tension(perturb_codebook)

            m_means.append(float(cons.metastability_ema.mean()))
            T_means.append(float(cons.splitting_tension.mean()))
            tr_sum = 0.0
            for k in range(cons.n_patterns):
                _, _, tr = cons._basin_covariance(k)
                tr_sum += tr
            tr_sums.append(tr_sum)
            psi_means.append(float(cons.drift_tension.mean()))
            # Replay sample-distribution entropy — derived from current
            # store priorities. Use _priorities() then normalize.
            priorities = unified.store._priorities()
            if priorities:
                tot = sum(p for p in priorities if p > 0)
                if tot > 0:
                    probs = [p / tot for p in priorities if p > 0]
                    h = -sum(p * math.log(p) for p in probs if p > 0)
                else:
                    h = 0.0
            else:
                h = 0.0
            sample_entropy.append(h)

        # Methodology per the C.2.5 precommit (notes/notes/2026-05-26-
        # c25-drift-replay-tension-precommit.md A9): sliding-window mean
        # (w=20) on the post-warmup trajectory, then per-event delta of the
        # smoothed series, with floor gating that switches to an absolute-
        # max check when the median delta is below the float32 noise floor.
        # Sliding-window averages out cue-noise from the store's stochastic
        # membership churn (especially visible in H_sample) and isolates
        # the underlying smooth dynamic — same rationale as
        # tests/test_metastability_replay_priority.py A4.
        FLOOR_MEDIAN = 1e-7
        FLOOR_ABS_MAX = 1e-3
        smooth_window = 20

        def _smoothness(name, traj):
            window = traj[warmup:]
            if len(window) < smooth_window:
                return name, 0.0, 0.0, 0.0
            smoothed = [
                sum(window[i:i + smooth_window]) / smooth_window
                for i in range(len(window) - smooth_window + 1)
            ]
            diffs = [
                abs(smoothed[i] - smoothed[i - 1])
                for i in range(1, len(smoothed))
            ]
            if not diffs:
                return name, 0.0, 0.0, 0.0
            med = statistics.median(diffs)
            mx = max(diffs)
            if med < 1e-12:
                ratio = 0.0 if mx < 1e-6 else float("inf")
            else:
                ratio = mx / med
            return name, med, mx, ratio

        results = []
        for name, traj in [
            ("m", m_means), ("T", T_means), ("trΣ", tr_sums),
            ("Psi", psi_means), ("H_sample", sample_entropy),
        ]:
            results.append(_smoothness(name, traj))

        for name, med, mx, ratio in results:
            print(
                f"\n[A9] {name}: median={med:.4e} max={mx:.4e} "
                f"ratio={ratio:.2f} (post-warmup [{warmup},{n_events}])"
            )

        # Floor gating per the precommit: "max/median ratio of sliding-window
        # Δ < 5 OR absolute-scale check passes." The absolute-scale check
        # catches dynamics that look ratio-noisy but are tiny in absolute
        # terms (the smoothed series can produce ratios > 5 from
        # numerical-noise-dominated fluctuations whose max is sub-1e-3).
        # FLOOR_ABS_MAX is set to the small-absolute-scale threshold — if
        # the smoothed series never moves by more than FLOOR_ABS_MAX in any
        # step, the dynamic is essentially stationary and ratio inflation
        # from a near-zero denominator is not a real finding.
        for name, med, mx, ratio in results:
            if mx < FLOOR_ABS_MAX:
                # Absolute scale negligible — dynamic is stationary at the
                # smoothing-noise floor; ratio inflation is denominator
                # collapse, not discontinuity.
                continue
            if med < FLOOR_MEDIAN:
                self.assertLess(
                    mx, FLOOR_ABS_MAX,
                    f"[A9] '{name}' median={med:.2e} below floor but "
                    f"max={mx:.2e} >= {FLOOR_ABS_MAX:.0e} — discontinuity.",
                )
            else:
                self.assertLess(
                    ratio, 5.0,
                    f"[A9] '{name}' max/median={ratio:.2f} >= 5 (max={mx:.2e}"
                    f", median={med:.2e}) — composition not principled. "
                    "Real finding, do NOT relax."
                )

    def test_a10a_realistic_joint_entropy_floor(self):
        # A10a (binding, realistic operating regime per WE1 precommit split):
        # kappa_meta=4.0 AND kappa_drift=2.0. m AND Psi maximal on the SAME
        # subset of atoms. Sample entropy must remain > 0.5 * log(N).
        #
        # Joint multiplier ratio at these parameters:
        #   (1 + 4 * 0.5)(1 + 2 * 0.5) = 3 * 2 = 6  (correlated-max atoms)
        #   (1 + 4 * 0.05)(1 + 2 * 0.05) = 1.2 * 1.1 = 1.32  (other atoms)
        #   ratio ~= 6 / 1.32 ~= 4.5x — substantial but well below the
        #   degenerate-collapse regime exercised by A10b.
        n = 8
        # Atoms 0-1: m=Psi=0.5 (correlated maxima). Atoms 2-7: m=Psi=0.05.
        m_values = [0.5, 0.5] + [0.05] * (n - 2)
        psi_values = [0.5, 0.5] + [0.05] * (n - 2)
        store, state = _build_store_with_drift(
            n_traces=n,
            psi_values=psi_values,
            kappa_drift=2.0,
            m_values=m_values,
            kappa_meta=4.0,
        )
        counts = _sample_counts(store, n_samples=1000, seed=61)
        total = sum(counts)
        probs = [c / total for c in counts if c > 0]
        entropy = -sum(p * math.log(p) for p in probs)
        threshold = 0.5 * math.log(n)
        print(
            f"\n[A10a] realistic kappa_meta=4.0, kappa_drift=2.0, "
            f"correlated maxima: entropy={entropy:.4f} "
            f"threshold=0.5*log({n})={threshold:.4f}"
        )
        # Binding. If this fails, the realistic operating regime is
        # collapsing the distribution on correlated maxima. REAL finding —
        # revise kappa defaults, do NOT relax. NO clamp on Psi (H23).
        self.assertGreater(
            entropy, threshold,
            f"[A10a] joint entropy={entropy:.4f} <= {threshold:.4f} — the "
            "realistic-regime joint multiplier collapsed the distribution "
            "on correlated maxima. REAL finding; do NOT relax."
        )

    def test_a10b_extreme_joint_regime_is_degenerate_by_design(self):
        # A10b (informational, extreme degenerate regime per WE1 precommit
        # split): kappa_meta=100 AND kappa_drift=100. Same correlated
        # maxima setup. The analytic entropy ceiling is ~0.91 nats (the
        # joint multiplier ratio is (1+100*0.5)^2 / (1+100*0.05)^2 =
        # 51^2 / 6^2 ~= 72x, which concentrates ~99% of probability mass
        # on the two correlated-max atoms). This test documents the
        # substrate's known degenerate regime.
        #
        # Documents the substrate's known degenerate regime per H23. Per
        # the precommit, the response is to operate at moderate gains,
        # not to clamp Psi. Reference: 2026-05-26 reviewer WE1 and the
        # precommit's A10 split.
        n = 8
        m_values = [0.5, 0.5] + [0.05] * (n - 2)
        psi_values = [0.5, 0.5] + [0.05] * (n - 2)
        store, state = _build_store_with_drift(
            n_traces=n,
            psi_values=psi_values,
            kappa_drift=100.0,
            m_values=m_values,
            kappa_meta=100.0,
        )
        counts = _sample_counts(store, n_samples=1000, seed=61)
        total = sum(counts)
        probs = [c / total for c in counts if c > 0]
        entropy = -sum(p * math.log(p) for p in probs)
        print(
            f"\n[A10b] extreme kappa_meta=kappa_drift=100, correlated "
            f"maxima: entropy={entropy:.4f} (analytic ceiling ~0.91 nats)"
        )
        # Upper bound: degenerate regime IS degenerate as expected.
        # Analytic ceiling ~0.91 nats; 1.0 covers float noise.
        self.assertLess(
            entropy, 1.0,
            f"[A10b] entropy={entropy:.4f} >= 1.0 — the extreme regime "
            "is NOT degenerate, which contradicts the H23 analysis. "
            "Investigate before treating as benign."
        )
        # Lower bound: still stochastic, just heavily concentrated.
        self.assertGreater(
            entropy, 0.5,
            f"[A10b] entropy={entropy:.4f} <= 0.5 — the extreme regime "
            "has collapsed to near-deterministic sampling. Sanity floor "
            "violated; investigate."
        )


# Helper: re-export encode_window so the per-event loop in A8/A9 stays one
# import per file (the loop body uses encode_window many times).
def encode_window_local(substrate, positions, codebook, window):
    from energy_memory.phase2.encoding import encode_window as _ew
    return _ew(substrate, positions, codebook, window)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
