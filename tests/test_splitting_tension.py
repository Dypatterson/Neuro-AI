"""C.2.2 — Splitting-tension energy as a substrate dynamic.

Per notes/notes/2026-05-26-c22-splitting-tension-precommit.md. The
anti-homunculus reviewer's 6 binding watch-edges:

  1. Convergence-equivalence includes the 2x2 ablation on a pathological
     basin (A7).
  2. H6 runtime test monkey-patches BOTH compute_bimodality_diagnostics
     AND ContextBagHistory (A5 dual patch).
  3. Identity claim softened in the precommit doc.
  4. Eigendecomposition stays on-device (signal is a tensor).
  5. min_basin_for_signal floor returns signal=0 as a tensor on the same
     device/dtype as the EMA.
  6. mu_T / tau_T smoke is structural (trajectory smoothness), not
     functional.

The actuator's substrate primitive is _basin_covariance — the same one
C.2.1 reads. It NEVER reads BimodalityDiagnostics or ContextBagHistory.
"""

from __future__ import annotations

import math
import statistics
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Synthetic basin builders
# ---------------------------------------------------------------------------


def _unimodal_basin(centroid, n, gen, scale=0.3, noise=0.005):
    """Single-axis spread — members lie along one direction from centroid.

    Sample x = centroid + scale * z * main_dir + noise * isotropic.
    Σ_k has ONE dominant eigenvalue (main_dir variance ≈ scale²) plus
    tiny isotropic noise; so λ_1 ≫ λ_2 → λ_2/λ_1 ≪ 1 ("unimodal" /
    rank-1 spread per the precommit's interpretation).
    """
    D = centroid.shape[0]
    main_dir = torch.randn(D, generator=gen)
    main_dir = main_dir / main_dir.norm().clamp_min(1e-12)
    samples = []
    for _ in range(n):
        z = torch.randn(1, generator=gen)
        eps = noise * torch.randn(D, generator=gen)
        s = centroid + scale * z * main_dir + eps
        s = s / s.norm().clamp_min(1e-12)
        samples.append(s)
    return samples


def _bimodal_basin(centroid, n, gen, scale=0.3, noise=0.005):
    """Two-axis spread — half members along e_a, half along e_b (orthogonal).

    The union spans a 2D subspace with equal variance per axis (≈ scale²/2
    each, halved because only half the mass is in each direction). So
    λ_1 ≈ λ_2 → λ_2/λ_1 ≈ 1 ("bimodal" / rank-2 spread per the precommit's
    interpretation: two clusters of comparable mass along different axes).
    """
    D = centroid.shape[0]
    e_a = torch.randn(D, generator=gen)
    e_a = e_a - (e_a @ centroid) * centroid / centroid.norm().clamp_min(1e-12) ** 2
    e_a = e_a / e_a.norm().clamp_min(1e-12)
    e_b = torch.randn(D, generator=gen)
    e_b = e_b - (e_b @ centroid) * centroid / centroid.norm().clamp_min(1e-12) ** 2
    e_b = e_b - (e_b @ e_a) * e_a
    e_b = e_b / e_b.norm().clamp_min(1e-12)
    samples = []
    half = n // 2
    for _ in range(half):
        z = torch.randn(1, generator=gen)
        eps = noise * torch.randn(D, generator=gen)
        s = centroid + scale * z * e_a + eps
        s = s / s.norm().clamp_min(1e-12)
        samples.append(s)
    for _ in range(n - half):
        z = torch.randn(1, generator=gen)
        eps = noise * torch.randn(D, generator=gen)
        s = centroid + scale * z * e_b + eps
        s = s / s.norm().clamp_min(1e-12)
        samples.append(s)
    return samples


def _pathological_bimodal_basin(centroid, n, gen, sep=0.5, sigma=0.005):
    """A7's pathological geometry: two TIGHT clusters far apart (small
    per-cluster trace + high inter-cluster λ_2/λ_1).

    Differs from `_bimodal_basin` (two orthogonal AXIS spreads through the
    centroid) in that each cluster is a small tight blob at a distinct
    point. Inter-cluster mass split dominates the spectrum but produces
    λ_1 along the (e_a-e_b)/2 direction; combined with the per-cluster
    sigma in two further directions, λ_2 is non-trivial → still high
    ratio but driven by *cluster split* not *axis spread*.
    """
    D = centroid.shape[0]
    e_a = torch.randn(D, generator=gen)
    e_a = e_a / e_a.norm().clamp_min(1e-12)
    e_b = torch.randn(D, generator=gen)
    e_b = e_b - (e_b @ e_a) * e_a
    e_b = e_b / e_b.norm().clamp_min(1e-12)
    cluster_a = centroid + sep * e_a
    cluster_b = centroid - sep * e_a + sep * e_b  # offset along two dirs
    samples = []
    half = n // 2
    for _ in range(half):
        s = cluster_a + sigma * torch.randn(D, generator=gen)
        s = s / s.norm().clamp_min(1e-12)
        samples.append(s)
    for _ in range(n - half):
        s = cluster_b + sigma * torch.randn(D, generator=gen)
        s = s / s.norm().clamp_min(1e-12)
        samples.append(s)
    return samples


def _build_codebook(K=4, D=32, seed=0, bimodal_atom=0,
                    samples_per_basin=16, bimodal_kind="axes"):
    gen = torch.Generator().manual_seed(seed)
    centroids = torch.randn(K, D, generator=gen)
    centroids = centroids / centroids.norm(dim=1, keepdim=True).clamp_min(1e-12)
    codebook = centroids.clone()

    samples = []  # (state, atom_idx)
    for k in range(K):
        if k == bimodal_atom:
            if bimodal_kind == "pathological":
                members = _pathological_bimodal_basin(
                    centroids[k], samples_per_basin, gen,
                )
            else:
                members = _bimodal_basin(centroids[k], samples_per_basin, gen)
        else:
            members = _unimodal_basin(centroids[k], samples_per_basin, gen)
        for s in members:
            samples.append((s, k))
    return codebook, centroids, samples


def _seed_state(state, samples):
    """Force-load the basin buffer (bypasses lambda_ac short-circuit)."""
    for s, k in samples:
        state._basin_buffer.append((s.detach().clone(), int(k)))


def _make_grouped(samples):
    g = {}
    for s, k in samples:
        g.setdefault(k, []).append(s.clone())
    return g


def _run_consolidation_loop(
    state, codebook, grouped, n_steps, lr=0.05,
    record_each_step=True, contract=0.05,
    track_tension_atom=None, track_tr_atom=None,
):
    """Mirror of the C.2.1 loop, but also exercises C.2.2.

    Per the precommit's wiring: update_splitting_tension() runs BEFORE
    the per-atom forces are applied; the modulation attenuates the
    combined (Hebbian + anti-collapse) net update.
    """
    tension_traj = []
    tr_traj = []
    for _ in range(n_steps):
        # Refresh tension once per consolidation event, before any force.
        state.update_splitting_tension()
        for k in sorted(grouped.keys()):
            members = torch.stack(grouped[k], dim=0)
            centroid = members.mean(dim=0)
            pre = codebook[k].detach().clone()
            update = lr * (centroid - codebook[k])
            force = state.anti_collapse_force(k, codebook[k])
            post_raw = codebook[k] + update + force
            modulation = state.splitting_tension_modulation(k)
            blended = pre + modulation * (post_raw - pre)
            codebook[k] = blended / blended.norm().clamp_min(1e-12)
            grouped[k] = [
                (1 - contract) * m + contract * codebook[k] for m in grouped[k]
            ]
        if record_each_step:
            state._basin_buffer.clear()
            for k_, ms in grouped.items():
                for m in ms:
                    state._basin_buffer.append((m.detach().clone(), int(k_)))
        if track_tension_atom is not None:
            tension_traj.append(float(state.splitting_tension[track_tension_atom]))
        if track_tr_atom is not None:
            _, _, tr = state._basin_covariance(track_tr_atom)
            tr_traj.append(tr)
    return tension_traj, tr_traj


def _make_state(mu_T=0.0, tau_T=0.5, lambda_ac=0.0, min_basin=4, device="cpu"):
    from energy_memory.phase4.consolidation import (
        ConsolidationConfig, ConsolidationState,
    )
    cfg = ConsolidationConfig(
        m=3,
        mu_T=mu_T, tau_T=tau_T, epsilon_T=1e-6,
        min_basin_for_signal=min_basin,
        lambda_ac=lambda_ac, epsilon_ac=1e-6,
        basin_trace_buffer_size=256,
    )
    return ConsolidationState(cfg, device=device)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch required")
class UnitTests(unittest.TestCase):

    def test_mu_T_zero_modulation_is_one(self):
        state = _make_state(mu_T=0.0)
        for _ in range(5):
            state.add_pattern()
        self.assertEqual(state.splitting_tension_modulation(0), 1.0)

    def test_mu_T_zero_early_exits_update(self):
        # No eigh, no signal computation. Even with a seeded buffer.
        state = _make_state(mu_T=0.0)
        for _ in range(4):
            state.add_pattern()
        # Seed a bimodal-looking buffer.
        D = 32
        for i in range(8):
            s = torch.randn(D, dtype=torch.float32)
            s = s / s.norm().clamp_min(1e-12)
            state._basin_buffer.append((s, 0))
        state.update_splitting_tension()
        self.assertTrue(torch.all(state.splitting_tension == 0))

    def test_modulation_exact_one_when_T_zero(self):
        state = _make_state(mu_T=0.1)
        state.add_pattern()
        # T_k starts at zero on add_pattern.
        self.assertEqual(float(state.splitting_tension[0]), 0.0)
        self.assertEqual(state.splitting_tension_modulation(0), 1.0)

    def test_modulation_in_open_unit_interval_when_T_positive(self):
        state = _make_state(mu_T=0.1, tau_T=0.5)
        state.add_pattern()
        state.splitting_tension[0] = 0.3
        m = state.splitting_tension_modulation(0)
        self.assertGreater(m, 0.0)
        self.assertLess(m, 1.0)

    def test_modulation_monotone_and_bounded_ratio(self):
        # Assertion 6: monotone decreasing; consecutive factor ratio ≤ 2.
        state = _make_state(mu_T=0.1, tau_T=0.5)
        state.add_pattern()
        T_values = [0.0, 0.1, 0.2, 0.5, 0.9]
        factors = []
        for T in T_values:
            state.splitting_tension[0] = T
            factors.append(state.splitting_tension_modulation(0))
        for a, b in zip(factors[:-1], factors[1:]):
            self.assertGreater(a, b, "modulation not monotone decreasing")
            self.assertLessEqual(a / b, 2.0,
                                 f"consecutive modulation ratio {a/b} > 2x")
        print(f"\n[A6] factors at T={T_values}: {factors}")

    def test_eigvalsh_on_complex_fhrr_tensor(self):
        # FHRR-shaped (complex) basin: eigh on Hermitian Gram → real.
        state = _make_state(mu_T=0.1, min_basin=4)
        state.add_pattern()
        D = 16
        gen = torch.Generator().manual_seed(0)
        # Make 6 complex samples drawn from two clusters.
        phases_a = torch.rand(D, generator=gen) * 2 * math.pi
        phases_b = torch.rand(D, generator=gen) * 2 * math.pi
        a = torch.polar(torch.ones(D), phases_a)
        b = torch.polar(torch.ones(D), phases_b)
        for _ in range(3):
            noise = 0.05 * torch.randn(D, generator=gen)
            s = a * torch.polar(torch.ones(D), noise)
            state._basin_buffer.append((s, 0))
            noise2 = 0.05 * torch.randn(D, generator=gen)
            s2 = b * torch.polar(torch.ones(D), noise2)
            state._basin_buffer.append((s2, 0))
        signal = state._spatial_bimodality_signal(0)
        self.assertTrue(torch.isfinite(signal).item())
        self.assertGreaterEqual(float(signal), 0.0)
        self.assertLessEqual(float(signal), 1.0 + 1e-3)

    def test_small_basin_returns_on_device_tensor(self):
        # Watch-edge #5: floor returns 0 as tensor on same device/dtype.
        state = _make_state(mu_T=0.1, min_basin=4)
        state.add_pattern()
        D = 16
        # Only 2 members — below the floor.
        for _ in range(2):
            s = torch.randn(D)
            s = s / s.norm().clamp_min(1e-12)
            state._basin_buffer.append((s, 0))
        signal = state._spatial_bimodality_signal(0)
        self.assertIsInstance(signal, torch.Tensor)
        self.assertEqual(signal.device, state.device)
        self.assertEqual(signal.dtype, torch.float32)
        self.assertEqual(float(signal), 0.0)

    def test_tension_bounded_in_unit_interval(self):
        # Tension stays in [0, 1] after many EMA updates.
        state = _make_state(mu_T=0.3, min_basin=4)
        state.add_pattern()
        D = 32
        gen = torch.Generator().manual_seed(1)
        # Seed a strongly bimodal basin; repeatedly call
        # update_splitting_tension() and verify boundedness.
        for _ in range(8):
            v = torch.randn(D, generator=gen)
            v = v / v.norm().clamp_min(1e-12)
            state._basin_buffer.append((v, 0))
        for _ in range(50):
            state.update_splitting_tension()
        self.assertGreaterEqual(float(state.splitting_tension[0]), 0.0)
        self.assertLessEqual(float(state.splitting_tension[0]), 1.0 + 1e-3)


# ---------------------------------------------------------------------------
# Convergence-equivalence binding assertions
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch required")
class ConvergenceEquivalenceTests(unittest.TestCase):

    def test_assertion_1_tension_grows_on_bimodal_atom(self):
        codebook, _, samples = _build_codebook(K=4, D=32, seed=7,
                                               bimodal_atom=0,
                                               samples_per_basin=16)
        state_off = _make_state(mu_T=0.0, min_basin=4)
        for _ in range(4):
            state_off.add_pattern()
        _seed_state(state_off, samples)
        _run_consolidation_loop(
            state_off, codebook.clone(), _make_grouped(samples),
            n_steps=50, record_each_step=True,
        )
        T_off = state_off.splitting_tension.clone()

        # With mu_T = 0.1.
        codebook_on, _, samples_on = _build_codebook(K=4, D=32, seed=7,
                                                    bimodal_atom=0,
                                                    samples_per_basin=16)
        state_on = _make_state(mu_T=0.1, tau_T=0.5, min_basin=4)
        for _ in range(4):
            state_on.add_pattern()
        _seed_state(state_on, samples_on)
        _run_consolidation_loop(
            state_on, codebook_on, _make_grouped(samples_on),
            n_steps=50, record_each_step=True,
        )
        T_on = state_on.splitting_tension.clone()
        T0, T1, T2, T3 = (float(T_on[i]) for i in range(4))
        print(f"\n[A1] T_off={T_off.tolist()}, T_on={T_on.tolist()}")
        self.assertEqual(float(T_off[0]), 0.0,
                         "mu_T=0 did not preserve T at zero")
        self.assertGreater(T0, float(T_off[0]),
                           "tension did not accumulate on bimodal atom")
        self.assertGreater(T0, T1)
        self.assertGreater(T0, T2)
        self.assertGreater(T0, T3)

    def test_assertion_2_unimodal_tensions_stay_low(self):
        codebook, _, samples = _build_codebook(K=4, D=32, seed=11,
                                               bimodal_atom=0,
                                               samples_per_basin=16)
        state = _make_state(mu_T=0.1, tau_T=0.5, min_basin=4)
        for _ in range(4):
            state.add_pattern()
        _seed_state(state, samples)
        _run_consolidation_loop(
            state, codebook, _make_grouped(samples), n_steps=50,
            record_each_step=True,
        )
        for k in (1, 2, 3):
            self.assertLess(float(state.splitting_tension[k]), 0.2,
                            f"T_{k}={float(state.splitting_tension[k])} "
                            "exceeded 0.2 on unimodal basin")
        print(f"\n[A2] T_1={float(state.splitting_tension[1]):.4f} "
              f"T_2={float(state.splitting_tension[2]):.4f} "
              f"T_3={float(state.splitting_tension[3]):.4f}")

    def test_assertion_3_tension_trajectory_is_smooth(self):
        # Per-event Δ T_0 across the run: max/median ratio < 3.
        codebook, _, samples = _build_codebook(K=4, D=32, seed=13,
                                               bimodal_atom=0,
                                               samples_per_basin=16)
        state = _make_state(mu_T=0.1, tau_T=0.5, min_basin=4)
        for _ in range(4):
            state.add_pattern()
        _seed_state(state, samples)
        # Warm-up so EMA has equilibrated; this matches C.2.1 A2 discipline.
        _run_consolidation_loop(
            state, codebook, _make_grouped(samples),
            n_steps=200, record_each_step=True, contract=0.02,
        )
        # Now record per-event T_0 trajectory.
        traj, _ = _run_consolidation_loop(
            state, codebook, _make_grouped(samples),
            n_steps=20, record_each_step=True, contract=0.02,
            track_tension_atom=0,
        )
        diffs = [abs(traj[i] - traj[i - 1]) for i in range(1, len(traj))]
        median_d = statistics.median(diffs) if diffs else 0.0
        max_d = max(diffs) if diffs else 0.0
        if median_d < 1e-9:
            self.assertLess(max_d, 1e-6)
        else:
            self.assertLess(max_d / median_d, 3.0,
                            f"per-event max ΔT_0 / median = "
                            f"{max_d / median_d:.3f} >= 3")
        print(f"\n[A3] median ΔT_0={median_d:.4e} max ΔT_0={max_d:.4e} "
              f"ratio={max_d / (median_d + 1e-12):.3f}")

    def test_assertion_4_mu_T_zero_byte_identical(self):
        # A pre-C.2.2 run is a run where mu_T=0 from construction. Two
        # runs with mu_T=0 must produce byte-identical codebook states
        # regardless of whether T_k starts at zero or has been written
        # to (it never is when mu_T=0, but verify the early-exit).
        codebook_a, _, samples_a = _build_codebook(K=4, D=32, seed=3,
                                                   bimodal_atom=0,
                                                   samples_per_basin=16)
        state_a = _make_state(mu_T=0.0, min_basin=4)
        for _ in range(4):
            state_a.add_pattern()
        _seed_state(state_a, samples_a)
        _run_consolidation_loop(
            state_a, codebook_a, _make_grouped(samples_a),
            n_steps=50, record_each_step=True,
        )

        codebook_b, _, samples_b = _build_codebook(K=4, D=32, seed=3,
                                                   bimodal_atom=0,
                                                   samples_per_basin=16)
        state_b = _make_state(mu_T=0.0, min_basin=4)
        for _ in range(4):
            state_b.add_pattern()
        _seed_state(state_b, samples_b)
        # Force-write T_k to nonzero values to make sure mu_T=0
        # early-exits cleanly without consuming them.
        state_b.splitting_tension[:] = 0.5
        # The modulation must still be 1.0 (mu_T=0 path).
        for k in range(4):
            self.assertEqual(state_b.splitting_tension_modulation(k), 1.0)
        # Reset for the apples-to-apples byte-identity test.
        state_b.splitting_tension[:] = 0.0
        _run_consolidation_loop(
            state_b, codebook_b, _make_grouped(samples_b),
            n_steps=50, record_each_step=True,
        )
        self.assertTrue(torch.equal(codebook_a, codebook_b),
                        "mu_T=0 produced divergent codebook — early-exit leaky")
        print("\n[A4] codebook byte-identical at mu_T=0")

    def test_assertion_5_dual_patch_h6(self):
        # Dual patch: BOTH compute_bimodality_diagnostics AND
        # ContextBagHistory replaced with raising stubs. The C.2.2
        # dynamic must run unchanged.
        codebook_clean, _, samples_clean = _build_codebook(
            K=4, D=32, seed=5, bimodal_atom=0, samples_per_basin=16,
        )
        state_clean = _make_state(mu_T=0.1, tau_T=0.5, min_basin=4)
        for _ in range(4):
            state_clean.add_pattern()
        _seed_state(state_clean, samples_clean)
        _run_consolidation_loop(
            state_clean, codebook_clean, _make_grouped(samples_clean),
            n_steps=50, record_each_step=True,
        )
        T_clean = state_clean.splitting_tension.clone()

        import energy_memory.phase3.bimodality_diagnostic as bd
        orig_compute = bd.compute_bimodality_diagnostics
        orig_history = bd.ContextBagHistory

        def boom_compute(*a, **k):
            raise RuntimeError(
                "C.2.2 actuator must not call compute_bimodality_diagnostics — H11"
            )

        class BoomHistory:  # pragma: no cover - construction itself raises
            def __init__(self, *a, **k):
                raise RuntimeError(
                    "C.2.2 actuator must not touch ContextBagHistory — H11"
                )

        bd.compute_bimodality_diagnostics = boom_compute
        bd.ContextBagHistory = BoomHistory
        try:
            codebook_patched, _, samples_patched = _build_codebook(
                K=4, D=32, seed=5, bimodal_atom=0, samples_per_basin=16,
            )
            state_patched = _make_state(mu_T=0.1, tau_T=0.5, min_basin=4)
            for _ in range(4):
                state_patched.add_pattern()
            _seed_state(state_patched, samples_patched)
            _run_consolidation_loop(
                state_patched, codebook_patched, _make_grouped(samples_patched),
                n_steps=50, record_each_step=True,
            )
        finally:
            bd.compute_bimodality_diagnostics = orig_compute
            bd.ContextBagHistory = orig_history

        self.assertTrue(
            torch.equal(codebook_clean, codebook_patched),
            "C.2.2 codebook diverged under dual patch — H11 violated",
        )
        self.assertTrue(
            torch.equal(T_clean, state_patched.splitting_tension),
            "C.2.2 tension diverged under dual patch — H11 violated",
        )
        print("\n[A5] dual-patch H6: codebook + T_k byte-identical")

    def test_assertion_7_2x2_ablation_no_phase_transition(self):
        # Pathological basin: two tight clusters far apart (small per-
        # cluster trace, high inter-cluster λ_2/λ_1). Run 50 events each
        # over the 2x2 cells.
        def run_cell(lambda_ac, mu_T):
            codebook, _, samples = _build_codebook(
                K=4, D=32, seed=21, bimodal_atom=0, samples_per_basin=16,
                bimodal_kind="pathological",
            )
            state = _make_state(mu_T=mu_T, tau_T=0.5,
                                lambda_ac=lambda_ac, min_basin=4)
            for _ in range(4):
                state.add_pattern()
            _seed_state(state, samples)
            tension_traj, tr_traj = _run_consolidation_loop(
                state, codebook, _make_grouped(samples),
                n_steps=50, record_each_step=True,
                track_tension_atom=0, track_tr_atom=0,
            )
            return tension_traj, tr_traj

        cells = {
            "off_off":   run_cell(0.0, 0.0),
            "ac_only":   run_cell(0.5, 0.0),
            "T_only":    run_cell(0.0, 0.1),
            "both_on":   run_cell(0.5, 0.1),
        }

        def ratio(traj):
            diffs = [abs(traj[i] - traj[i - 1]) for i in range(1, len(traj))]
            med = statistics.median(diffs) if diffs else 0.0
            mx = max(diffs) if diffs else 0.0
            if med < 1e-12:
                return float("nan") if mx > 0 else 1.0
            return mx / med

        # For T_k: only mu_T>0 cells produce nonzero trajectory; the
        # off_off and ac_only T_traj are all zeros (ratio undefined).
        # Compare both-on against the single-mechanism cells whose
        # trajectory is *non-degenerate* (median > 0).
        ratios_T = {name: ratio(c[0]) for name, c in cells.items()}
        ratios_tr = {name: ratio(c[1]) for name, c in cells.items()}
        print(f"\n[A7] ratios T_0 = {ratios_T}")
        print(f"[A7] ratios tr(Σ_0) = {ratios_tr}")

        # Single-mechanism cells (max-ratio over each quantity that
        # actually moves in that cell).
        def safe(r):
            return r if math.isfinite(r) else 0.0

        # tr(Σ_0) moves in every cell. Take the worst single-mechanism
        # cell as the comparison baseline.
        single_tr = max(safe(ratios_tr["ac_only"]), safe(ratios_tr["T_only"]),
                        safe(ratios_tr["off_off"]))
        both_tr = safe(ratios_tr["both_on"])
        self.assertLessEqual(
            both_tr, single_tr + 1.0,
            f"both-on tr(Σ_0) ratio {both_tr:.3f} > worst single {single_tr:.3f} + 1.0",
        )

        # T_0 only moves in mu_T>0 cells. Compare both-on against T_only.
        single_T = safe(ratios_T["T_only"])
        both_T = safe(ratios_T["both_on"])
        self.assertLessEqual(
            both_T, single_T + 1.0,
            f"both-on T_0 ratio {both_T:.3f} > T-only ratio {single_T:.3f} + 1.0",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
