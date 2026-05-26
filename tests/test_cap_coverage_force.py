"""C.2.3 — Cap-coverage error gradient as a substrate dynamic.

Per notes/notes/2026-05-26-c23-cap-coverage-gradient-precommit.md. The
anti-homunculus reviewer's 5 binding watch-edges, of which 2 are
test-plan strengthenings:

  - A6 (binding part ii): the sigmoid transition width 4·τ_cc covers
    ≥ 10% of the IQR of the substrate's |θ_cc − sim_i| distribution.
    Catches "formally continuous but operationally a step."
  - A7 (binding 7b + 7c): three-way composition over C.2.1 × C.2.2 × C.2.3
    with superposition-distance + Δ-spike cap, not just max/median.

The actuator's substrate primitive is _basin_covariance — the same one
C.2.1 and C.2.2 read. Per H14 it NEVER reads
``src/energy_memory/phase2/metrics.py``.
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


def _unit(v):
    return v / v.norm().clamp_min(1e-12)


def _build_mixed_coverage_basin(
    D=32,
    n_covered=4,
    n_uncovered=4,
    covered_noise=0.02,
    seed=0,
):
    """Atom 0 at +x; basin members split into "covered" (sim ≈ 1 to atom)
    and "uncovered" (sim ≈ 0, orthogonal to atom).

    Returns (atom_0_state, samples) where samples is a list of
    (state, atom_idx) pairs with atom_idx=0.
    """
    gen = torch.Generator().manual_seed(seed)
    atom = torch.zeros(D)
    atom[0] = 1.0
    samples = []
    for _ in range(n_covered):
        s = atom + covered_noise * torch.randn(D, generator=gen)
        samples.append((_unit(s), 0))
    # Uncovered: orthogonal direction (+y axis with small +x bleed). Their
    # similarity to atom (+x) is near 0, well below θ_cc=0.5.
    for i in range(n_uncovered):
        s = torch.zeros(D)
        s[1] = 1.0
        s = s + 0.05 * torch.randn(D, generator=gen)
        samples.append((_unit(s), 0))
    return atom, samples


def _make_state(
    lambda_cc=0.0, theta_cc=0.5, tau_cc=0.1,
    lambda_ac=0.0, mu_T=0.0, tau_T=0.5,
    min_basin=4, buffer=256, device="cpu",
):
    from energy_memory.phase4.consolidation import (
        ConsolidationConfig, ConsolidationState,
    )
    cfg = ConsolidationConfig(
        m=3,
        lambda_ac=lambda_ac, epsilon_ac=1e-6,
        mu_T=mu_T, tau_T=tau_T, epsilon_T=1e-6,
        lambda_cc=lambda_cc, theta_cc=theta_cc, tau_cc=tau_cc,
        min_basin_for_signal=min_basin,
        basin_trace_buffer_size=buffer,
    )
    return ConsolidationState(cfg, device=device)


def _seed_state(state, samples):
    """Force-load buffer (bypasses any short-circuit in record_retrieval)."""
    for s, k in samples:
        state._basin_buffer.append((s.detach().clone(), int(k)))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch required")
class UnitTests(unittest.TestCase):

    def test_lambda_zero_returns_zero_tensor_with_same_shape(self):
        state = _make_state(lambda_cc=0.0)
        atom = torch.randn(32)
        force = state.cap_coverage_force(0, atom)
        self.assertEqual(force.shape, atom.shape)
        self.assertEqual(force.dtype, atom.dtype)
        self.assertTrue(torch.equal(force, torch.zeros_like(atom)))

    def test_force_zero_when_atom_coincides_with_all_members(self):
        state = _make_state(lambda_cc=0.5)
        D = 32
        atom = _unit(torch.randn(D))
        for _ in range(6):
            state._basin_buffer.append((atom.clone(), 0))
        force = state.cap_coverage_force(0, atom)
        self.assertTrue(torch.allclose(force, torch.zeros_like(atom), atol=1e-6))

    def test_force_direction_positive_x_when_members_at_plus_x(self):
        # Atom at origin (zero vector) in 4D, members are unit-positive +x
        # with cosine-similarity-to-zero undefined but our cap-coverage
        # formula uses a normalized similarity that is well-defined when
        # the atom has nonzero norm. Use a tiny norm in +z so similarity
        # of +x members to the atom is ~0, well below theta_cc=0.5.
        state = _make_state(lambda_cc=1.0, theta_cc=0.5, tau_cc=0.1)
        D = 4
        atom = torch.tensor([0.0, 0.0, 1.0, 0.0])  # +z direction
        # Members at +x: sim(member, atom) = 0, w_cc ≈ σ(0.5/0.1) = σ(5) ≈ 1.
        for _ in range(6):
            m = torch.tensor([1.0, 0.0, 0.0, 0.0])
            state._basin_buffer.append((m, 0))
        force = state.cap_coverage_force(0, atom)
        # Force = λ · mean(w · (m − atom)) ≈ 1.0 · 1.0 · ([1,0,0,0]-[0,0,1,0])
        #       = [1, 0, -1, 0]
        self.assertGreater(force[0].item(), 0.0)
        self.assertLess(force[2].item(), 0.0)

    def test_complex_fhrr_similarity_is_real(self):
        state = _make_state(lambda_cc=0.5)
        D = 32
        gen = torch.Generator().manual_seed(0)
        re = torch.randn(D, generator=gen)
        im = torch.randn(D, generator=gen)
        atom = torch.complex(re, im)
        atom = atom / atom.abs().clamp_min(1e-12)
        for _ in range(6):
            r2 = re + 0.5 * torch.randn(D, generator=gen)
            i2 = im + 0.5 * torch.randn(D, generator=gen)
            z = torch.complex(r2, i2)
            z = z / z.abs().clamp_min(1e-12)
            state._basin_buffer.append((z, 0))
        force = state.cap_coverage_force(0, atom)
        # Force is the same dtype as atom (complex). It must be finite.
        self.assertTrue(torch.is_complex(force))
        self.assertTrue(torch.isfinite(force.real).all())
        self.assertTrue(torch.isfinite(force.imag).all())

    def test_basin_below_min_returns_zero_on_device(self):
        state = _make_state(lambda_cc=0.5, min_basin=4)
        D = 16
        atom = _unit(torch.randn(D))
        # Only 2 members — below min_basin_for_signal.
        for _ in range(2):
            state._basin_buffer.append((_unit(torch.randn(D)), 0))
        force = state.cap_coverage_force(0, atom)
        self.assertEqual(force.shape, atom.shape)
        self.assertEqual(force.device, atom.device)
        self.assertTrue(torch.equal(force, torch.zeros_like(atom)))

    def test_empty_basin_returns_zero_on_device(self):
        state = _make_state(lambda_cc=0.5)
        atom = torch.randn(16)
        force = state.cap_coverage_force(0, atom)
        self.assertTrue(torch.equal(force, torch.zeros_like(atom)))


# ---------------------------------------------------------------------------
# Convergence-equivalence binding assertions (A1–A5)
# ---------------------------------------------------------------------------


def _run_consolidation_loop(
    state, codebook, grouped, n_steps,
    lr=0.05, contract=0.05,
    record_each_step=True,
    track_atom=None,
    use_anti_collapse=True,
    use_cap_coverage=True,
    use_splitting_tension=True,
    record_sim_gaps=False,
):
    """Synthetic consolidation: pull + (optional) anti-collapse +
    (optional) cap-coverage, attenuated by (optional) splitting tension.

    Order per precommit:
      update = pull
      update += anti_collapse_force
      update += cap_coverage_force
      post = atom + update
      blended = pre + modulation * (post - pre)
      atom <- normalize(blended)
    """
    pos_traj = []
    sim_gaps_all = []
    for _ in range(n_steps):
        if use_splitting_tension:
            state.update_splitting_tension()
        for k in sorted(grouped.keys()):
            members = torch.stack(grouped[k], dim=0)
            centroid = members.mean(dim=0)
            pre = codebook[k].detach().clone()
            update = lr * (centroid - codebook[k])
            if use_anti_collapse:
                update = update + state.anti_collapse_force(k, codebook[k])
            if use_cap_coverage:
                update = update + state.cap_coverage_force(k, codebook[k])
            post = codebook[k] + update
            if use_splitting_tension:
                modulation = state.splitting_tension_modulation(k)
                blended = pre + modulation * (post - pre)
            else:
                blended = post
            codebook[k] = blended / blended.norm().clamp_min(1e-12)
            grouped[k] = [
                (1 - contract) * m + contract * codebook[k] for m in grouped[k]
            ]
        if record_each_step:
            state._basin_buffer.clear()
            for k_, ms in grouped.items():
                for m in ms:
                    state._basin_buffer.append((m.detach().clone(), int(k_)))
        if track_atom is not None:
            pos_traj.append(codebook[track_atom].detach().clone())
        if record_sim_gaps:
            # Record |theta_cc - sim_i| for all members of all basins this event.
            for k_, ms in grouped.items():
                a = codebook[k_]
                a_n = a.norm().clamp_min(1e-12)
                for m in ms:
                    m_n = m.norm().clamp_min(1e-12)
                    sim = float((m @ a) / (m_n * a_n))
                    sim_gaps_all.append(abs(state.config.theta_cc - sim))
    return pos_traj, sim_gaps_all


def _make_grouped(samples):
    g = {}
    for s, k in samples:
        g.setdefault(k, []).append(s.clone())
    return g


@unittest.skipIf(torch is None, "torch required")
class ConvergenceEquivalenceTests(unittest.TestCase):

    def test_assertion_1_force_pulls_toward_uncovered(self):
        """A1: force has positive cosine with mean(uncovered) - atom."""
        D = 32
        atom_0, samples = _build_mixed_coverage_basin(
            D=D, n_covered=6, n_uncovered=6, seed=7,
        )
        state = _make_state(lambda_cc=0.5, theta_cc=0.5, tau_cc=0.1)
        _seed_state(state, samples)

        atom_state = _unit(atom_0)
        force = state.cap_coverage_force(0, atom_state)

        # Identify uncovered members empirically (those with sim < theta_cc).
        uncovered = []
        for s, k in samples:
            sim = float(s @ atom_state)
            if sim < state.config.theta_cc:
                uncovered.append(s)
        uncovered_mean = torch.stack(uncovered).mean(dim=0)
        target_dir = uncovered_mean - atom_state

        cos = float(
            (force @ target_dir)
            / (force.norm().clamp_min(1e-12) * target_dir.norm().clamp_min(1e-12))
        )
        mag = float(force.norm())
        print(f"\n[A1] cosine(force, mean_uncovered - atom) = {cos:.4f}, "
              f"|force|={mag:.4e}")
        self.assertGreater(cos, 0.0,
                           "cap-coverage force not aligned with uncovered direction")

    def test_assertion_2_zero_force_when_all_covered(self):
        """A2: force ≈ 0 when all members have sim > 0.9 (well above θ_cc=0.5)."""
        D = 32
        atom = _unit(torch.tensor([1.0] + [0.0] * (D - 1)))
        state = _make_state(lambda_cc=0.5, theta_cc=0.5, tau_cc=0.1)
        # All members close to atom: similarity > 0.9.
        for _ in range(8):
            s = atom + 0.05 * torch.randn(D)
            s = _unit(s)
            # Verify it's actually covered.
            self.assertGreater(float(s @ atom), 0.9)
            state._basin_buffer.append((s, 0))

        force = state.cap_coverage_force(0, atom)
        mag = float(force.norm())
        print(f"\n[A2] |force| with all-covered basin = {mag:.6e}")
        self.assertLess(mag, 1e-3,
                        f"|force|={mag} not < 1e-3 with all-covered basin")

    def test_assertion_3_trajectory_smoothness(self):
        """A3: max/median ratio of per-event Δ atom_0 position < 3.

        Static basin (no contraction); the smoothness measurement is then
        a property of cap_coverage_force itself, not of a co-evolving
        geometry feeding back into the force on the measurement window.
        """
        D = 32
        atom_0, samples = _build_mixed_coverage_basin(D=D, seed=11)
        state = _make_state(lambda_cc=0.5, theta_cc=0.5, tau_cc=0.1)
        codebook = _unit(atom_0).unsqueeze(0).clone()  # shape [1, D]
        _seed_state(state, samples)
        # Warm up so initial transient doesn't dominate.
        _run_consolidation_loop(
            state, codebook, _make_grouped(samples),
            n_steps=400, record_each_step=False,
            use_anti_collapse=False, use_splitting_tension=False,
            track_atom=None, contract=0.0, lr=0.02,
        )
        traj, _ = _run_consolidation_loop(
            state, codebook, _make_grouped(samples),
            n_steps=50, record_each_step=False,
            use_anti_collapse=False, use_splitting_tension=False,
            track_atom=0, contract=0.0, lr=0.02,
        )
        diffs = [float((traj[i] - traj[i - 1]).norm()) for i in range(1, len(traj))]
        median_d = statistics.median(diffs) if diffs else 0.0
        max_d = max(diffs) if diffs else 0.0
        ratio = max_d / (median_d + 1e-12)
        print(f"\n[A3] median Δatom_0={median_d:.4e}, max Δ={max_d:.4e}, "
              f"ratio={ratio:.3f}")
        if median_d < 1e-9:
            self.assertLess(max_d, 1e-6)
        else:
            self.assertLess(ratio, 3.0,
                            f"per-event max/median = {ratio:.3f} >= 3 (binding)")

    def test_assertion_4_lambda_zero_byte_identical(self):
        """A4: λ_cc=0 produces byte-identical codebook to a pre-C.2.3 run."""
        D = 32
        atom_0, samples_a = _build_mixed_coverage_basin(D=D, seed=3)

        # Run A: λ_cc=0, cap-coverage call still happens but force=0.
        state_a = _make_state(lambda_cc=0.0)
        codebook_a = _unit(atom_0).unsqueeze(0).clone()
        _seed_state(state_a, samples_a)
        _run_consolidation_loop(
            state_a, codebook_a, _make_grouped(samples_a),
            n_steps=50, record_each_step=True,
            use_anti_collapse=False, use_splitting_tension=False,
        )

        # Run B: λ_cc=0 but with use_cap_coverage=False (simulates pre-C.2.3).
        atom_0b, samples_b = _build_mixed_coverage_basin(D=D, seed=3)
        state_b = _make_state(lambda_cc=0.0)
        codebook_b = _unit(atom_0b).unsqueeze(0).clone()
        _seed_state(state_b, samples_b)
        _run_consolidation_loop(
            state_b, codebook_b, _make_grouped(samples_b),
            n_steps=50, record_each_step=True,
            use_anti_collapse=False, use_splitting_tension=False,
            use_cap_coverage=False,
        )

        self.assertTrue(
            torch.equal(codebook_a, codebook_b),
            "λ_cc=0 produced divergent codebook — early-exit is leaky",
        )
        print("\n[A4] codebook byte-identical at λ_cc=0")

    def test_assertion_5_phase2_metrics_monkey_patch_ineffective(self):
        """A5: monkey-patch phase2.metrics to raise; C.2.3 must run unchanged."""
        D = 32

        # Run 1: normal C.2.3.
        atom_0, samples_1 = _build_mixed_coverage_basin(D=D, seed=5)
        state_1 = _make_state(lambda_cc=0.5, theta_cc=0.5, tau_cc=0.1)
        codebook_1 = _unit(atom_0).unsqueeze(0).clone()
        _seed_state(state_1, samples_1)
        _run_consolidation_loop(
            state_1, codebook_1, _make_grouped(samples_1),
            n_steps=20, record_each_step=True,
            use_anti_collapse=False, use_splitting_tension=False,
        )

        # Run 2: monkey-patch phase2.metrics so any attribute access raises.
        import energy_memory.phase2.metrics as m2

        # Capture all non-dunder attributes; replace each with a raising stub.
        original_attrs = {}
        for name in list(vars(m2).keys()):
            if name.startswith("__"):
                continue
            original_attrs[name] = getattr(m2, name)

        def _boom(*args, **kwargs):
            raise RuntimeError(
                "C.2.3 actuator must not consume phase2.metrics — H14 violated."
            )

        try:
            for name in original_attrs:
                # Only replace callables / dataclasses so simple constants do
                # not trigger unrelated failures. The intent is to catch any
                # code path that *calls* into phase2.metrics.
                obj = original_attrs[name]
                if callable(obj):
                    setattr(m2, name, _boom)
            atom_0b, samples_2 = _build_mixed_coverage_basin(D=D, seed=5)
            state_2 = _make_state(lambda_cc=0.5, theta_cc=0.5, tau_cc=0.1)
            codebook_2 = _unit(atom_0b).unsqueeze(0).clone()
            _seed_state(state_2, samples_2)
            _run_consolidation_loop(
                state_2, codebook_2, _make_grouped(samples_2),
                n_steps=20, record_each_step=True,
                use_anti_collapse=False, use_splitting_tension=False,
            )
        finally:
            for name, obj in original_attrs.items():
                setattr(m2, name, obj)

        self.assertTrue(
            torch.equal(codebook_1, codebook_2),
            "C.2.3 codebook differs under phase2.metrics monkey-patch — H14 violated",
        )
        print("\n[A5] C.2.3 dynamic unchanged with phase2.metrics stubbed")


# ---------------------------------------------------------------------------
# A6 — strengthened smoothness + IQR (binding reviewer watch-edge)
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch required")
class StrengthenedSmoothnessTests(unittest.TestCase):

    def _build_wide_sim_basin(self, D=32, seed=17, samples_per_band=4):
        """Basin with members at four similarity bands relative to atom (+x):
        near 1.0, near 0.7, near 0.4, near 0.1. Produces a wide
        |θ_cc - sim| IQR (~0.3-0.4) — ideal for testing the A6-ii IQR
        check across τ_cc values.
        """
        gen = torch.Generator().manual_seed(seed)
        atom = torch.zeros(D)
        atom[0] = 1.0
        samples = []
        # Four bands; each member at the band's target similarity by mixing
        # +x and a fresh orthogonal direction.
        targets = [0.95, 0.7, 0.4, 0.1]
        for tgt in targets:
            for _ in range(samples_per_band):
                # Orthogonal-ish direction.
                v = torch.randn(D, generator=gen)
                v[0] = 0.0  # force orthogonal to atom on the dominant axis.
                v = _unit(v)
                # Project to target similarity: s = tgt·atom + sqrt(1-tgt²)·v.
                s = tgt * atom + math.sqrt(max(0.0, 1.0 - tgt * tgt)) * v
                s = _unit(s)
                samples.append((s, 0))
        return atom, samples

    def _smoothness_and_iqr(self, tau_cc):
        D = 32
        # Build a basin whose members span a broad similarity range to atom
        # (covered near 1, partially-covered around 0.7, around 0.4, around
        # 0.1) so the substrate's natural |θ_cc - sim| distribution has a
        # large IQR. A small τ_cc relative to that IQR makes the sigmoid
        # operationally a step; A6 part (ii) catches this directly.
        atom_0, samples = self._build_wide_sim_basin(D=D, seed=17)
        state = _make_state(lambda_cc=0.5, theta_cc=0.5, tau_cc=tau_cc)
        codebook = _unit(atom_0).unsqueeze(0).clone()
        _seed_state(state, samples)
        # Static basin warm-up (no contraction) so smoothness reads the
        # actuator dynamic, not a co-evolving geometry.
        _run_consolidation_loop(
            state, codebook, _make_grouped(samples),
            n_steps=400, record_each_step=False,
            use_anti_collapse=False, use_splitting_tension=False,
            contract=0.0, lr=0.02,
        )
        # Measurement run with gap recording.
        traj, sim_gaps = _run_consolidation_loop(
            state, codebook, _make_grouped(samples),
            n_steps=50, record_each_step=False,
            use_anti_collapse=False, use_splitting_tension=False,
            track_atom=0, contract=0.0, lr=0.02,
            record_sim_gaps=True,
        )
        diffs = [float((traj[i] - traj[i - 1]).norm()) for i in range(1, len(traj))]
        median_d = statistics.median(diffs) if diffs else 0.0
        max_d = max(diffs) if diffs else 0.0
        ratio = max_d / (median_d + 1e-12)

        # IQR of |θ_cc - sim_i| across all members across all events.
        sorted_gaps = sorted(sim_gaps)
        n = len(sorted_gaps)
        if n >= 4:
            q1 = sorted_gaps[n // 4]
            q3 = sorted_gaps[(3 * n) // 4]
            iqr = q3 - q1
        else:
            iqr = 0.0
        transition_width = 4.0 * tau_cc
        # "covers >= 10% of IQR" — interpret: transition_width / IQR >= 0.10.
        # (Sigmoid spans [0.05, 0.95] over 4·τ_cc on the sim axis; this is
        # the operationally-smooth region. If that region is small relative
        # to the substrate's similarity-gap IQR, the sigmoid acts as a step.)
        coverage_ratio = transition_width / (iqr + 1e-12)
        return ratio, iqr, transition_width, coverage_ratio

    def test_assertion_6_part_i_loosened_smoothness_small_tau(self):
        """A6 part (i): tau_cc=0.001 — max/median ratio < 5 (loosened)."""
        ratio, _, _, _ = self._smoothness_and_iqr(tau_cc=0.001)
        print(f"\n[A6-i] tau_cc=0.001 max/median ratio = {ratio:.3f}")
        self.assertLess(ratio, 5.0,
                        f"tau_cc=0.001 ratio={ratio:.3f} >= 5 — true step-like")

    def test_assertion_6_part_ii_iqr_check_default_passes(self):
        """A6 part (ii) — default tau_cc=0.1 satisfies IQR coverage ≥ 10%."""
        ratio_default, iqr_default, tw_default, cov_default = (
            self._smoothness_and_iqr(tau_cc=0.1)
        )
        print(f"\n[A6-ii default] tau_cc=0.1 IQR(|θ-sim|)={iqr_default:.4e}, "
              f"4·tau_cc={tw_default:.4e}, ratio={cov_default:.3f}, "
              f"max/median={ratio_default:.3f}")
        self.assertGreaterEqual(
            cov_default, 0.10,
            f"default tau_cc=0.1 covers {cov_default:.3f} of IQR — < 10%",
        )

        ratio_small, iqr_small, tw_small, cov_small = (
            self._smoothness_and_iqr(tau_cc=0.001)
        )
        print(f"[A6-ii small] tau_cc=0.001 IQR(|θ-sim|)={iqr_small:.4e}, "
              f"4·tau_cc={tw_small:.4e}, ratio={cov_small:.3f}, "
              f"max/median={ratio_small:.3f}")
        # Small tau_cc fails the IQR check (operationally a step).
        self.assertLess(
            cov_small, 0.10,
            f"tau_cc=0.001 unexpectedly covers {cov_small:.3f} of IQR "
            "— substrate's similarity gaps must be very tight",
        )


# ---------------------------------------------------------------------------
# A7 — strengthened three-way composition (binding 7a + 7b + 7c)
# ---------------------------------------------------------------------------


def _pathological_atom_basin(D=32, n_samples=12, seed=23):
    """A basin that is low-trace (tight inside) AND high-bimodality (two
    clusters) AND low-coverage (most members far from atom_0).

    atom_0 starts at +x; basin is two tight clusters offset toward +y and -y
    (so members are bimodal in the y-axis, all uncovered relative to +x).
    """
    gen = torch.Generator().manual_seed(seed)
    D_ = D
    atom = torch.zeros(D_)
    atom[0] = 1.0
    samples = []
    half = n_samples // 2
    for _ in range(half):
        s = atom.clone()
        s[1] += 1.2
        s = s + 0.01 * torch.randn(D_, generator=gen)
        samples.append((_unit(s), 0))
    for _ in range(n_samples - half):
        s = atom.clone()
        s[1] -= 1.2
        s = s + 0.01 * torch.randn(D_, generator=gen)
        samples.append((_unit(s), 0))
    return atom, samples


@unittest.skipIf(torch is None, "torch required")
class ThreeWayCompositionTests(unittest.TestCase):

    def _run_cell(self, on_ac, on_mu, on_cc, seed=23):
        atom_0, samples = _pathological_atom_basin(D=32, n_samples=12, seed=seed)
        state = _make_state(
            lambda_cc=(0.5 if on_cc else 0.0),
            lambda_ac=(0.3 if on_ac else 0.0),
            mu_T=(0.1 if on_mu else 0.0),
            tau_T=0.5,
            min_basin=4,
        )
        # state.add_pattern bookkeeping: only needed when mu_T or splitting
        # tension code paths touch per-pattern arrays. Add one entry so
        # splitting_tension[0] exists when mu_T > 0.
        state.add_pattern()
        codebook = _unit(atom_0).unsqueeze(0).clone()
        _seed_state(state, samples)
        # Warm-up.
        _run_consolidation_loop(
            state, codebook, _make_grouped(samples),
            n_steps=30, record_each_step=True,
            use_anti_collapse=on_ac,
            use_cap_coverage=on_cc,
            use_splitting_tension=on_mu,
            contract=0.02,
        )
        # Measurement run.
        traj, _ = _run_consolidation_loop(
            state, codebook, _make_grouped(samples),
            n_steps=50, record_each_step=True,
            use_anti_collapse=on_ac,
            use_cap_coverage=on_cc,
            use_splitting_tension=on_mu,
            track_atom=0, contract=0.02,
        )
        return traj

    def test_assertion_7_three_way_composition(self):
        # 2×2×2 cells over (λ_ac, μ_T, λ_cc) ∈ {0, on}³.
        cells = {}
        for on_ac in (0, 1):
            for on_mu in (0, 1):
                for on_cc in (0, 1):
                    traj = self._run_cell(bool(on_ac), bool(on_mu), bool(on_cc))
                    cells[(on_ac, on_mu, on_cc)] = traj

        def diffs(traj):
            return [float((traj[i] - traj[i - 1]).norm())
                    for i in range(1, len(traj))]

        def ratio(traj):
            d = diffs(traj)
            med = statistics.median(d) if d else 0.0
            mx = max(d) if d else 0.0
            return mx / (med + 1e-12), mx, med

        # Compute ratios for each cell.
        cell_stats = {}
        for key, traj in cells.items():
            r, mx, med = ratio(traj)
            cell_stats[key] = {"ratio": r, "max_delta": mx, "median_delta": med}

        print("\n[A7] cell ratios (max/median) and max-Δ:")
        for key in sorted(cells.keys()):
            cs = cell_stats[key]
            print(f"   λ_ac={key[0]}, μ_T={key[1]}, λ_cc={key[2]}: "
                  f"ratio={cs['ratio']:.3f}, max_Δ={cs['max_delta']:.4e}, "
                  f"median_Δ={cs['median_delta']:.4e}")

        # 7a: all-three-on ratio ≤ worst two-mechanism cell + 1.0.
        all_three = cell_stats[(1, 1, 1)]["ratio"]
        two_on_cells = [
            cell_stats[k]["ratio"] for k in cells.keys() if sum(k) == 2
        ]
        worst_two = max(two_on_cells)
        print(f"[A7a] all-three-on ratio={all_three:.3f}, "
              f"worst two-mechanism={worst_two:.3f}")
        self.assertLessEqual(
            all_three, worst_two + 1.0,
            f"all-three-on ratio {all_three:.3f} > worst-two {worst_two:.3f} + 1.0",
        )

        # 7b: superposition-distance — atom_0 trajectory in all-three-on cell
        # vs linear superposition of three single-on trajectories.
        baseline = cells[(0, 0, 0)]
        single_ac = cells[(1, 0, 0)]
        single_mu = cells[(0, 1, 0)]
        single_cc = cells[(0, 0, 1)]

        def superpose(singles_list, base):
            # Per-event: super = base + Σ (single_i - base).
            T = min(len(base), *(len(s) for s in singles_list))
            return [
                base[t] + sum((s[t] - base[t]) for s in singles_list)
                for t in range(T)
            ]

        all_three_traj = cells[(1, 1, 1)]
        super_three = superpose([single_ac, single_mu, single_cc], baseline)
        T3 = min(len(all_three_traj), len(super_three))
        l2_three = sum(
            float((all_three_traj[t] - super_three[t]).norm()) for t in range(T3)
        ) / max(T3, 1)

        # Reference: worst pair's deviation from its two-single-on superposition.
        pair_devs = []
        pair_keys = [
            ((1, 1, 0), [single_ac, single_mu]),
            ((1, 0, 1), [single_ac, single_cc]),
            ((0, 1, 1), [single_mu, single_cc]),
        ]
        for cell_key, singles in pair_keys:
            pair_traj = cells[cell_key]
            super_pair = superpose(singles, baseline)
            T2 = min(len(pair_traj), len(super_pair))
            dev = sum(
                float((pair_traj[t] - super_pair[t]).norm()) for t in range(T2)
            ) / max(T2, 1)
            pair_devs.append(dev)
        worst_pair_dev = max(pair_devs) if pair_devs else 0.0
        ratio_7b = l2_three / (worst_pair_dev + 1e-12)
        print(f"[A7b] all-three superposition L2/event={l2_three:.4e}, "
              f"worst pair superposition L2/event={worst_pair_dev:.4e}, "
              f"ratio={ratio_7b:.3f}")
        self.assertLessEqual(
            l2_three, 2.0 * worst_pair_dev + 1e-9,
            f"three-way superposition deviation {l2_three:.4e} > "
            f"2x worst pair {worst_pair_dev:.4e} — emergent phase transition",
        )

        # 7c: Δ-spike cap — no per-event Δ in all-three-on exceeds 1.5× the
        # worst per-event Δ from any single-on or two-on cell.
        all_three_max = cell_stats[(1, 1, 1)]["max_delta"]
        non_three_keys = [k for k in cells.keys() if sum(k) in (1, 2)]
        worst_non_three_max = max(cell_stats[k]["max_delta"] for k in non_three_keys)
        spike_ratio = all_three_max / (worst_non_three_max + 1e-12)
        print(f"[A7c] all-three max_Δ={all_three_max:.4e}, "
              f"worst non-three max_Δ={worst_non_three_max:.4e}, "
              f"ratio={spike_ratio:.3f}")
        self.assertLessEqual(
            all_three_max, 1.5 * worst_non_three_max + 1e-9,
            f"all-three max_Δ {all_three_max:.4e} > 1.5x worst non-three "
            f"{worst_non_three_max:.4e} — Δ-spike phase transition",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
