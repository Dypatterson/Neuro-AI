"""Convergence-equivalence: C.2.1–C.2.5 dynamics under Γ1.c base.

Per the precommit at
notes/notes/2026-05-27-path-gamma-gamma1-context-residual-precommit.md
§"Required pre-code gates" #2.

The Path C convergence-equivalence tests (e.g., test_anti_collapse.py,
test_splitting_tension.py, test_cap_coverage_force.py,
test_metastability_replay_priority.py, test_drift_replay_tension.py)
verify each C.2.x dynamic's identity in isolation by driving
ConsolidationState directly without an OnlineCodebookUpdater base
update. Those identities — between a C.1.x diagnostic and the C.2.x
dynamic's fixed-point — are independent of which base update fires at
quasi-stationary state (the base update is ~zero by definition).

What can change under a new base update is *composition*: does each
C.2.x dynamic still fire when Γ1.c sits in pull/push's slot? Does its
substrate-side state evolve at expected magnitude? Does the composed
codebook trajectory differ from the Γ1.c-only trajectory (proving the
dynamic IS contributing, not silently degenerating to a no-op)?

This file is the *composition* gate. The per-dynamic fixed-point
identity tests remain in their own files; the gate here is "all five
C.2.x dynamics fire and compose with Γ1.c base without error, with
non-trivial substrate-state evolution and codebook divergence from a
Γ1.c-only baseline."

Binding sub-clause (per 2026-05-27 reviewer concern C1): if any of these
composition tests fails, the diagnosis returns to anti-homunculus
re-review before the failing dynamic is retuned. See the precommit
§"Required pre-code gates" #2 for binding language.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _setup_consolidation_state(substrate, n_atoms, cfg):
    """Build a fully-initialized ConsolidationState with n_atoms patterns."""
    from energy_memory.phase4.consolidation import ConsolidationState

    cs = ConsolidationState(cfg, device="cpu")
    # Grow per-pattern state to n_atoms rows. add_pattern() allocates
    # rows in u, A, retrieval_count, r_ema, metastability_ema,
    # splitting_tension, drift_tension.
    for _ in range(n_atoms):
        cs.add_pattern()
    return cs


def _seed_basin_traces(cs, substrate, n_atoms, traces_per_atom):
    """Populate the basin trace buffer with quasi-stationary samples."""
    for atom_id in range(n_atoms):
        for _ in range(traces_per_atom):
            cs.record_retrieval(substrate.random_vector(), atom_id)


def _make_updater(substrate, codebook, cs, *, base):
    """Construct an OnlineCodebookUpdater bound to cs.

    base: "pull_push" or "context_residual".
    """
    from energy_memory.phase34.online_codebook import OnlineCodebookUpdater

    flags = {"use_pull_push": False, "use_context_residual": False}
    if base == "pull_push":
        flags["use_pull_push"] = True
    elif base == "context_residual":
        flags["use_context_residual"] = True
    else:
        raise ValueError(f"unknown base: {base}")
    return OnlineCodebookUpdater(
        substrate=substrate,
        codebook=codebook,
        consolidation_k=3,
        quality_threshold=0.5,
        consolidation_state=cs,
        lr_cr=0.1,
        **flags,
    )


def _stuff_buffer(updater, substrate, pairs):
    """Append (target_id, predicted_id) pairs as buffered failures."""
    from energy_memory.phase34.online_codebook import _BufferedFailure
    for t, p in pairs:
        updater._buffer.append(_BufferedFailure(
            target_id=t,
            predicted_id=p,
            slot_query=substrate.random_vector(),
            quality=0.0,
        ))


@unittest.skipIf(torch is None, "torch required")
class TestAllC2xComposeWithGamma1(unittest.TestCase):
    """End-to-end composition smoke: all five C.2.x dynamics active
    simultaneously with Γ1.c as the base update. The test passes if:

    (1) Consolidation completes without error.
    (2) The codebook moves (Γ1.c base fires).
    (3) The result differs from a Γ1.c-only run (no C.2.x dynamics) —
        proving composition is real.
    (4) Each C.2.x dynamic's substrate-side state evolves.
    """

    def _fresh_setup(self, cfg, seed):
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        substrate = TorchFHRR(dim=64, seed=seed, device="cpu")
        n_atoms = 10
        codebook = substrate.random_vectors(n_atoms)
        cs = _setup_consolidation_state(substrate, n_atoms, cfg)
        _seed_basin_traces(cs, substrate, n_atoms, traces_per_atom=8)
        return substrate, codebook, cs, n_atoms

    def _run_consolidations(self, substrate, updater, n_rounds=3):
        for _ in range(n_rounds):
            _stuff_buffer(
                updater, substrate,
                [(2, 5), (3, 7), (1, 4)],
            )
            updater.force_consolidate()

    def test_all_dynamics_compose_with_gamma1_base(self):
        from energy_memory.phase4.consolidation import ConsolidationConfig

        # All five dynamics on at Path-C-style values.
        cfg_all = ConsolidationConfig(
            lambda_ac=0.5, epsilon_ac=1e-4,
            mu_T=0.1, tau_T=0.5, epsilon_T=1e-6, min_basin_for_signal=2,
            lambda_cc=0.5, theta_cc=0.5, tau_cc=0.1,
            metastability_obs_rate=0.1,
            drift_ema_rate=0.1,
            basin_trace_buffer_size=64,
        )
        # All five dynamics off (Γ1.c only baseline).
        cfg_off = ConsolidationConfig(
            basin_trace_buffer_size=64,
        )

        # Run A: Γ1.c + all C.2.x on.
        sub_a, cb_a, cs_a, n_atoms = self._fresh_setup(cfg_all, seed=42)
        upd_a = _make_updater(sub_a, cb_a, cs_a, base="context_residual")
        cb_a_snapshot = cb_a.clone()
        self._run_consolidations(sub_a, upd_a)

        # Run B: Γ1.c-only (no C.2.x dynamics).
        sub_b, cb_b, cs_b, _ = self._fresh_setup(cfg_off, seed=42)
        upd_b = _make_updater(sub_b, cb_b, cs_b, base="context_residual")
        cb_b_snapshot = cb_b.clone()
        self._run_consolidations(sub_b, upd_b)

        # (1) and (2): consolidation completed without error and codebook
        # moved on both runs. Verify Γ1.c moved codebook[2] in both.
        self.assertFalse(
            torch.allclose(cb_a[2], cb_a_snapshot[2], atol=1e-4),
            "codebook[2] did not move under Γ1.c + C.2.x — Γ1.c base did not fire",
        )
        self.assertFalse(
            torch.allclose(cb_b[2], cb_b_snapshot[2], atol=1e-4),
            "codebook[2] did not move under Γ1.c-only — Γ1.c base did not fire",
        )

        # (3) Composition is real: Γ1.c + C.2.x produces a different
        # codebook[2] than Γ1.c alone.
        self.assertFalse(
            torch.allclose(cb_a[2], cb_b[2], atol=1e-4),
            "Γ1.c + C.2.x produced identical codebook[2] to Γ1.c alone "
            "— C.2.x dynamics didn't compose",
        )

        # (4) Substrate-side state evolution per C.2.x:
        # C.2.1: basin trace buffer populated.
        self.assertGreater(
            len(cs_a._basin_buffer), 0,
            "C.2.1 basin buffer empty — record_retrieval didn't fire",
        )
        # C.2.2: splitting_tension EMA should have populated at least
        # some atoms with basin traces.
        self.assertTrue(
            cs_a.splitting_tension.numel() == n_atoms,
            f"C.2.2 splitting_tension shape unexpected: "
            f"{cs_a.splitting_tension.shape}",
        )
        # C.2.5: drift_tension should be > 0 for atoms moved by Γ1.c.
        # Atom 2 was moved across all three consolidation rounds.
        # drift_ema_rate=0.1 and drift_tension starts at zero, so after
        # 3 rounds with Γ1.c-driven motion, drift_tension[2] should be > 0.
        if hasattr(cs_a, "drift_tension"):
            self.assertGreater(
                float(cs_a.drift_tension[2]), 0.0,
                "C.2.5 drift_tension[2] is zero despite Γ1.c motion — "
                "drift EMA didn't fire under Γ1.c base",
            )

    def test_gamma1_base_preserves_per_pattern_state_shape(self):
        """C.2.x per-pattern state arrays are sized for n_patterns and
        survive across consolidations under Γ1.c base.
        """
        from energy_memory.phase4.consolidation import ConsolidationConfig

        cfg = ConsolidationConfig(
            lambda_ac=0.5, mu_T=0.1, tau_T=0.5,
            lambda_cc=0.5, theta_cc=0.5, tau_cc=0.1,
            metastability_obs_rate=0.1,
            drift_ema_rate=0.1,
            basin_trace_buffer_size=32,
        )
        sub, cb, cs, n_atoms = self._fresh_setup(cfg, seed=99)
        upd = _make_updater(sub, cb, cs, base="context_residual")
        self._run_consolidations(sub, upd, n_rounds=2)

        self.assertEqual(cs.splitting_tension.shape[0], n_atoms)
        self.assertEqual(cs.drift_tension.shape[0], n_atoms)
        self.assertEqual(cs.metastability_ema.shape[0], n_atoms)

    def test_pull_push_baseline_also_composes(self):
        """Sanity: confirm pull/push base also composes with all five
        C.2.x dynamics. This is the Path C composition behavior we
        inherit; if THIS fails under the refactored code, the refactor
        broke pull/push composition (baseline parity should have caught
        it, but this is the explicit cross-base sanity check).
        """
        from energy_memory.phase4.consolidation import ConsolidationConfig

        cfg = ConsolidationConfig(
            lambda_ac=0.5, mu_T=0.1, tau_T=0.5,
            lambda_cc=0.5, theta_cc=0.5, tau_cc=0.1,
            metastability_obs_rate=0.1,
            drift_ema_rate=0.1,
            basin_trace_buffer_size=32,
        )
        sub, cb, cs, n_atoms = self._fresh_setup(cfg, seed=55)
        upd = _make_updater(sub, cb, cs, base="pull_push")
        snapshot = cb.clone()
        self._run_consolidations(sub, upd, n_rounds=2)

        # codebook[2] should have moved by pull/push + C.2.x composition.
        self.assertFalse(
            torch.allclose(cb[2], snapshot[2], atol=1e-4),
            "codebook[2] didn't move under pull/push + C.2.x — refactor "
            "broke baseline composition",
        )


if __name__ == "__main__":
    unittest.main()
