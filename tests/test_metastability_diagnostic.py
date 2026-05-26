"""Tests for passive per-atom metastability EMA diagnostic (C.1.5)."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestMetastabilityDiagnostic(unittest.TestCase):

    def _build_state(self, obs_rate: float, n_patterns: int):
        from energy_memory.phase4.consolidation import (
            ConsolidationConfig, ConsolidationState,
        )
        cfg = ConsolidationConfig(m=4, alpha=0.25, metastability_obs_rate=obs_rate)
        state = ConsolidationState(cfg, device="cpu")
        for _ in range(n_patterns):
            state.add_pattern(novelty_strength=1.0)
        return state

    def test_obs_rate_zero_baseline(self):
        """T1: obs_rate=0.0 — EMA stays at zero (κ=0 control baseline)."""
        from energy_memory.phase3.metastability_diagnostic import (
            compute_metastability_diagnostics,
        )
        state = self._build_state(obs_rate=0.0, n_patterns=4)
        contribution = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        for _ in range(5):
            state.update_metastability(contribution)
        diag = compute_metastability_diagnostics(state)
        self.assertFalse(diag.obs_rate_active)
        self.assertEqual(diag.n_atoms, 4)
        for i in range(4):
            self.assertEqual(diag.per_atom[i], 0.0)
        self.assertEqual(diag.mean, 0.0)
        self.assertEqual(diag.max, 0.0)

    def test_obs_rate_positive_produces_nonzero_ema(self):
        """T2: obs_rate>0 — repeated positive contributions yield positive EMA."""
        from energy_memory.phase3.metastability_diagnostic import (
            compute_metastability_diagnostics,
        )
        state = self._build_state(obs_rate=0.1, n_patterns=4)
        contribution = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        for _ in range(10):
            state.update_metastability(contribution)
        diag = compute_metastability_diagnostics(state)
        self.assertTrue(diag.obs_rate_active)
        self.assertEqual(diag.n_atoms, 4)
        for i in range(4):
            self.assertGreater(diag.per_atom[i], 0.0)
        self.assertGreater(diag.mean, 0.0)
        self.assertGreater(diag.max, 0.0)
        # Expose values for the writeup.
        print(f"\n[T2] per_atom EMA after 10 updates (μ=0.1, c=[0.1,0.2,0.3,0.4]): {diag.per_atom}")
        print(f"[T2] mean={diag.mean:.6f} max={diag.max:.6f}")

    def test_ema_convergence(self):
        """T3: with constant input, EMA converges to that input."""
        from energy_memory.phase3.metastability_diagnostic import (
            compute_metastability_diagnostics,
        )
        state = self._build_state(obs_rate=0.5, n_patterns=4)
        target = [0.1, 0.2, 0.3, 0.4]
        contribution = torch.tensor(target, dtype=torch.float32)
        for _ in range(100):
            state.update_metastability(contribution)
        diag = compute_metastability_diagnostics(state)
        for i, t in enumerate(target):
            self.assertAlmostEqual(diag.per_atom[i], t, delta=0.01)
        print(f"\n[T3] per_atom EMA after 100 updates (μ=0.5, target=[0.1,0.2,0.3,0.4]): {diag.per_atom}")

    def test_empty_state(self):
        """T4: empty state — zero atoms, zero stats, flag reflects config."""
        from energy_memory.phase3.metastability_diagnostic import (
            compute_metastability_diagnostics,
        )
        state_off = self._build_state(obs_rate=0.0, n_patterns=0)
        diag_off = compute_metastability_diagnostics(state_off)
        self.assertEqual(diag_off.per_atom, {})
        self.assertEqual(diag_off.n_atoms, 0)
        self.assertEqual(diag_off.mean, 0.0)
        self.assertEqual(diag_off.max, 0.0)
        self.assertFalse(diag_off.obs_rate_active)

        state_on = self._build_state(obs_rate=0.1, n_patterns=0)
        diag_on = compute_metastability_diagnostics(state_on)
        self.assertEqual(diag_on.per_atom, {})
        self.assertEqual(diag_on.n_atoms, 0)
        self.assertTrue(diag_on.obs_rate_active)

    def test_payback_effect(self):
        """T5: metastability_payback decays a single atom's EMA proportionally."""
        from energy_memory.phase3.metastability_diagnostic import (
            compute_metastability_diagnostics,
        )
        state = self._build_state(obs_rate=0.5, n_patterns=4)
        contribution = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        for _ in range(20):
            state.update_metastability(contribution)
        diag_before = compute_metastability_diagnostics(state)
        m0_before = diag_before.per_atom[0]
        self.assertGreater(m0_before, 0.0)
        state.metastability_payback(idx=0, factor=0.5)
        diag_after = compute_metastability_diagnostics(state)
        self.assertAlmostEqual(diag_after.per_atom[0], m0_before * 0.5, places=6)
        # Other atoms untouched.
        for i in (1, 2, 3):
            self.assertAlmostEqual(diag_after.per_atom[i], diag_before.per_atom[i], places=6)


if __name__ == "__main__":
    unittest.main()
