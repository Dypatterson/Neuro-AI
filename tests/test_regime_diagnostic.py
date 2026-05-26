import math
import unittest

import torch

from energy_memory.phase3.regime_diagnostic import (
    AtomRegime,
    CodebookRegimeDiagnostics,
    compute_codebook_regime_diagnostics,
    pairwise_fhrr_distance,
    pairwise_fhrr_similarity,
    participation_ratio,
    per_atom_regime_diagnostics,
    summary_stats,
)


def _unit_complex(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    z = torch.complex(re, im)
    # Per-component unit-magnitude FHRR normalization.
    return z / z.abs().clamp(min=1e-12)


def _random_unit_complex(d: int, generator: torch.Generator) -> torch.Tensor:
    re = torch.randn(d, generator=generator)
    im = torch.randn(d, generator=generator)
    return _unit_complex(re, im)


class RegimeDiagnosticTests(unittest.TestCase):
    def test_t1_synthetic_tight_regime(self):
        gen = torch.Generator().manual_seed(101)
        K, D = 8, 128
        base = _random_unit_complex(D, gen)
        atoms = []
        for _ in range(K):
            re_noise = 0.01 * torch.randn(D, generator=gen)
            im_noise = 0.01 * torch.randn(D, generator=gen)
            perturb = base + torch.complex(re_noise, im_noise)
            atoms.append(perturb / perturb.abs().clamp(min=1e-12))
        patterns = torch.stack(atoms, dim=0)

        diag = compute_codebook_regime_diagnostics(patterns, k_nn=4, beta=10.0)
        n_tight = diag.regime_counts.get("tight", 0)
        self.assertGreaterEqual(n_tight, K - 1, f"expected >= {K-1} tight, got counts={diag.regime_counts}")
        d_bar_mean = diag.summary["d_bar"]["mean"]
        self.assertLess(d_bar_mean, 0.05, f"tight d_bar mean={d_bar_mean} not small")
        # Record for the report.
        self._last_t1 = (d_bar_mean, diag.summary["d_eff"]["mean"], diag.regime_counts)

    def test_t2_synthetic_spread_regime(self):
        gen = torch.Generator().manual_seed(202)
        K, D = 8, 128
        atoms = [_random_unit_complex(D, gen) for _ in range(K)]
        patterns = torch.stack(atoms, dim=0)

        diag = compute_codebook_regime_diagnostics(patterns, k_nn=4, beta=10.0)
        n_spread = diag.regime_counts.get("spread", 0)
        self.assertGreaterEqual(n_spread, K - 1, f"expected >= {K-1} spread, got counts={diag.regime_counts}")
        d_bar_mean = diag.summary["d_bar"]["mean"]
        self.assertGreater(d_bar_mean, 0.1, f"spread d_bar mean={d_bar_mean} not large enough")
        self._last_t2 = (d_bar_mean, diag.summary["d_eff"]["mean"], diag.regime_counts)

    def test_t3_theta_prime_fn_injection(self):
        gen = torch.Generator().manual_seed(303)
        K, D = 8, 128
        # Build a codebook with intermediate d_bar (mild noise).
        base = _random_unit_complex(D, gen)
        atoms = []
        for _ in range(K):
            re_noise = 0.5 * torch.randn(D, generator=gen)
            im_noise = 0.5 * torch.randn(D, generator=gen)
            perturb = base + torch.complex(re_noise, im_noise)
            atoms.append(perturb / perturb.abs().clamp(min=1e-12))
        patterns = torch.stack(atoms, dim=0)

        diag_default = compute_codebook_regime_diagnostics(patterns, k_nn=4, beta=10.0)
        # All atoms should have theta_prime_used = 0.1.
        for atom in diag_default.per_atom.values():
            self.assertAlmostEqual(atom.theta_prime_used, 0.1, places=6)

        diag_custom = compute_codebook_regime_diagnostics(
            patterns, k_nn=4, beta=10.0, theta_prime_fn=lambda b: 0.5
        )
        for atom in diag_custom.per_atom.values():
            self.assertAlmostEqual(atom.theta_prime_used, 0.5, places=6)

        # Classification should change: with theta'=0.5, more atoms should be tight
        # than with theta'=0.1 (since the threshold is higher).
        tight_default = diag_default.regime_counts.get("tight", 0)
        tight_custom = diag_custom.regime_counts.get("tight", 0)
        self.assertGreaterEqual(
            tight_custom, tight_default,
            f"custom theta' should not decrease tight count: {tight_custom} vs {tight_default}",
        )

    def test_t4_regime_counts_sum(self):
        gen = torch.Generator().manual_seed(404)
        K, D = 12, 64
        atoms = [_random_unit_complex(D, gen) for _ in range(K)]
        patterns = torch.stack(atoms, dim=0)

        diag = compute_codebook_regime_diagnostics(patterns, k_nn=4, beta=10.0)
        total = sum(diag.regime_counts.values())
        self.assertEqual(total, K)
        self.assertEqual(len(diag.per_atom), K)

    def test_t5_single_atom_edge_case(self):
        gen = torch.Generator().manual_seed(505)
        D = 64
        patterns = _random_unit_complex(D, gen).unsqueeze(0)

        diag = compute_codebook_regime_diagnostics(patterns, k_nn=4, beta=10.0)
        self.assertEqual(len(diag.per_atom), 1)
        atom = diag.per_atom[0]
        self.assertIsInstance(atom, AtomRegime)
        self.assertIn(atom.regime, {"tight", "spread", "borderline"})
        self.assertTrue(math.isnan(atom.d_bar))
        self.assertEqual(sum(diag.regime_counts.values()), 1)

    def test_t6_complex_fhrr_tensors(self):
        gen = torch.Generator().manual_seed(606)
        K, D = 6, 64
        atoms = [_random_unit_complex(D, gen) for _ in range(K)]
        patterns = torch.stack(atoms, dim=0)

        sim = pairwise_fhrr_similarity(patterns)
        dist = pairwise_fhrr_distance(patterns)
        pr = participation_ratio(patterns)
        self.assertEqual(sim.shape, (K, K))
        self.assertEqual(dist.shape, (K, K))
        self.assertTrue(torch.isfinite(sim).all().item())
        self.assertTrue(torch.isfinite(dist).all().item())
        self.assertTrue(math.isfinite(pr))

        diag = compute_codebook_regime_diagnostics(patterns, k_nn=3, beta=10.0)
        for atom in diag.per_atom.values():
            self.assertTrue(math.isfinite(atom.d_bar))
            self.assertTrue(math.isfinite(atom.d_eff))
            self.assertTrue(math.isfinite(atom.theta_prime_used))

        stats = summary_stats([atom.d_bar for atom in diag.per_atom.values()])
        self.assertEqual(stats["n"], K)
        self.assertTrue(math.isfinite(stats["mean"]))

    def test_t7_matches_script_helper(self):
        # Verify the lifted module's per-atom math matches the script's local
        # _per_atom_diagnostics on the same input.
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from consolidation_geometry_diagnostic import _per_atom_diagnostics

        gen = torch.Generator().manual_seed(707)
        K, D = 6, 64
        patterns = torch.stack([_random_unit_complex(D, gen) for _ in range(K)], dim=0)

        script_out = _per_atom_diagnostics(patterns, k_nn=3)
        module_out = per_atom_regime_diagnostics(patterns, k_nn=3, beta=10.0)
        self.assertEqual(len(script_out), len(module_out))
        for entry in script_out:
            i = entry["atom_idx"]
            atom = module_out[i]
            if math.isfinite(entry["d_bar"]):
                self.assertAlmostEqual(entry["d_bar"], atom.d_bar, places=6)
            if math.isfinite(entry["d_eff"]):
                self.assertAlmostEqual(entry["d_eff"], atom.d_eff, places=4)


if __name__ == "__main__":
    unittest.main()
