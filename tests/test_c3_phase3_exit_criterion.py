"""Minimal unit tests for the C.3 Phase 3 exit-criterion driver.

These tests exercise:
  T1 — the driver module imports without error.
  T2 — ``compute_codebook_regime_diagnostics`` stratifies a tiny
       synthetic codebook the way the C.3 driver consumes it.
  T3 — the calibrated ``theta_prime_fn`` loads correctly from the JSON
       artifact at ``notes/emergent-codebook/theta_prime_calibration.json``.
  T4 — Recall@K computation on a tiny memorized landscape recovers the
       masked token from the standard codebook.
"""

from __future__ import annotations

import importlib
import unittest

import torch

from energy_memory.phase2.encoding import build_position_vectors, encode_window
from energy_memory.phase3.regime_diagnostic import (
    compute_codebook_regime_diagnostics,
)
from energy_memory.phase3.theta_prime_calibration import (
    load_theta_prime_calibration,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR

# Import the driver via importlib so the test passes even when invoked
# from outside the repo root (tests live in tests/, driver lives in
# experiments/, which is not on sys.path by default; we add it inside
# the test).
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


class TestC3DriverImport(unittest.TestCase):
    """T1 — the driver imports cleanly."""

    def test_import_driver(self):
        mod = importlib.import_module("c3_phase3_exit_criterion")
        self.assertTrue(hasattr(mod, "run"))
        self.assertTrue(hasattr(mod, "main"))
        # Public-shape helpers used inside the run loop.
        self.assertTrue(hasattr(mod, "_evaluate_recall_at_k"))
        self.assertTrue(hasattr(mod, "_aggregate_by_stratum"))


class TestRegimeDiagnosticIntegration(unittest.TestCase):
    """T2 — regime classification on a tiny synthetic codebook."""

    def test_tight_codebook_classified_tight(self):
        # Two near-identical atoms repeated several times → tight.
        torch.manual_seed(0)
        D = 64
        base = torch.complex(torch.cos(torch.zeros(D)), torch.sin(torch.zeros(D)))
        # Six near-duplicate copies plus tiny phase noise.
        codebook = torch.stack(
            [base * torch.exp(1j * 1e-4 * torch.randn(D)) for _ in range(6)],
            dim=0,
        ).to(torch.complex64)
        # Use beta=1.0 so default theta_prime = 1/beta = 1.0; anything
        # below d_bar=1.0 will be classified 'tight'.
        diag = compute_codebook_regime_diagnostics(codebook, k_nn=5, beta=1.0)
        # At least one atom should be classified — and most should be
        # 'tight' since the cluster is collapsed.
        self.assertEqual(len(diag.per_atom), 6)
        regimes = [a.regime for a in diag.per_atom.values()]
        self.assertGreater(regimes.count("tight"), 0)

    def test_spread_codebook_classified_spread(self):
        # I.i.d. random complex unit vectors → d_bar ≈ 1.0 → spread for
        # any theta_prime < 1.0.
        torch.manual_seed(1)
        D = 64
        N = 8
        phase = torch.rand(N, D) * 2.0 * 3.141592653589793
        codebook = torch.polar(torch.ones(N, D), phase).to(torch.complex64)
        # beta=10 → default theta_prime = 0.1, much smaller than the
        # spread cluster's d_bar ≈ 1.0.
        diag = compute_codebook_regime_diagnostics(codebook, k_nn=4, beta=10.0)
        regimes = [a.regime for a in diag.per_atom.values()]
        # Random codebook in low D should overwhelmingly classify as
        # 'spread'.
        self.assertGreater(regimes.count("spread"), len(regimes) // 2)


class TestThetaPrimeCalibrationLoad(unittest.TestCase):
    """T3 — calibrated theta_prime_fn loads from the JSON artifact."""

    def test_calibration_loads_and_returns_finite_values(self):
        fn = load_theta_prime_calibration()
        # The repo ships a calibration JSON at
        # notes/emergent-codebook/theta_prime_calibration.json, so this
        # MUST load on a fresh checkout.
        self.assertIsNotNone(fn, "calibration JSON not found on disk")
        # Exact calibrated betas.
        self.assertAlmostEqual(fn(0.01), 0.05, places=6)
        self.assertAlmostEqual(fn(0.1), 0.05, places=6)
        self.assertAlmostEqual(fn(1.0), 0.9, places=6)
        # In-range log-beta interpolation between 0.1 and 1.0:
        # both endpoints differ, so an in-between beta must lie between
        # them.
        mid = fn(0.5)
        self.assertGreaterEqual(mid, 0.05)
        self.assertLessEqual(mid, 0.9)


class TestRecallAtKOnTinyExample(unittest.TestCase):
    """T4 — Recall@K on memorized window recovers the masked token."""

    def test_recall_at_k_recovers_memorized_token(self):
        # Memorize exactly one window. With K large enough, the masked
        # token MUST be in the top-K when the same window is queried.
        torch.manual_seed(7)
        substrate = TorchFHRR(dim=512, seed=7, device="cpu")
        vocab_size = 16
        codebook = substrate.random_vectors(vocab_size)
        window_size = 4
        positions = build_position_vectors(substrate, window_size)
        # Window: [3, 7, 2, 11], mask the last position.
        window = [3, 7, 2, 11]
        masked_idx = window_size - 1

        # Memorize via the same path the driver uses.
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory

        memory = TorchHopfieldMemory[str](substrate)
        memory.store(encode_window(substrate, positions, codebook, window), label="w0")

        # Build the masked cue using the driver's logic — a mask
        # placeholder vector for the masked slot.
        mask_vector = substrate.random_vector()
        cue_atoms = [codebook[t] for t in window]
        cue_atoms[masked_idx] = mask_vector
        terms = [substrate.bind(positions[i], cue_atoms[i]) for i in range(window_size)]
        cue = substrate.bundle(terms)

        # Retrieve, unbind the masked position, rank codebook atoms.
        result = memory.retrieve(cue, beta=10.0, max_iter=12)
        slot_query = substrate.unbind(result.state, positions[masked_idx])
        scores = substrate.similarity_matrix(slot_query, codebook)
        # K=5 → 16/5 baseline if random; the true masked token (11)
        # should rank within the top-5 with one stored pattern and
        # only one binding to disambiguate.
        topk = torch.topk(scores, k=5).indices.detach().cpu().tolist()
        self.assertIn(11, topk)


if __name__ == "__main__":
    unittest.main()
