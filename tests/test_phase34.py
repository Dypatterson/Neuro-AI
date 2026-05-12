"""Tests for Phase 3+4 integration components."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestOnlineCodebookUpdater(unittest.TestCase):

    def setUp(self):
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.phase34.online_codebook import OnlineCodebookUpdater

        self.substrate = TorchFHRR(dim=128, seed=17, device="cpu")
        self.codebook = self.substrate.random_vectors(20)
        self.updater = OnlineCodebookUpdater(
            substrate=self.substrate,
            codebook=self.codebook,
            lr_pull=0.1,
            lr_push=0.05,
            consolidation_k=5,
            quality_threshold=0.5,
        )

    def test_observation_above_threshold_does_not_buffer(self):
        # slot_query = exactly codebook[target] → quality = 1.0 → no buffer
        self.updater.observe(
            target_id=3,
            slot_query=self.codebook[3].clone(),
            predicted_id=3,
        )
        self.assertEqual(len(self.updater._buffer), 0)

    def test_observation_below_threshold_buffers(self):
        # slot_query = unrelated random vec → quality near 0 → buffer
        self.updater.observe(
            target_id=3,
            slot_query=self.substrate.random_vector(),
            predicted_id=7,
        )
        self.assertEqual(len(self.updater._buffer), 1)

    def test_consolidation_fires_at_k(self):
        for i in range(5):
            ready = self.updater.observe(
                target_id=i,
                slot_query=self.substrate.random_vector(),
                predicted_id=(i + 1) % 20,
            )
            if i < 4:
                self.assertFalse(ready)
            else:
                self.assertTrue(ready)
        diag = self.updater.consolidate_if_ready()
        self.assertIsNotNone(diag)
        self.assertEqual(diag["consolidation"], 1)
        self.assertEqual(len(self.updater._buffer), 0)

    def test_consolidate_if_ready_returns_none_below_k(self):
        self.updater.observe(
            target_id=3,
            slot_query=self.substrate.random_vector(),
            predicted_id=7,
        )
        self.assertIsNone(self.updater.consolidate_if_ready())

    def test_consolidation_moves_codebook(self):
        original = self.codebook[3].clone()
        for _ in range(5):
            self.updater.observe(
                target_id=3,
                slot_query=self.substrate.random_vector(),
                predicted_id=7,
            )
        self.updater.consolidate_if_ready()
        # codebook[3] should have moved (pulled toward the buffered queries)
        self.assertFalse(torch.allclose(original, self.codebook[3], atol=1e-3))

    def test_consolidation_preserves_unit_modulus(self):
        for _ in range(5):
            self.updater.observe(
                target_id=3,
                slot_query=self.substrate.random_vector(),
                predicted_id=7,
            )
        self.updater.consolidate_if_ready()
        mags = self.codebook.abs()
        self.assertTrue(
            torch.allclose(mags, torch.ones_like(mags), atol=1e-4),
            f"max magnitude error: {(mags - 1.0).abs().max().item()}",
        )


@unittest.skipIf(torch is None, "torch required")
class TestReencoding(unittest.TestCase):

    def setUp(self):
        from energy_memory.substrate.torch_fhrr import TorchFHRR
        from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
        from energy_memory.phase2.encoding import (
            build_position_vectors, encode_window,
        )
        self.substrate = TorchFHRR(dim=128, seed=17, device="cpu")
        self.positions = build_position_vectors(self.substrate, 3)
        self.codebook = self.substrate.random_vectors(20)
        self.memory = TorchHopfieldMemory(self.substrate)
        self.windows = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        for w in self.windows:
            self.memory.store(
                encode_window(self.substrate, self.positions, self.codebook, w),
                label=str(w),
            )

    def test_reencode_changes_patterns_after_codebook_drift(self):
        from energy_memory.phase34.reencoding import reencode_patterns

        original = [p.clone() for p in self.memory._patterns]
        # Significantly perturb codebook
        new_codebook = self.substrate.normalize(
            self.codebook + 0.5 * self.substrate.random_vectors(self.codebook.shape[0])
        )
        n = reencode_patterns(
            memory=self.memory,
            source_windows=self.windows,
            substrate=self.substrate,
            positions=self.positions,
            codebook=new_codebook,
        )
        self.assertEqual(n, 3)
        for i, w in enumerate(self.windows):
            self.assertFalse(
                torch.allclose(original[i], self.memory._patterns[i], atol=1e-2),
                f"pattern {i} did not change after re-encoding"
            )

    def test_reencode_skips_none_source(self):
        from energy_memory.phase34.reencoding import reencode_patterns

        source_windows = [self.windows[0], None, self.windows[2]]
        original_p1 = self.memory._patterns[1].clone()
        # Perturb codebook
        new_codebook = self.substrate.normalize(
            self.codebook + 0.5 * self.substrate.random_vectors(self.codebook.shape[0])
        )
        n = reencode_patterns(
            memory=self.memory,
            source_windows=source_windows,
            substrate=self.substrate,
            positions=self.positions,
            codebook=new_codebook,
        )
        self.assertEqual(n, 2)
        # Pattern 1 (None source) should be unchanged
        self.assertTrue(torch.allclose(original_p1, self.memory._patterns[1]))

    def test_codebook_drift_zero_for_identical(self):
        from energy_memory.phase34.reencoding import codebook_drift

        drift = codebook_drift(self.codebook, self.codebook.clone())
        self.assertLess(drift, 1e-5)

    def test_codebook_drift_nonzero_after_perturbation(self):
        from energy_memory.phase34.reencoding import codebook_drift

        perturbed = self.substrate.normalize(
            self.codebook + 0.3 * self.substrate.random_vectors(self.codebook.shape[0])
        )
        drift = codebook_drift(self.codebook, perturbed)
        self.assertGreater(drift, 0.01)


if __name__ == "__main__":
    unittest.main()
