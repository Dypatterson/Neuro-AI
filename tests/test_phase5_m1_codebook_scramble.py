"""Tests for Phase 5 M1 codebook scramble controls."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestM1CodebookScramble(unittest.TestCase):

    def test_rowwise_coordinate_permutation_is_reproducible_and_norm_preserving(self):
        from scripts.phase5_m1_codebook_scramble import rowwise_coordinate_permutation

        codebook = torch.arange(60, dtype=torch.float32).reshape(5, 12)
        first = rowwise_coordinate_permutation(codebook, seed=4242)
        second = rowwise_coordinate_permutation(codebook, seed=4242)

        self.assertEqual(first.shape, codebook.shape)
        self.assertEqual(first.dtype, codebook.dtype)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.allclose(first.norm(dim=1), codebook.norm(dim=1)))
        self.assertFalse(torch.equal(first, codebook))

    def test_rowwise_coordinate_permutation_is_seed_sensitive(self):
        from scripts.phase5_m1_codebook_scramble import rowwise_coordinate_permutation

        codebook = torch.arange(60, dtype=torch.float32).reshape(5, 12)
        first = rowwise_coordinate_permutation(codebook, seed=4242)
        second = rowwise_coordinate_permutation(codebook, seed=4243)

        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.allclose(first.norm(dim=1), second.norm(dim=1)))

    def test_rowwise_coordinate_permutation_rejects_non_matrix(self):
        from scripts.phase5_m1_codebook_scramble import rowwise_coordinate_permutation

        with self.assertRaisesRegex(ValueError, "codebook must be a \\[N, D\\] tensor"):
            rowwise_coordinate_permutation(torch.arange(12, dtype=torch.float32))

    def test_scramble_codebook_rejects_unknown_method(self):
        from scripts.phase5_m1_codebook_scramble import scramble_codebook

        codebook = torch.arange(60, dtype=torch.float32).reshape(5, 12)

        with self.assertRaisesRegex(ValueError, "unsupported scramble method"):
            scramble_codebook(codebook, method="row_permutation")

    def test_rowwise_coordinate_permutation_breaks_synthetic_alignment(self):
        from energy_memory.phase5.m1_role_energy import RoleBindingStats
        from scripts.phase5_m1_codebook_scramble import rowwise_coordinate_permutation

        generator = torch.Generator(device="cpu")
        generator.manual_seed(123)
        codebook = torch.randn((16, 512), generator=generator, dtype=torch.float32)
        codebook = codebook / codebook.norm(dim=1, keepdim=True).clamp(min=1e-12)
        queries = codebook[[0, 1, 2, 3]].clone()

        aligned_density = RoleBindingStats._topk_cosine_density(
            queries,
            codebook,
            neighbor_k=1,
        ).mean()
        scrambled = rowwise_coordinate_permutation(codebook, seed=4242)
        scrambled_density = RoleBindingStats._topk_cosine_density(
            queries,
            scrambled,
            neighbor_k=1,
        ).mean()

        self.assertGreater(float(aligned_density), 0.8)
        self.assertLess(float(scrambled_density), 0.2)

    def test_write_scrambled_codebook_records_manifest(self):
        from energy_memory.phase2.persistence import save_codebook
        from scripts.phase5_m1_codebook_scramble import write_scrambled_codebook

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "codebook.pt"
            output_path = tmp_path / "scrambled.pt"
            manifest_path = tmp_path / "manifest.json"
            codebook = torch.arange(60, dtype=torch.float32).reshape(5, 12)
            save_codebook(codebook, input_path)

            manifest = write_scrambled_codebook(
                input_codebook=input_path,
                output=output_path,
                manifest=manifest_path,
                seed=4242,
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["method"], "rowwise_coordinate_permutation")
            self.assertEqual(manifest["seed"], 4242)
            self.assertEqual(manifest["shape"], [5, 12])
            self.assertEqual(manifest["dtype"], "torch.float32")
            self.assertLessEqual(manifest["row_norm_max_abs_diff"], 1e-6)
            self.assertIn("codebook_not_in_registry", manifest["registry_policy"])


if __name__ == "__main__":
    unittest.main()
