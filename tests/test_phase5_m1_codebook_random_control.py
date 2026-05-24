"""Tests for Phase 5 M1 random/unit codebook controls."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch required")
class TestM1RandomUnitCodebookControl(unittest.TestCase):

    def test_iid_unit_complex_codebook_is_reproducible_and_unit_magnitude(self):
        from scripts.phase5_m1_codebook_random_control import iid_unit_complex_codebook

        first = iid_unit_complex_codebook(rows=5, dim=12, seed=20260524)
        second = iid_unit_complex_codebook(rows=5, dim=12, seed=20260524)

        self.assertEqual(first.shape, (5, 12))
        self.assertEqual(first.dtype, torch.complex64)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.allclose(first.abs(), torch.ones_like(first.real)))
        self.assertTrue(
            torch.allclose(
                first.norm(dim=1),
                torch.full((5,), 12 ** 0.5),
            )
        )

    def test_iid_unit_complex_codebook_is_seed_sensitive(self):
        from scripts.phase5_m1_codebook_random_control import iid_unit_complex_codebook

        first = iid_unit_complex_codebook(rows=5, dim=12, seed=20260524)
        second = iid_unit_complex_codebook(rows=5, dim=12, seed=20260525)

        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.allclose(first.abs(), second.abs()))

    def test_iid_unit_complex_codebook_rejects_invalid_shape(self):
        from scripts.phase5_m1_codebook_random_control import iid_unit_complex_codebook

        with self.assertRaisesRegex(ValueError, "rows must be positive"):
            iid_unit_complex_codebook(rows=0, dim=12)
        with self.assertRaisesRegex(ValueError, "dim must be positive"):
            iid_unit_complex_codebook(rows=5, dim=0)

    def test_write_random_unit_codebook_records_manifest(self):
        from scripts.phase5_m1_codebook_random_control import write_random_unit_codebook

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_path = tmp_path / "random_unit.pt"
            manifest_path = tmp_path / "manifest.json"

            manifest = write_random_unit_codebook(
                output=output_path,
                manifest=manifest_path,
                rows=5,
                dim=12,
                seed=20260524,
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["method"], "iid_unit_complex_phases")
            self.assertEqual(manifest["seed"], 20260524)
            self.assertEqual(manifest["shape"], [5, 12])
            self.assertEqual(manifest["dtype"], "torch.complex64")
            self.assertLessEqual(abs(manifest["coordinate_abs_min"] - 1.0), 1e-6)
            self.assertLessEqual(abs(manifest["coordinate_abs_max"] - 1.0), 1e-6)
            self.assertIn("no corpus", manifest["lineage_policy"])
            self.assertIn("codebook_not_in_registry", manifest["registry_policy"])

    def test_write_random_unit_codebook_rejects_unknown_method(self):
        from scripts.phase5_m1_codebook_random_control import write_random_unit_codebook

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unsupported random-control method"):
                write_random_unit_codebook(
                    output=Path(tmp) / "random_unit.pt",
                    manifest=Path(tmp) / "manifest.json",
                    rows=5,
                    dim=12,
                    method="gaussian",
                )


if __name__ == "__main__":
    unittest.main()
