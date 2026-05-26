"""Tests for the Phase 5' non-synthetic native provenance preflight."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_repo_sample(root: Path) -> None:
    notes = root / "notes"
    notes.mkdir(parents=True)
    base_tokens = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
        "theta", "iota", "kappa", "lambda", "mu", "nu", "xi",
        "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi",
        "chi", "psi", "omega",
    ]
    text = " ".join(base_tokens * 20)
    (root / "README.md").write_text(text, encoding="utf-8")
    for idx in range(3):
        rotated = base_tokens[idx:] + base_tokens[:idx]
        (notes / f"note_{idx}.md").write_text(
            " ".join(rotated * 20),
            encoding="utf-8",
        )


@unittest.skipIf(torch is None, "torch required")
class TestNonsyntheticNativePreflight(unittest.TestCase):

    def test_seed_preflight_builds_replay_native_rows_and_holds_out_query_role(self):
        from scripts.phase5_prime_nonsynthetic_native_preflight import (
            _load_source_windows,
            _seed_preflight,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_repo_sample(root)
            windows, summary = _load_source_windows(
                corpus_source="repo_sample",
                repo_root=root,
                c_codebook=32,
                k_roles=4,
            )
            self.assertGreaterEqual(len(windows), 6)
            self.assertEqual(summary["corpus_source"], "repo_sample")

            payload = _seed_preflight(
                seed=17,
                source_windows=windows,
                D=64,
                N=6,
                K_roles=4,
                C_codebook=32,
                context_roles=2,
                n_queries=16,
            )

            self.assertEqual(payload["support"]["used_rows"], 6)
            self.assertEqual(payload["support"]["replay_store_rows"], 6)
            self.assertEqual(payload["support"]["rows_invalid"], 0)
            self.assertEqual(payload["support"]["rows_too_short"], 0)
            self.assertEqual(payload["query_fraction_in_source_role_support"], 1.0)
            self.assertEqual(payload["query_fraction_in_own_observed_context"], 0.0)
            geom = payload["source_observed_context_to_full_context"]
            self.assertIn("diag_top1_rate", geom)
            self.assertIn("diag_mean_rank", geom)

    def test_main_writes_source_artifact_and_preflight_manifest_sha(self):
        from scripts import phase5_prime_nonsynthetic_native_preflight as preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            _write_repo_sample(root)
            out_dir = Path(tmp) / "out"
            source_path = out_dir / "source.json"
            preflight_path = out_dir / "preflight.json"

            argv = [
                "phase5_prime_nonsynthetic_native_preflight.py",
                "--repo-root",
                str(root),
                "--source-out",
                str(source_path),
                "--preflight-out",
                str(preflight_path),
                "--D",
                "64",
                "--N",
                "6",
                "--K_roles",
                "4",
                "--C_codebook",
                "32",
                "--context_roles",
                "2",
                "--n_queries",
                "16",
                "--seeds",
                "17",
                "11",
            ]
            with mock.patch("sys.argv", argv):
                self.assertEqual(preflight.main(), 0)

            source = json.loads(source_path.read_text())
            payload = json.loads(preflight_path.read_text())
            self.assertEqual(
                source["framing"]["source_family"],
                "trajectory_derived_native",
            )
            self.assertEqual(
                payload["source_manifest"]["source_artifact_sha256"],
                _sha256_file(source_path),
            )
            self.assertTrue(all(payload["pass_criteria"].values()))


if __name__ == "__main__":
    unittest.main()
