"""Tests for the Phase 5' non-synthetic native provenance gate."""

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


def _write_source_and_preflight(
    *,
    source_path: Path,
    preflight_path: Path,
    seeds: list[int],
    D: int,
    N: int,
    K_roles: int,
    C_codebook: int,
    context_roles: int,
    n_queries: int,
) -> None:
    from scripts.phase5_prime_nonsynthetic_native_gate import (
        COOCCURRENCE,
        SOURCE_FAMILY,
        SOURCE_KIND,
        SOURCE_NAME,
    )

    rows_by_seed = {}
    query_plan_by_seed = {}
    for seed in seeds:
        rows = [
            [int((seed + scene * K_roles + role) % C_codebook) for role in range(K_roles)]
            for scene in range(N)
        ]
        rows_by_seed[str(seed)] = rows
        query_plan_by_seed[str(seed)] = [
            {
                "scene": int(query_idx % N),
                "known_role": 0,
                "query_role": 2,
                "observed_roles": [0, 1],
            }
            for query_idx in range(n_queries)
        ]

    source_payload = {
        "framing": {
            "source_name": SOURCE_NAME,
            "source_family": SOURCE_FAMILY,
            "source_kind": SOURCE_KIND,
        },
        "config": {
            "D": D,
            "N": N,
            "K_roles": K_roles,
            "C_codebook": C_codebook,
            "context_roles": context_roles,
            "cooccurrence": COOCCURRENCE,
            "n_queries": n_queries,
            "seeds": seeds,
        },
        "source_rows_by_seed": rows_by_seed,
        "query_plan_by_seed": query_plan_by_seed,
    }
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(source_payload, indent=2, sort_keys=True) + "\n")
    source_sha = _sha256_file(source_path)

    preflight_payload = {
        "framing": {"source_name": SOURCE_NAME},
        "source_manifest": {
            "source_artifact_path": str(source_path),
            "source_artifact_sha256": source_sha,
        },
        "aggregate": {
            "support": {},
            "source_observed_context_to_full_context": {
                "diag_top1_rate_mean": 1.0,
            },
        },
        "pass_criteria": {
            "eligible_rows_ge_N": True,
            "used_rows_eq_N": True,
            "replay_store_rows_eq_N": True,
            "no_invalid_rows": True,
            "no_too_short_rows": True,
            "query_roles_in_source_support": True,
            "query_role_held_out_from_own_observed_context": True,
            "source_artifact_sha256_recorded": True,
            "geometry_reported": True,
        },
    }
    preflight_path.write_text(
        json.dumps(preflight_payload, indent=2, sort_keys=True) + "\n"
    )


@unittest.skipIf(torch is None, "torch required")
class TestNonsyntheticNativeGate(unittest.TestCase):

    def test_main_writes_gate_with_fixed_source_contract(self):
        from scripts import phase5_prime_nonsynthetic_native_gate as gate

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            source_path = out_dir / "source.json"
            preflight_path = out_dir / "preflight.json"
            gate_path = out_dir / "gate.json"
            seeds = [17, 11]
            _write_source_and_preflight(
                source_path=source_path,
                preflight_path=preflight_path,
                seeds=seeds,
                D=64,
                N=6,
                K_roles=4,
                C_codebook=32,
                context_roles=2,
                n_queries=8,
            )

            argv = [
                "phase5_prime_nonsynthetic_native_gate.py",
                "--source",
                str(source_path),
                "--preflight",
                str(preflight_path),
                "--out",
                str(gate_path),
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
                "8",
                "--seeds",
                "17",
                "11",
                "--conditions",
                "candidate",
                "content_cleanup_positive",
                "--device",
                "cpu",
                "--max_iter",
                "3",
            ]
            with mock.patch("sys.argv", argv):
                self.assertEqual(gate.main(), 0)

            payload = json.loads(gate_path.read_text())
            self.assertEqual(payload["framing"]["source_name"], gate.SOURCE_NAME)
            self.assertEqual(payload["config"]["cooccurrence"], gate.COOCCURRENCE)
            self.assertEqual(
                payload["source_manifest"]["source_artifact_sha256"],
                _sha256_file(source_path),
            )
            self.assertEqual(len(payload["aggregates"]), 2)
            for agg in payload["aggregates"].values():
                self.assertEqual(agg["n_total"], 16)
                self.assertEqual(len(agg["per_seed_top1"]), 2)
                self.assertEqual(len(agg["leave_one_seed_out_top1"]), 2)

    def test_validate_rejects_query_role_leakage(self):
        from scripts import phase5_prime_nonsynthetic_native_gate as gate

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            source_path = out_dir / "source.json"
            preflight_path = out_dir / "preflight.json"
            seeds = [17]
            _write_source_and_preflight(
                source_path=source_path,
                preflight_path=preflight_path,
                seeds=seeds,
                D=64,
                N=6,
                K_roles=4,
                C_codebook=32,
                context_roles=2,
                n_queries=8,
            )
            source = json.loads(source_path.read_text())
            source["query_plan_by_seed"]["17"][0]["observed_roles"] = [0, 2]

            args = mock.Mock(
                D=64,
                N=6,
                K_roles=4,
                C_codebook=32,
                context_roles=2,
                cooccurrence=gate.COOCCURRENCE,
                n_queries=8,
                seeds=seeds,
            )
            with self.assertRaisesRegex(ValueError, "leaks query_role"):
                gate._validate_source_artifact(source, args)


if __name__ == "__main__":
    unittest.main()
