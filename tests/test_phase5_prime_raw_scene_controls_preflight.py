"""Tests for the raw-scene bridge controls preflight planner."""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


def _tiny_source(*, seeds: list[int], n_rows: int, k_roles: int, c_codebook: int) -> dict:
    rows_by_seed = {}
    query_plan_by_seed = {}
    for seed in seeds:
        rows = [
            [int((seed + scene * k_roles + role) % c_codebook) for role in range(k_roles)]
            for scene in range(n_rows)
        ]
        rows_by_seed[str(seed)] = rows
        query_plan_by_seed[str(seed)] = [
            {
                "scene": int(query_idx % n_rows),
                "known_role": 0,
                "query_role": 2,
                "observed_roles": [0, 1],
            }
            for query_idx in range(8)
        ]
    return {
        "source_rows_by_seed": rows_by_seed,
        "query_plan_by_seed": query_plan_by_seed,
    }


@unittest.skipIf(torch is None, "torch required")
class TestRawSceneControlsPreflight(unittest.TestCase):

    def _config(self):
        from energy_memory.phase5.bundle_first_scene_memory import BundleFirstConfig

        return BundleFirstConfig(
            D=64,
            N=6,
            K_roles=4,
            C_codebook=32,
            context_roles=2,
            n_queries=8,
            beta=30.0,
            max_iter=0,
            scene_token_weight=0.25,
            cooccurrence="repo_sample_natural",
            source_name="trajectory_native_provenance_context_trace",
        )

    def test_plans_required_control_families_without_retrieval(self):
        from scripts.phase5_prime_raw_scene_controls_preflight import (
            planned_control_cells,
        )

        cells = planned_control_cells(config=self._config())
        families = {cell["control_family"] for cell in cells}

        self.assertIn("paired_delta_main", families)
        self.assertIn("random_schema", families)
        self.assertIn("k1", families)
        self.assertIn("no_prior", families)
        self.assertIn("no_schema_store", families)
        self.assertTrue(all(cell["score_bias"] is None for cell in cells))
        self.assertTrue(all(not cell["retrieval_executed"] for cell in cells))

    def test_preflight_payload_is_static_and_flags_legacy_default_mismatch(self):
        from scripts.phase5_prime_raw_scene_controls_preflight import (
            run_controls_preflight,
        )

        source = _tiny_source(seeds=[17, 11], n_rows=6, k_roles=4, c_codebook=32)
        payload = run_controls_preflight(
            source=source,
            source_path=Path("tiny_source.json"),
            source_sha="sha",
            cleanup_preflight_path=None,
            cleanup_preflight_sha=None,
            prior_gate_path=None,
            prior_gate_sha=None,
            config=self._config(),
            protocol_name="tiny_protocol",
            seeds=[17, 11],
            shape_probe_seed=17,
            beta=30.0,
            gamma=0.5,
            k_main=4,
            device="cpu",
        )

        self.assertTrue(payload["framing"]["preflight_only"])
        self.assertTrue(payload["passes_all_criteria"])
        self.assertFalse(payload["retrieval_executed"])
        self.assertEqual(payload["planned_cell_count"], 9)
        self.assertTrue(payload["pass_criteria"]["no_schema_store_pair_planned"])
        self.assertTrue(payload["pass_criteria"]["query_plan_integrity_ok"])
        mismatch_fields = {
            row["field"] for row in payload["legacy_headline_default_mismatches"]
        }
        self.assertEqual(mismatch_fields, {"beta", "k_main"})
        self.assertEqual(payload["shape_probe"]["scene_matrix_shape"], [6, 64])


if __name__ == "__main__":
    unittest.main()
