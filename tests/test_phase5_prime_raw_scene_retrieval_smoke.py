"""Tests for the raw-scene Phase 5' retrieval smoke."""

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
class TestRawSceneRetrievalSmoke(unittest.TestCase):

    def _config(self):
        from energy_memory.phase5.bundle_first_scene_memory import BundleFirstConfig

        return BundleFirstConfig(
            D=64,
            N=6,
            K_roles=4,
            C_codebook=32,
            context_roles=2,
            n_queries=8,
            beta=10.0,
            max_iter=3,
            scene_token_weight=0.25,
            cooccurrence="repo_sample_natural",
            source_name="trajectory_native_provenance_context_trace",
        )

    def test_raw_scene_smoke_executes_all_precommitted_cells(self):
        from scripts.phase5_prime_raw_scene_retrieval_smoke import (
            run_raw_scene_retrieval_smoke,
        )

        source = _tiny_source(seeds=[17], n_rows=6, k_roles=4, c_codebook=32)
        payload = run_raw_scene_retrieval_smoke(
            seed=17,
            source=source,
            source_path=Path("tiny_source.json"),
            source_sha="sha",
            cleanup_preflight_path=None,
            cleanup_preflight_sha=None,
            prior_gate_path=None,
            prior_gate_sha=None,
            controls_preflight_path=None,
            controls_preflight_sha=None,
            config=self._config(),
            protocol_name="tiny_protocol",
            max_probes=2,
            beta=10.0,
            gamma=0.5,
            k_main=2,
            temperature=1.0,
            delta_energy=0.1,
            delta_state=0.3,
            delta_redundant=0.95,
            max_settling_iter=3,
            device="cpu",
        )

        self.assertTrue(payload["framing"]["retrieval_executed"])
        self.assertTrue(payload["framing"]["pilot_smoke_only"])
        self.assertTrue(payload["passes_all_criteria"])
        self.assertEqual(payload["config"]["seed"], 17)
        self.assertEqual(payload["config"]["n_probe_cues"], 2)
        self.assertEqual(payload["config"]["planned_cell_count"], 9)
        self.assertEqual(len(payload["cell_results"]), 9)
        self.assertIn("no_schema_store", payload["paired_delta"])
        self.assertTrue(payload["paired_delta"]["main"]["delta_e_computable"])
        self.assertTrue(payload["paired_delta"]["no_schema_store"]["delta_e_computable"])
        self.assertTrue(
            payload["pass_criteria"][
                "step3_energy_equals_raw_because_score_bias_none"
            ]
        )
        self.assertEqual(
            payload["interface_shapes"]["content_schema_bindings"],
            [32, 4, 64],
        )


if __name__ == "__main__":
    unittest.main()
