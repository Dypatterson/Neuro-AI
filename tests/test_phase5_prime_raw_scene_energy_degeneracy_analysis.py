"""Tests for the raw-scene energy degeneracy analysis."""

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
class TestRawSceneEnergyDegeneracyAnalysis(unittest.TestCase):

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

    def test_analysis_identifies_saturated_scene_energy(self):
        from scripts.phase5_prime_raw_scene_energy_degeneracy_analysis import (
            run_raw_scene_energy_degeneracy_analysis,
        )

        source = _tiny_source(seeds=[17], n_rows=6, k_roles=4, c_codebook=32)
        payload = run_raw_scene_energy_degeneracy_analysis(
            seed=17,
            source=source,
            source_path=Path("tiny_source.json"),
            source_sha="sha",
            cleanup_preflight_path=None,
            cleanup_preflight_sha=None,
            prior_gate_path=None,
            prior_gate_sha=None,
            smoke_artifact_path=None,
            smoke_artifact_sha=None,
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

        self.assertTrue(payload["framing"]["analysis_only"])
        self.assertTrue(payload["passes_all_criteria"])
        self.assertTrue(payload["diagnosis"]["raw_scene_energy_v0_degenerate"])
        self.assertEqual(payload["aggregate"]["total_branches"], 32)
        self.assertGreater(payload["aggregate"]["unique_top_scene_index_count"], 1)
        self.assertTrue(payload["pass_criteria"]["all_paired_delta_groups_zero"])
        self.assertTrue(payload["pass_criteria"]["all_final_top_scores_saturated"])
        self.assertTrue(payload["pass_criteria"]["all_energies_saturated_at_negative_one"])
        self.assertTrue(payload["pass_criteria"]["recomputed_energy_matches_branch_readback"])


if __name__ == "__main__":
    unittest.main()
