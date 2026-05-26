"""Tests for reusable Phase 5' bundle-first scene-memory mechanics."""

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
class TestBundleFirstSceneMemory(unittest.TestCase):

    def test_seed_condition_supports_no_scene_token_candidate(self):
        from energy_memory.phase5.bundle_first_scene_memory import (
            BundleFirstConfig,
            run_bundle_first_seed_condition,
        )

        source = _tiny_source(seeds=[17], n_rows=6, k_roles=4, c_codebook=32)
        config = BundleFirstConfig(
            D=64,
            N=6,
            K_roles=4,
            C_codebook=32,
            context_roles=2,
            n_queries=8,
            beta=30.0,
            max_iter=3,
            scene_token_weight=0.25,
            cooccurrence="repo_sample_natural",
            source_name="trajectory_native_provenance_context_trace",
        )

        result = run_bundle_first_seed_condition(
            condition="candidate",
            seed=17,
            source=source,
            source_path=Path("source.json"),
            source_sha="sha",
            config=config,
            cue_noise=0.0,
            scene_token_weight=0.0,
            device="cpu",
        )

        self.assertEqual(result.condition, "candidate")
        self.assertEqual(result.scene_token_weight, 0.0)
        self.assertEqual(result.n_queries, 8)
        self.assertEqual(result.source_name, config.source_name)

    def test_positive_conditions_and_aggregation_share_result_shape(self):
        from energy_memory.phase5.bundle_first_scene_memory import (
            BundleFirstConfig,
            aggregate_bundle_first_results,
            run_bundle_first_seed_condition,
        )

        seeds = [17, 11]
        source = _tiny_source(seeds=seeds, n_rows=6, k_roles=4, c_codebook=32)
        config = BundleFirstConfig(
            D=64,
            N=6,
            K_roles=4,
            C_codebook=32,
            context_roles=2,
            n_queries=8,
            beta=30.0,
            max_iter=3,
            scene_token_weight=0.25,
            cooccurrence="repo_sample_natural",
            source_name="trajectory_native_provenance_context_trace",
        )
        results = [
            run_bundle_first_seed_condition(
                condition="bundle_positive",
                seed=seed,
                source=source,
                source_path=Path("source.json"),
                source_sha="sha",
                config=config,
                cue_noise=0.0,
                device="cpu",
            )
            for seed in seeds
        ]

        aggregate = aggregate_bundle_first_results(results)

        self.assertEqual(aggregate["n_total"], 16)
        self.assertEqual(len(aggregate["per_seed_top1"]), 2)
        self.assertEqual(len(aggregate["leave_one_seed_out_top1"]), 2)
        self.assertIn("mean_scene_margin", aggregate)


if __name__ == "__main__":
    unittest.main()
