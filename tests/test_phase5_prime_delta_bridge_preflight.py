"""Tests for the Phase 5' Delta E bridge preflight."""

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
class TestPhase5PrimeDeltaBridgePreflight(unittest.TestCase):

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

    def test_seed_state_exposes_scene_matrix_for_bridge(self):
        from energy_memory.phase5.bundle_first_scene_memory import (
            build_bundle_first_seed_state,
        )

        source = _tiny_source(seeds=[17], n_rows=6, k_roles=4, c_codebook=32)
        state = build_bundle_first_seed_state(
            seed=17,
            source=source,
            config=self._config(),
            device="cpu",
        )

        self.assertEqual(list(state.scene_matrix.shape), [6, 64])
        self.assertEqual(list(state.query_context_tokens.shape), [8, 64])
        self.assertEqual(list(state.roles.shape), [4, 64])
        self.assertEqual(list(state.content.shape), [32, 64])
        self.assertEqual(len(state.query_plan), 8)

    def test_bridge_preflight_is_probe_only_and_delta_e_computable(self):
        from scripts.phase5_prime_delta_bridge_preflight import (
            run_seed_bridge_preflight,
        )

        source = _tiny_source(seeds=[17], n_rows=6, k_roles=4, c_codebook=32)
        payload = run_seed_bridge_preflight(
            seed=17,
            source=source,
            source_path=Path("tiny_source.json"),
            source_sha="sha",
            config=self._config(),
            protocol_name="tiny_protocol",
            max_probe_cues=2,
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

        self.assertTrue(payload["framing"]["preflight_only"])
        self.assertTrue(payload["framing"]["no_new_headline"])
        self.assertTrue(payload["passes_all_criteria"])
        self.assertFalse(payload["production_headline_ready"])
        self.assertEqual(payload["config"]["score_bias"], None)
        self.assertEqual(payload["config"]["n_probe_cues"], 2)
        self.assertTrue(payload["pass_criteria"]["delta_e_computable_all_probes"])
        self.assertTrue(
            payload["diagnostic_summary"][
                "step3_energy_equals_raw_because_score_bias_none"
            ]
        )
        self.assertEqual(len(payload["probes"]), 2)


if __name__ == "__main__":
    unittest.main()
