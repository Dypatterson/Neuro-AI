"""Tests for the Phase 5' cue-conditioned condition-collapse analysis."""

from __future__ import annotations

import unittest


CELL_SPECS = [
    ("k1_content_g0.5", "k1", "content", "k1", "scene_store", 1, 0.5),
    ("k1_role_g0.5", "k1", "role", "k1", "scene_store", 1, 0.5),
    ("main_content_K4_g0.5", "main", "content", "main", "scene_store", 2, 0.5),
    ("main_role_K4_g0.5", "main", "role", "main", "scene_store", 2, 0.5),
    ("no_prior_content_K4_g0", "no_prior", "content", "no_prior", "scene_store", 2, 0.0),
    ("no_prior_role_K4_g0", "no_prior", "role", "no_prior", "scene_store", 2, 0.0),
    (
        "no_schema_store_content_K4_g0.5",
        "no_schema_store",
        "content",
        "no_schema_store",
        "content_codebook",
        2,
        0.5,
    ),
    (
        "no_schema_store_role_K4_g0.5",
        "no_schema_store",
        "role",
        "no_schema_store",
        "content_codebook",
        2,
        0.5,
    ),
    (
        "random_schema_K4_g0.5",
        "random_schema",
        "random",
        None,
        "random_scene_store",
        2,
        0.5,
    ),
]


def _branch(
    *,
    branch_id: int,
    top_scene_index: int,
    energy: float,
    prior_source: str = "content",
) -> dict:
    return {
        "branch_id": branch_id,
        "prior_source": prior_source,
        "schema_index": branch_id,
        "schema_atom_index": None,
        "top_scene_index": top_scene_index,
        "top_scene_score": 1.0,
        "cue_conditioned_scene_energy": energy,
        "raw_scene_energy_step3": -1.0,
    }


def _probe(*, probe_index: int, scene: int, branch_count: int, prior_source: str) -> dict:
    branches = [
        _branch(
            branch_id=0,
            top_scene_index=scene,
            energy=-0.25 - (0.01 * probe_index),
            prior_source=prior_source,
        )
    ]
    if branch_count > 1:
        branches.append(
            _branch(
                branch_id=1,
                top_scene_index=99,
                energy=-0.20 - (0.01 * probe_index),
                prior_source=prior_source,
            )
        )
    return {
        "probe_index": probe_index,
        "scene": scene,
        "branch_count": branch_count,
        "branches": branches,
    }


def _smoke_payload() -> dict:
    cell_results = {}
    for cell_id, family, prior, group, schema_source, k_main, gamma in CELL_SPECS:
        cell_results[cell_id] = {
            "cell_id": cell_id,
            "control_family": family,
            "prior_type": prior,
            "paired_delta_group": group,
            "schema_source": schema_source,
            "k_main": k_main,
            "gamma": gamma,
            "probes": [
                _probe(
                    probe_index=0,
                    scene=0,
                    branch_count=k_main,
                    prior_source=prior,
                ),
                _probe(
                    probe_index=1,
                    scene=1,
                    branch_count=k_main,
                    prior_source=prior,
                ),
            ],
        }
    return {
        "passes_all_criteria": True,
        "config": {
            "seed": 17,
            "n_probe_cues": 2,
        },
        "readout": {
            "readout_id": "cue_conditioned_scene_energy_v1",
            "delta_e": "Delta E = E_content-prior - E_role-prior",
        },
        "paired_delta": {
            "k1": {"delta_e_values": [0.0, 0.0]},
            "main": {"delta_e_values": [0.0, 0.0]},
            "no_prior": {"delta_e_values": [0.0, 0.0]},
            "no_schema_store": {"delta_e_values": [0.0, 0.0]},
        },
        "cell_results": cell_results,
    }


class TestConditionCollapseAnalysis(unittest.TestCase):

    def test_artifact_analysis_identifies_min_branch_collapse(self):
        from scripts.phase5_prime_condition_collapse_analysis import (
            analyze_condition_collapse,
        )

        payload = analyze_condition_collapse(_smoke_payload())

        self.assertTrue(payload["framing"]["analysis_only"])
        self.assertFalse(payload["framing"]["retrieval_executed"])
        self.assertTrue(payload["passes_all_criteria"])
        self.assertTrue(payload["diagnosis"]["condition_collapse_confirmed"])
        self.assertEqual(
            payload["diagnosis"]["collapse_mode"],
            "min_branch_target_scene_attractor_collapse",
        )
        self.assertEqual(payload["aggregate"]["total_cells"], 9)
        self.assertEqual(payload["aggregate"]["min_selection_target_rate"], 1.0)
        self.assertLess(payload["aggregate"]["branch_target_scene_rate"], 1.0)
        self.assertTrue(
            payload["pass_criteria"]["cue_conditioned_readout_non_saturated"]
        )
        self.assertTrue(payload["pass_criteria"]["raw_scene_energy_still_saturated"])
        self.assertTrue(
            payload["pass_criteria"]["all_probe_minima_collapse_to_one_scene_set"]
        )
        self.assertTrue(
            payload["pass_criteria"][
                "all_minimum_selections_equal_probe_target_scene"
            ]
        )


if __name__ == "__main__":
    unittest.main()
