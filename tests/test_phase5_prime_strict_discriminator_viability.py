"""Tests for strict discriminator viability summaries."""

from __future__ import annotations

import unittest


def _probe(content_energy, role_energy, random_energy, *, target=1, scenes=None):
    if scenes is None:
        scenes = {"content": 0, "role": 1, "random": 0}
    return {
        "scene": target,
        "conditions": {
            "content": {
                "bundle": {
                    "cue_conditioned_scene_energy": content_energy,
                    "top_scene_index": scenes["content"],
                },
                "greedy": {"cue_conditioned_scene_energy": content_energy},
                "softmax_weights": [0.25, 0.25, 0.25, 0.25],
                "all_branch_step3_energies_saturated": True,
            },
            "role": {
                "bundle": {
                    "cue_conditioned_scene_energy": role_energy,
                    "top_scene_index": scenes["role"],
                },
                "greedy": {"cue_conditioned_scene_energy": role_energy},
                "softmax_weights": [0.25, 0.25, 0.25, 0.25],
                "all_branch_step3_energies_saturated": True,
            },
            "random": {
                "bundle": {
                    "cue_conditioned_scene_energy": random_energy,
                    "top_scene_index": scenes["random"],
                },
                "greedy": {"cue_conditioned_scene_energy": random_energy},
                "softmax_weights": [0.25, 0.25, 0.25, 0.25],
                "all_branch_step3_energies_saturated": True,
            },
        },
    }


class TestStrictDiscriminatorViability(unittest.TestCase):

    def test_operating_point_requires_delta_and_control_separation(self):
        from scripts.phase5_prime_strict_discriminator_viability import (
            _summarize_operating_point,
        )

        summary = _summarize_operating_point(
            beta=30.0,
            gamma=0.5,
            magnitude_floor=0.0055,
            probe_results=[
                _probe(-0.20, -0.21, -0.20),
                _probe(-0.20, -0.21, -0.20),
                _probe(-0.20, -0.21, -0.20),
                _probe(-0.20, -0.21, -0.20),
            ],
        )

        self.assertTrue(summary["viable_operating_point"])
        self.assertTrue(
            summary["viability_criteria"][
                "mean_delta_e_bundle_ge_magnitude_floor"
            ]
        )
        self.assertTrue(summary["all_softmax_weights_uniform"])

    def test_operating_point_fails_when_role_matches_random(self):
        from scripts.phase5_prime_strict_discriminator_viability import (
            _summarize_operating_point,
        )

        summary = _summarize_operating_point(
            beta=30.0,
            gamma=0.5,
            magnitude_floor=0.0055,
            probe_results=[
                _probe(
                    -0.20,
                    -0.201,
                    -0.201,
                    scenes={"content": 0, "role": 0, "random": 0},
                )
                for _ in range(4)
            ],
        )

        self.assertFalse(summary["viable_operating_point"])
        self.assertFalse(
            summary["viability_criteria"][
                "mean_delta_e_bundle_ge_magnitude_floor"
            ]
        )
        self.assertFalse(
            summary["viability_criteria"]["role_bundle_target_scene_rate_gt_random"]
        )


if __name__ == "__main__":
    unittest.main()
