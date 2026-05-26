"""Tests for the Phase 5' natural-source mini-matrix preflight planner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase5_prime_natural_source_mini_matrix_preflight import (
    DEFAULT_CONDITIONS,
    DEFAULT_CUE_NOISE,
    build_payload,
    main,
)


class TestNaturalSourceMiniMatrixPreflight(unittest.TestCase):

    def _load_default_payload(self) -> dict:
        source_path = Path("reports/phase5_prime_nonsynthetic_native_context_source.json")
        cleanup_path = Path(
            "reports/phase5_prime_natural_source_control_cleanup_preflight.json"
        )
        prior_gate_path = Path("reports/phase5_prime_nonsynthetic_native_gate.json")
        source = json.loads(source_path.read_text())
        cleanup = json.loads(cleanup_path.read_text())
        source_sha = cleanup["source_manifest"]["source_artifact_sha256"]
        cleanup_sha = "cleanup-sha-not-validated-by-builder"
        prior_gate_sha = cleanup["source_manifest"]["gate_artifact_sha256"]
        return build_payload(
            source=source,
            cleanup_preflight=cleanup,
            source_path=source_path,
            cleanup_path=cleanup_path,
            prior_gate_path=prior_gate_path,
            source_sha=source_sha,
            cleanup_sha=cleanup_sha,
            prior_gate_sha=prior_gate_sha,
            protocol_name="non_special_unique_target_freq_le_32",
            cue_noise=DEFAULT_CUE_NOISE,
            conditions=DEFAULT_CONDITIONS,
        )

    def test_default_payload_is_preflight_only_and_static(self):
        payload = self._load_default_payload()

        self.assertTrue(payload["framing"]["preflight_only"])
        self.assertTrue(payload["framing"]["no_candidate_control_retrieval"])
        self.assertTrue(payload["framing"]["no_delta_e_headline"])
        self.assertTrue(payload["passes_all_criteria"])
        self.assertEqual(payload["config"]["cue_noise_sweep"], [0.0, 0.05, 0.1, 0.15])
        self.assertEqual(payload["cleanup_protocol_summary"]["protocol_name"], "non_special_unique_target_freq_le_32")
        self.assertEqual(len(payload["planned_cells"]), 32)
        self.assertTrue(all(not cell["retrieval_run"] for cell in payload["planned_cells"]))
        self.assertTrue(all(cell["conditions_are_static"] for cell in payload["planned_cells"]))
        self.assertNotIn("aggregates", payload)
        self.assertNotIn("raw", payload)

    def test_role_negative_controls_have_zero_exact_opportunity(self):
        payload = self._load_default_payload()

        controls = payload["control_opportunity_summary"]
        for condition in [
            "random_role",
            "deranged_role",
            "fixedpoint_free_shuffled_role",
        ]:
            self.assertTrue(controls[condition]["passes_zero_exact_opportunity"])
            self.assertEqual(
                controls[condition]["same_scene_exact_rate"]["max"],
                0.0,
            )

    def test_no_scene_token_baseline_maps_to_candidate_without_scene_token(self):
        payload = self._load_default_payload()
        cells = [
            cell
            for cell in payload["planned_cells"]
            if cell["cell_kind"] == "no_scene_token_baseline"
        ]

        self.assertEqual(len(cells), 4)
        self.assertTrue(all(cell["gate_condition"] == "candidate" for cell in cells))
        self.assertTrue(all(not cell["scene_token"] for cell in cells))
        self.assertTrue(all(cell["scene_token_weight"] == 0.0 for cell in cells))
        self.assertTrue(all(cell["scene_token_source"] == "none" for cell in cells))

    def test_unknown_condition_is_rejected(self):
        source_path = Path("reports/phase5_prime_nonsynthetic_native_context_source.json")
        cleanup_path = Path(
            "reports/phase5_prime_natural_source_control_cleanup_preflight.json"
        )
        prior_gate_path = Path("reports/phase5_prime_nonsynthetic_native_gate.json")
        source = json.loads(source_path.read_text())
        cleanup = json.loads(cleanup_path.read_text())

        with self.assertRaisesRegex(ValueError, "unknown mini-matrix"):
            build_payload(
                source=source,
                cleanup_preflight=cleanup,
                source_path=source_path,
                cleanup_path=cleanup_path,
                prior_gate_path=prior_gate_path,
                source_sha=cleanup["source_manifest"]["source_artifact_sha256"],
                cleanup_sha="cleanup-sha",
                prior_gate_sha=cleanup["source_manifest"]["gate_artifact_sha256"],
                protocol_name="non_special_unique_target_freq_le_32",
                cue_noise=DEFAULT_CUE_NOISE,
                conditions=["candidate", "adaptive_best_of_n"],
            )

    def test_main_writes_preflight_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "mini_matrix_preflight.json"
            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "phase5_prime_natural_source_mini_matrix_preflight.py",
                    "--out",
                    str(out_path),
                ]
                rc = main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)
            payload = json.loads(out_path.read_text())
            self.assertTrue(payload["passes_all_criteria"])
            self.assertTrue(payload["framing"]["preflight_only"])


if __name__ == "__main__":
    unittest.main()
