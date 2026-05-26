"""Tests for the Phase 5' bridge-readout precommit artifact."""

from __future__ import annotations

import unittest
from pathlib import Path


class TestBridgeReadoutPrecommit(unittest.TestCase):

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
            max_iter=0,
            scene_token_weight=0.25,
            cooccurrence="repo_sample_natural",
            source_name="trajectory_native_provenance_context_trace",
        )

    def test_precommit_payload_is_static_and_label_free(self):
        from scripts.phase5_prime_bridge_readout_precommit import (
            build_bridge_readout_precommit,
        )

        payload = build_bridge_readout_precommit(
            config=self._config(),
            protocol_name="tiny_protocol",
            analysis_path=Path("analysis.json"),
            analysis_sha="sha",
            analysis_payload={
                "diagnosis": {"raw_scene_energy_v0_degenerate": True}
            },
            beta=10.0,
            gamma=0.5,
            k_main=2,
        )

        self.assertTrue(payload["passes_all_criteria"])
        self.assertFalse(payload["framing"]["retrieval_executed"])
        self.assertEqual(
            payload["readout"]["readout_id"],
            "cue_conditioned_scene_energy_v1",
        )
        self.assertIn("target atom label", payload["readout"]["forbidden_inputs"])
        self.assertIn("branch prior vector", payload["readout"]["forbidden_inputs"])
        self.assertEqual(len(payload["planned_cells"]), 9)
        self.assertTrue(payload["toy_sanity_checks"]["different_scene_not_saturated"])


if __name__ == "__main__":
    unittest.main()
