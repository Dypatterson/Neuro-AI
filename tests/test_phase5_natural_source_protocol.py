"""Tests for reusable Phase 5' natural-source protocol helpers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


class TestNaturalSourceProtocol(unittest.TestCase):

    def test_eligible_triples_exclude_special_and_same_row_duplicate_targets(self):
        from energy_memory.phase5.natural_source_protocol import (
            eligible_triples_for_seed,
        )

        rows = [
            [0, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 9, 11],
        ]
        selected_context_roles = [[1], [0], [1]]

        eligible = eligible_triples_for_seed(
            rows=rows,
            selected_context_roles=selected_context_roles,
            k_roles=4,
            cap=None,
        )

        self.assertEqual(len(eligible), 6)
        self.assertNotIn(0, {item["target_atom"] for item in eligible})
        self.assertNotIn(9, {item["target_atom"] for item in eligible})
        self.assertTrue(
            all(
                rows[item["scene"]].count(item["target_atom"]) == 1
                for item in eligible
            )
        )

    def test_select_queries_is_seeded_and_deterministic(self):
        from energy_memory.phase5.natural_source_protocol import select_queries

        eligible = [
            {
                "scene": scene,
                "known_role": scene % 2,
                "query_role": (scene + 1) % 4,
                "observed_roles": [scene % 2],
                "target_atom": 100 + scene,
                "target_frequency": scene + 1,
            }
            for scene in range(20)
        ]

        left = select_queries(
            seed=17,
            eligible=eligible,
            n_queries=8,
            cap=32,
            k_roles=4,
            n_rows=20,
        )
        right = select_queries(
            seed=17,
            eligible=eligible,
            n_queries=8,
            cap=32,
            k_roles=4,
            n_rows=20,
        )

        self.assertEqual(left, right)
        self.assertEqual(len(left), 8)

    @unittest.skipIf(torch is None, "torch required")
    def test_fixedpoint_free_shuffle_matches_report_092_seed17_control(self):
        from energy_memory.phase5.natural_source_protocol import (
            fixedpoint_free_shuffle,
        )

        perm = fixedpoint_free_shuffle(seed=17, k_roles=16)

        self.assertEqual(
            perm,
            [8, 0, 11, 2, 6, 12, 1, 4, 13, 5, 9, 14, 3, 7, 15, 10],
        )
        self.assertEqual(sorted(perm), list(range(16)))
        self.assertTrue(all(role != mapped for role, mapped in enumerate(perm)))

    @unittest.skipIf(torch is None, "torch required")
    def test_same_scene_opportunities_count_fixed_controls(self):
        from energy_memory.phase5.natural_source_protocol import (
            same_scene_opportunities,
        )

        rows = [
            [10, 10, 12, 13],
            [20, 21, 22, 23],
        ]
        selected = [
            {"scene": 0, "query_role": 0, "target_atom": 10},
            {"scene": 1, "query_role": 2, "target_atom": 22},
        ]

        opportunities = same_scene_opportunities(
            rows=rows,
            selected=selected,
            seed=17,
            k_roles=4,
        )

        self.assertIn("random_exact_rate", opportunities)
        self.assertIn("deranged_exact_rate", opportunities)
        self.assertIn("fixedpoint_free_shuffled_exact_rate", opportunities)
        self.assertEqual(sorted(opportunities["fixedpoint_free_shuffle"]), [0, 1, 2, 3])
        self.assertTrue(
            all(
                role != mapped
                for role, mapped in enumerate(opportunities["fixedpoint_free_shuffle"])
            )
        )

    @unittest.skipIf(torch is None, "torch required")
    def test_report_092_protocol_plan_parity(self):
        from energy_memory.phase5.natural_source_protocol import (
            protocol_for_frequency_cap,
            select_recommended_protocol,
        )

        source = json.loads(
            Path("reports/phase5_prime_nonsynthetic_native_context_source.json").read_text()
        )
        expected_payload = json.loads(
            Path(
                "reports/phase5_prime_natural_source_control_cleanup_preflight.json"
            ).read_text()
        )
        expected = next(
            protocol
            for protocol in expected_payload["protocols"]
            if protocol["protocol_name"] == "non_special_unique_target_freq_le_32"
        )

        protocol = protocol_for_frequency_cap(
            cap=32,
            source=source,
            vocab={"id_to_token": []},
            n_queries=512,
        )

        self.assertEqual(protocol["protocol_name"], expected["protocol_name"])
        self.assertEqual(protocol["pass_criteria"], expected["pass_criteria"])
        self.assertEqual(protocol["aggregate"], expected["aggregate"])
        self.assertEqual(
            protocol["selected_query_plan_by_seed"],
            expected["selected_query_plan_by_seed"],
        )
        self.assertEqual(select_recommended_protocol([protocol]), protocol["protocol_name"])

    def test_protocol_payload_and_sha_validation_are_fixed_contracts(self):
        from energy_memory.phase5.natural_source_protocol import (
            protocol_payload,
            source_with_protocol_plan,
            validate_cleanup_preflight,
        )

        preflight = {
            "recommended_protocol": "ok",
            "framing": {"preflight_only": True},
            "source_manifest": {
                "source_artifact_sha256": "source-sha",
                "gate_artifact_sha256": "gate-sha",
            },
            "protocols": [
                {
                    "protocol_name": "ok",
                    "passes_all_criteria": True,
                    "required_queries_per_seed": 2,
                    "selected_query_plan_by_seed": {"17": [{"scene": 0}]},
                },
                {"protocol_name": "bad", "passes_all_criteria": False},
            ],
        }

        protocol = protocol_payload(preflight, None)
        source = {"config": {"n_queries": 10}, "query_plan_by_seed": {}}
        planned = source_with_protocol_plan(source, protocol)

        self.assertEqual(planned["config"]["n_queries"], 2)
        self.assertEqual(planned["query_plan_by_seed"], {"17": [{"scene": 0}]})
        validate_cleanup_preflight(
            preflight,
            source_sha="source-sha",
            gate_sha="gate-sha",
        )
        with self.assertRaisesRegex(ValueError, "does not pass"):
            protocol_payload(preflight, "bad")
        with self.assertRaisesRegex(ValueError, "source SHA"):
            validate_cleanup_preflight(
                preflight,
                source_sha="wrong",
                gate_sha="gate-sha",
            )


if __name__ == "__main__":
    unittest.main()
