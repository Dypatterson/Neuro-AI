"""Tests for the Report 110/111 range replay downstream helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest


def _row(condition: str, scale: int, seed: int, **overrides):
    base = {
        "condition": condition,
        "scale": scale,
        "seed": seed,
        "candidates": 10.0,
        "provenance_cells": 8.0,
        "provenance_rect": 0.7,
        "stored_near_duplicate_rate": 1.0,
        "final_near_duplicate_rate": 1.0,
        "query_near_existing_rate": 0.0,
        "d_eff_final": 6.0,
        "heldout_top1": 0.01,
        "heldout_topk": 0.10,
        "heldout_cap_t05": 0.0,
    }
    base.update(overrides)
    return base


class TestRangeReplayDownstreamPrecommit(unittest.TestCase):

    def test_precommit_payload_freezes_lane_and_inputs(self):
        from scripts.phase5_prime_range_replay_downstream_precommit import (
            DEFAULT_CONDITIONS,
            build_precommit_payload,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codebook = root / "fake_codebook.pt"
            codebook.write_bytes(b"codebook")

            payload = build_precommit_payload(
                repo_root=root,
                codebook_path="fake_codebook.pt",
            )

        self.assertEqual(payload["report_id"], 110)
        self.assertEqual(payload["lane_id"], "range_shaped_replay_downstream_f9_v1")
        self.assertEqual(payload["conditions"].keys(), set(DEFAULT_CONDITIONS))
        self.assertFalse(payload["stop_criteria"]["phase5_escalation_allowed"])
        self.assertIn("n=3", payload["bridge_path_freeze"]["boundary"])
        self.assertIn("heldout_top1", payload["metrics"])


class TestRangeReplayDownstreamAnalysis(unittest.TestCase):

    def test_analysis_stops_when_only_presettle_novelty_moves(self):
        from scripts.phase5_prime_range_replay_downstream_analysis import (
            analyze_payload,
        )

        rows = []
        for seed in [1, 2, 3]:
            rows.append(_row("standard", 3, seed))
            rows.append(
                _row(
                    "range_postsettle",
                    3,
                    seed,
                    candidates=24.0,
                    provenance_cells=20.0,
                    provenance_rect=0.2,
                    d_eff_final=5.0,
                )
            )
            rows.append(
                _row(
                    "range_presettle",
                    3,
                    seed,
                    candidates=35.0,
                    provenance_cells=50.0,
                    provenance_rect=0.3,
                    stored_near_duplicate_rate=0.0,
                    d_eff_final=20.0,
                    heldout_top1=0.01,
                    heldout_topk=0.10,
                )
            )

        out = analyze_payload(
            results={"rows": rows, "config": {}},
            precommit={"lane_id": "test", "report_id": 110},
            bootstrap_samples=50,
        )

        self.assertEqual(
            out["decision"]["id"],
            "not_viable_current_range_replay_downstream_novelty_without_retrieval",
        )
        self.assertFalse(out["decision"]["phase5_delta_e_run_authorized"])
        self.assertIn("W3", out["decision"]["presettle_novelty_scales"])

    def test_analysis_marks_current_path_viable_only_with_downstream_movement(self):
        from scripts.phase5_prime_range_replay_downstream_analysis import (
            analyze_payload,
        )

        rows = []
        for seed in [1, 2, 3]:
            rows.append(_row("standard", 3, seed))
            rows.append(
                _row(
                    "range_postsettle",
                    3,
                    seed,
                    candidates=24.0,
                    provenance_cells=20.0,
                    provenance_rect=0.2,
                    stored_near_duplicate_rate=0.1,
                    d_eff_final=8.0,
                    heldout_top1=0.04,
                    heldout_topk=0.16,
                )
            )
            rows.append(
                _row(
                    "range_presettle",
                    3,
                    seed,
                    candidates=35.0,
                    provenance_cells=50.0,
                    provenance_rect=0.3,
                    stored_near_duplicate_rate=0.0,
                    d_eff_final=20.0,
                )
            )

        out = analyze_payload(
            results={"rows": rows, "config": {}},
            precommit={"lane_id": "test", "report_id": 110},
            bootstrap_samples=50,
        )

        self.assertEqual(
            out["decision"]["id"],
            "range_postsettle_viable_for_fresh_phase5_precommit",
        )
        self.assertIn("W3", out["decision"]["current_path_pass_scales"])


if __name__ == "__main__":
    unittest.main()
