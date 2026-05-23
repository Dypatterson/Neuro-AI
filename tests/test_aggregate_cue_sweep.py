"""Tests for cue-regime sweep aggregation discipline."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _import_module():
    if "aggregate_cue_sweep" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "aggregate_cue_sweep", "scripts/aggregate_cue_sweep.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["aggregate_cue_sweep"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["aggregate_cue_sweep"]


def _cell(bns: float, cd: float, delta_e: float, hit_role: float) -> dict:
    return {
        "binding_noise_std": bns,
        "content_distortion": cd,
        "mean_delta_e_raw": delta_e,
        "mean_delta_e_step3": delta_e,
        "frac_positive_raw": 1.0 if delta_e > 0 else 0.0,
        "frac_role_lt_content_lt_random": 0.5,
        "frac_role_lt_content": 0.75,
        "frac_random_lowest": 0.25,
        "per_condition_basin_hit_rate": {
            "role": hit_role,
            "content": 0.1,
            "random": 0.0,
        },
        "per_condition_mean_role_target_rank": {
            "role": 2.0,
            "content": 5.0,
            "random": 9.0,
        },
    }


def _seed_doc(cells: list[dict]) -> dict:
    return {"cue_regime_sweep": {"cells": cells}}


class CueSweepAggregatorTests(unittest.TestCase):
    def test_render_does_not_invite_posthoc_best_cell_rerun(self):
        mod = _import_module()
        cells = [
            _cell(0.01, 0.0, 0.0062, 0.45),  # fake above-floor cell
            _cell(0.05, 0.2, 0.0004, 0.05),
        ]
        agg = mod._aggregate({17: {"cells": cells}, 11: {"cells": cells}})

        md = mod._render_markdown(agg)

        self.assertIn("No cue-regime cell", md)
        self.assertIn("does not graduate", md)
        self.assertIn("successor pre-committed cue distribution", md)
        self.assertNotIn("rerun full headline", md)
        self.assertNotIn("at that cell", md)

    def test_loader_reports_expected_missing_seeds_and_skips_non_sweeps(self):
        mod = _import_module()
        cells = [_cell(0.01, 0.0, 0.0001, 0.1)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "seed17.json").write_text(json.dumps(_seed_doc(cells)))
            (root / "seed11.json").write_text(json.dumps({"other": "shape"}))

            per_seed, missing = mod._load_per_seed_sweeps(
                root, expected_seeds=[17, 11, 23]
            )

        self.assertEqual(sorted(per_seed), [17])
        self.assertEqual(missing, [11, 23])


if __name__ == "__main__":
    unittest.main()
