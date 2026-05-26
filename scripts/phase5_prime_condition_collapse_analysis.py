"""Analyze cue-conditioned Phase 5' condition collapse from Report 106.

This is an artifact-only readback over the seed-17/four-probe cue-conditioned
bridge smoke. It does not rerun retrieval, widen seeds, or change the headline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _energy(branch: dict) -> float:
    return float(branch["cue_conditioned_scene_energy"])


def _isclose(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def _ranked_branches(probe: dict) -> List[dict]:
    rows = []
    ordered = sorted(
        probe["branches"],
        key=lambda branch: (
            _energy(branch),
            int(branch["top_scene_index"]),
            int(branch["branch_id"]),
        ),
    )
    for rank, branch in enumerate(ordered, start=1):
        rows.append({
            "rank": rank,
            "branch_id": int(branch["branch_id"]),
            "prior_source": str(branch["prior_source"]),
            "schema_index": int(branch["schema_index"]),
            "schema_atom_index": branch.get("schema_atom_index"),
            "top_scene_index": int(branch["top_scene_index"]),
            "top_equals_probe_scene": int(branch["top_scene_index"])
            == int(probe["scene"]),
            "cue_conditioned_scene_energy": _energy(branch),
            "raw_scene_energy_step3": float(branch["raw_scene_energy_step3"]),
            "top_scene_score": float(branch["top_scene_score"]),
        })
    return rows


def _min_winners(probe: dict) -> List[dict]:
    ranked = _ranked_branches(probe)
    if not ranked:
        return []
    best_energy = ranked[0]["cue_conditioned_scene_energy"]
    return [
        row for row in ranked if _isclose(row["cue_conditioned_scene_energy"], best_energy)
    ]


def _paired_delta_values(smoke_payload: dict) -> List[float]:
    values: List[float] = []
    for group in smoke_payload.get("paired_delta", {}).values():
        values.extend(
            float(value)
            for value in group.get("delta_e_values", [])
            if value is not None
        )
    return values


def _sorted_cells(cell_results: Dict[str, dict]) -> Iterable[tuple[str, dict]]:
    return sorted(cell_results.items(), key=lambda item: item[0])


def analyze_condition_collapse(
    smoke_payload: dict,
    *,
    smoke_artifact_path: Optional[Path] = None,
    smoke_artifact_sha: Optional[str] = None,
) -> dict:
    cell_results = smoke_payload["cell_results"]
    n_probe_cues = int(smoke_payload["config"]["n_probe_cues"])
    expected_probe_indices = list(range(n_probe_cues))

    cell_summaries = {}
    branch_scene_counts: Dict[str, int] = {}
    branch_energy_values: List[float] = []
    raw_scene_energy_values: List[float] = []
    total_branches = 0
    target_branch_hits = 0
    min_selection_rows = []

    for cell_id, cell in _sorted_cells(cell_results):
        probe_summaries = []
        for probe in cell["probes"]:
            probe_index = int(probe["probe_index"])
            target_scene = int(probe["scene"])
            ranked = _ranked_branches(probe)
            winners = _min_winners(probe)
            if not winners:
                raise ValueError(f"cell {cell_id} probe {probe_index} returned no branches")
            scene_counts: Dict[str, int] = {}
            for row in ranked:
                scene_key = str(row["top_scene_index"])
                scene_counts[scene_key] = scene_counts.get(scene_key, 0) + 1
                branch_scene_counts[scene_key] = branch_scene_counts.get(scene_key, 0) + 1
                branch_energy_values.append(row["cue_conditioned_scene_energy"])
                raw_scene_energy_values.append(row["raw_scene_energy_step3"])
                total_branches += 1
                target_branch_hits += int(row["top_equals_probe_scene"])

            best_energy = winners[0]["cue_conditioned_scene_energy"]
            best_scene_indices = sorted({row["top_scene_index"] for row in winners})
            min_selection_rows.append({
                "cell_id": cell_id,
                "probe_index": probe_index,
                "target_scene": target_scene,
                "min_energy": best_energy,
                "min_top_scene_indices": best_scene_indices,
                "min_top_scene_equals_target": best_scene_indices == [target_scene],
                "n_tied_min_branches": len(winners),
            })
            probe_summaries.append({
                "probe_index": probe_index,
                "target_scene": target_scene,
                "branch_count": len(ranked),
                "branch_scene_counts": scene_counts,
                "min_energy": best_energy,
                "min_top_scene_indices": best_scene_indices,
                "min_top_scene_equals_target": best_scene_indices == [target_scene],
                "ranked_branches": ranked,
            })
        cell_summaries[cell_id] = {
            "cell_id": cell_id,
            "control_family": cell["control_family"],
            "prior_type": cell["prior_type"],
            "paired_delta_group": cell["paired_delta_group"],
            "schema_source": cell["schema_source"],
            "k_main": int(cell["k_main"]),
            "gamma": float(cell["gamma"]),
            "probe_indices": [row["probe_index"] for row in probe_summaries],
            "min_top_scene_indices_by_probe": [
                row["min_top_scene_indices"] for row in probe_summaries
            ],
            "min_top_scene_equals_target_all_probes": all(
                row["min_top_scene_equals_target"] for row in probe_summaries
            ),
            "probes": probe_summaries,
        }

    per_probe = []
    for probe_index in expected_probe_indices:
        rows = [
            row
            for row in min_selection_rows
            if int(row["probe_index"]) == int(probe_index)
        ]
        if not rows:
            raise ValueError(f"missing probe {probe_index}")
        target_scenes = sorted({int(row["target_scene"]) for row in rows})
        min_scene_sets = [tuple(row["min_top_scene_indices"]) for row in rows]
        unique_min_scene_sets = sorted({scene_set for scene_set in min_scene_sets})
        per_probe.append({
            "probe_index": probe_index,
            "target_scenes": target_scenes,
            "cell_count": len(rows),
            "unique_min_top_scene_sets": [list(scene_set) for scene_set in unique_min_scene_sets],
            "unique_min_scene_set_count": len(unique_min_scene_sets),
            "all_cells_choose_same_min_scene": len(unique_min_scene_sets) == 1,
            "all_min_scenes_equal_target": all(
                row["min_top_scene_equals_target"] for row in rows
            ),
            "min_energy_values": [row["min_energy"] for row in rows],
            "unique_min_energy_values": sorted(
                {round(float(row["min_energy"]), 15) for row in rows}
            ),
        })

    paired_delta_values = _paired_delta_values(smoke_payload)
    branch_target_scene_rate = (
        target_branch_hits / total_branches if total_branches else math.nan
    )
    min_selection_target_hits = sum(
        int(row["min_top_scene_equals_target"]) for row in min_selection_rows
    )
    min_selection_target_rate = (
        min_selection_target_hits / len(min_selection_rows)
        if min_selection_rows
        else math.nan
    )
    all_raw_scene_energies_saturated = all(
        abs(value + 1.0) <= 1e-9 for value in raw_scene_energy_values
    )
    unique_branch_energies = sorted({round(value, 15) for value in branch_energy_values})

    pass_criteria = {
        "artifact_only_analysis": True,
        "source_smoke_passed_all_criteria": bool(
            smoke_payload.get("passes_all_criteria")
        ),
        "seed17_only": int(smoke_payload["config"]["seed"]) == 17,
        "pilot_probe_subset_only": n_probe_cues <= 4,
        "planned_cell_count_is_nine": len(cell_results) == 9,
        "same_query_subset_all_cells": all(
            cell_summaries[cell_id]["probe_indices"] == expected_probe_indices
            for cell_id in cell_summaries
        ),
        "all_cells_returned_branches": total_branches > 0
        and all(
            probe["branch_count"] > 0
            for cell in cell_summaries.values()
            for probe in cell["probes"]
        ),
        "cue_conditioned_readout_non_saturated": len(unique_branch_energies) > 1
        and not all(abs(value + 1.0) <= 1e-9 for value in branch_energy_values),
        "raw_scene_energy_still_saturated": all_raw_scene_energies_saturated,
        "all_paired_delta_values_zero": bool(paired_delta_values)
        and all(abs(value) <= 1e-9 for value in paired_delta_values),
        "all_probe_minima_collapse_to_one_scene_set": all(
            row["all_cells_choose_same_min_scene"] for row in per_probe
        ),
        "all_minimum_selections_equal_probe_target_scene": all(
            row["all_min_scenes_equal_target"] for row in per_probe
        ),
        "no_n3_or_n10_claim": True,
        "no_full_matrix_claim": True,
        "not_headline_verification": True,
        "not_graduation": True,
    }

    diagnosis = {
        "condition_collapse_confirmed": all(
            [
                pass_criteria["all_paired_delta_values_zero"],
                pass_criteria["all_probe_minima_collapse_to_one_scene_set"],
                pass_criteria["all_minimum_selections_equal_probe_target_scene"],
            ]
        ),
        "collapse_mode": "min_branch_target_scene_attractor_collapse",
        "summary": (
            "The cue-conditioned readout is non-saturated, but the branch set "
            "for every condition contains the same cue-compatible target-scene "
            "minimum for each probe. The current min-within-condition summary "
            "therefore makes content, role, no-prior, no-schema-store, and "
            "random-schema cells indistinguishable at the paired Delta E level."
        ),
        "not_explained_by": [
            "raw cue-conditioned bridge-energy saturation",
            "missing paired Delta E values",
            "missing control-cell execution",
        ],
        "next_allowed_work": (
            "Precommit a stricter discriminator or query subset that cannot be "
            "solved by every condition selecting the same target-scene minimum; "
            "do not widen to n=3/n=10 before that precommit."
        ),
    }

    return {
        "framing": {
            "phase": "Phase 5 prime cue-conditioned condition-collapse analysis",
            "analysis_only": True,
            "retrieval_executed": False,
            "pilot_scope_only": True,
            "not_a_gate": True,
            "not_n3": True,
            "not_n10": True,
            "not_graduation": True,
            "no_full_matrix": True,
            "no_m1_escalation": True,
            "no_m2": True,
            "no_new_headline": True,
            "readout_id": smoke_payload["readout"]["readout_id"],
            "headline_metric": smoke_payload["readout"]["delta_e"],
        },
        "source_manifest": {
            "smoke_artifact_path": str(smoke_artifact_path)
            if smoke_artifact_path
            else None,
            "smoke_artifact_sha256": smoke_artifact_sha,
            "source_report": "Report 106",
            "source_readout_id": smoke_payload["readout"]["readout_id"],
        },
        "config": {
            "seed": int(smoke_payload["config"]["seed"]),
            "n_probe_cues": n_probe_cues,
            "planned_cell_count": len(cell_results),
            "total_branches": total_branches,
        },
        "aggregate": {
            "total_cells": len(cell_results),
            "total_probes_by_cell": len(min_selection_rows),
            "total_branches": total_branches,
            "branch_target_scene_hits": target_branch_hits,
            "branch_target_scene_rate": branch_target_scene_rate,
            "min_selection_target_hits": min_selection_target_hits,
            "min_selection_target_rate": min_selection_target_rate,
            "branch_scene_counts": branch_scene_counts,
            "unique_branch_energy_values": unique_branch_energies,
            "raw_scene_energy_values_unique": sorted(
                {round(value, 15) for value in raw_scene_energy_values}
            ),
            "paired_delta_values": paired_delta_values,
        },
        "per_probe_collapse": per_probe,
        "cell_summaries": cell_summaries,
        "pass_criteria": pass_criteria,
        "passes_all_criteria": all(bool(value) for value in pass_criteria.values()),
        "diagnosis": diagnosis,
        "anti_homunculus_check": {
            "pass": True,
            "reason": (
                "Passive artifact readback only; no metric-triggered route choice, "
                "condition selection, or retrieval rerun is introduced."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        default="reports/phase5_prime_cue_conditioned_bridge_smoke_seed17.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_condition_collapse_analysis.json",
    )
    args = parser.parse_args()

    smoke_path = Path(args.smoke)
    smoke_payload = _load_json(smoke_path)
    payload = analyze_condition_collapse(
        smoke_payload,
        smoke_artifact_path=smoke_path,
        smoke_artifact_sha=_sha256(smoke_path),
    )
    _write_json(Path(args.out), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
