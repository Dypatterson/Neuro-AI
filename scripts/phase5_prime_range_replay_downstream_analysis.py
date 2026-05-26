"""Analyze the fixed range-shaped replay downstream lane.

The input is the Experiment 47 JSON emitted by the Report 110 precommit plan.
This script performs seed-paired comparisons against ``standard`` and writes a
bounded viability decision without running any replay or retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


BASELINE = "standard"
RANGE_POSTSETTLE = "range_postsettle"
RANGE_PRESETTLE = "range_presettle"
SUPPORT_METRICS = ("candidates", "provenance_cells", "provenance_rect")
DOWNSTREAM_METRICS = ("heldout_top1", "heldout_topk", "heldout_cap_t05")
GEOMETRY_METRICS = (
    "stored_near_duplicate_rate",
    "final_near_duplicate_rate",
    "query_near_existing_rate",
    "d_eff_final",
)
TOP1_MIN_DELTA = 0.02
TOPK_MIN_DELTA = 0.05
CAP_T05_MIN_DELTA = 0.02
NEAR_DUPLICATE_MAX_FOR_NOVELTY = 0.25
PRESETTLE_STRONG_NOVELTY_MAX = 0.10
PRESETTLE_DEFF_MIN_DELTA = 5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def bootstrap_ci(
    values: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 110,
) -> dict:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    if len(vals) == 1 or samples <= 0:
        value = vals[0] if len(vals) == 1 else _mean(vals)
        return {"mean": value, "ci_low": value, "ci_high": value, "n": len(vals)}
    rng = random.Random(seed)
    boot = []
    n = len(vals)
    for _ in range(samples):
        boot.append(_mean([vals[rng.randrange(n)] for _ in range(n)]))
    boot.sort()
    low_i = int(0.025 * (len(boot) - 1))
    high_i = int(0.975 * (len(boot) - 1))
    return {
        "mean": _mean(vals),
        "ci_low": float(boot[low_i]),
        "ci_high": float(boot[high_i]),
        "n": n,
    }


def _row_index(rows: Iterable[dict]) -> Dict[Tuple[str, int, int], dict]:
    out: Dict[Tuple[str, int, int], dict] = {}
    for row in rows:
        key = (str(row["condition"]), int(row["scale"]), int(row["seed"]))
        if key in out:
            raise ValueError(f"duplicate row for {key}")
        out[key] = row
    return out


def _available_scales(rows: Sequence[dict]) -> List[int]:
    return sorted({int(row["scale"]) for row in rows})


def _available_seeds(rows: Sequence[dict]) -> List[int]:
    return sorted({int(row["seed"]) for row in rows})


def _condition_mean(
    index: Dict[Tuple[str, int, int], dict],
    *,
    condition: str,
    scale: int,
    seeds: Sequence[int],
    metric: str,
) -> float:
    values = [
        float(index[(condition, scale, seed)][metric])
        for seed in seeds
        if (condition, scale, seed) in index
    ]
    return _mean(values)


def paired_delta(
    index: Dict[Tuple[str, int, int], dict],
    *,
    condition: str,
    scale: int,
    metric: str,
    seeds: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    deltas = []
    used = []
    for seed in seeds:
        base_key = (BASELINE, scale, seed)
        cond_key = (condition, scale, seed)
        if base_key in index and cond_key in index:
            deltas.append(float(index[cond_key][metric]) - float(index[base_key][metric]))
            used.append(seed)
    ci = bootstrap_ci(
        deltas,
        samples=bootstrap_samples,
        seed=bootstrap_seed + scale * 31 + sum(ord(c) for c in metric + condition),
    )
    ci["seeds"] = used
    return ci


def _support_moves(deltas: dict) -> bool:
    return (
        deltas["candidates"]["mean"] > 0.0
        and deltas["provenance_cells"]["mean"] > 0.0
        and deltas["provenance_rect"]["mean"] < 0.0
    )


def _downstream_moves(deltas: dict) -> bool:
    top1 = deltas["heldout_top1"]
    topk = deltas["heldout_topk"]
    cap = deltas["heldout_cap_t05"]
    return (
        top1["mean"] >= TOP1_MIN_DELTA
        and top1["ci_low"] >= 0.0
    ) or (
        topk["mean"] >= TOPK_MIN_DELTA
        and topk["ci_low"] >= 0.0
    ) or (
        cap["mean"] >= CAP_T05_MIN_DELTA
        and cap["ci_low"] >= 0.0
    )


def compare_condition(
    rows: Sequence[dict],
    *,
    condition: str,
    scale: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    index = _row_index(rows)
    seeds = _available_seeds(rows)
    metrics = list(SUPPORT_METRICS + GEOMETRY_METRICS + DOWNSTREAM_METRICS)
    deltas = {
        metric: paired_delta(
            index,
            condition=condition,
            scale=scale,
            metric=metric,
            seeds=seeds,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for metric in metrics
    }
    means = {
        BASELINE: {
            metric: _condition_mean(
                index,
                condition=BASELINE,
                scale=scale,
                seeds=seeds,
                metric=metric,
            )
            for metric in metrics
        },
        condition: {
            metric: _condition_mean(
                index,
                condition=condition,
                scale=scale,
                seeds=seeds,
                metric=metric,
            )
            for metric in metrics
        },
    }
    support_moves = _support_moves(deltas)
    downstream_moves = _downstream_moves(deltas)
    novelty_moves = (
        means[condition]["stored_near_duplicate_rate"]
        <= NEAR_DUPLICATE_MAX_FOR_NOVELTY
    )
    deff_nonnegative = deltas["d_eff_final"]["mean"] >= 0.0
    decision_scale = scale in {3, 4}
    presettle_strong_novelty = (
        condition == RANGE_PRESETTLE
        and decision_scale
        and means[condition]["stored_near_duplicate_rate"]
        <= PRESETTLE_STRONG_NOVELTY_MAX
        and deltas["d_eff_final"]["mean"] >= PRESETTLE_DEFF_MIN_DELTA
    )
    current_path_viable = (
        condition == RANGE_POSTSETTLE
        and decision_scale
        and support_moves
        and novelty_moves
        and deff_nonnegative
        and downstream_moves
    )
    return {
        "condition": condition,
        "scale": scale,
        "means": means,
        "seed_paired_deltas": deltas,
        "criteria": {
            "support_moves": support_moves,
            "novelty_moves": novelty_moves,
            "d_eff_nonnegative": deff_nonnegative,
            "downstream_moves": downstream_moves,
            "current_path_viable": current_path_viable,
            "presettle_strong_novelty": presettle_strong_novelty,
            "decision_scale": decision_scale,
        },
    }


def analyze_payload(
    *,
    results: dict,
    precommit: dict,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 110,
) -> dict:
    rows = results.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("results JSON has no rows")
    conditions = sorted({str(row["condition"]) for row in rows})
    required = {BASELINE, RANGE_POSTSETTLE, RANGE_PRESETTLE}
    missing = sorted(required - set(conditions))
    if missing:
        raise ValueError(f"missing required conditions: {missing}")
    scales = _available_scales(rows)
    comparisons: Dict[str, Dict[str, dict]] = {
        RANGE_POSTSETTLE: {},
        RANGE_PRESETTLE: {},
    }
    for condition in [RANGE_POSTSETTLE, RANGE_PRESETTLE]:
        for scale in scales:
            comparisons[condition][f"W{scale}"] = compare_condition(
                rows,
                condition=condition,
                scale=scale,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )

    current_path_pass_scales = [
        key
        for key, comparison in comparisons[RANGE_POSTSETTLE].items()
        if comparison["criteria"]["current_path_viable"]
    ]
    presettle_downstream_scales = [
        key
        for key, comparison in comparisons[RANGE_PRESETTLE].items()
        if comparison["criteria"]["presettle_strong_novelty"]
        and comparison["criteria"]["downstream_moves"]
    ]
    presettle_novelty_scales = [
        key
        for key, comparison in comparisons[RANGE_PRESETTLE].items()
        if comparison["criteria"]["presettle_strong_novelty"]
    ]

    if current_path_pass_scales:
        decision = "range_postsettle_viable_for_fresh_phase5_precommit"
        bounded_viability = "viable_for_followup_not_graduation"
    elif presettle_downstream_scales:
        decision = "positive_control_downstream_movement_requires_fresh_mechanism_precommit"
        bounded_viability = "positive_control_only_not_current_path"
    elif presettle_novelty_scales:
        decision = "not_viable_current_range_replay_downstream_novelty_without_retrieval"
        bounded_viability = "not_viable_current_lane"
    else:
        decision = "not_viable_current_range_replay_downstream_no_useful_novelty"
        bounded_viability = "not_viable_current_lane"

    return {
        "analysis_id": "range_shaped_replay_downstream_analysis_v1",
        "lane_id": precommit.get("lane_id", "unknown"),
        "report_id": 111,
        "precommit_report_id": precommit.get("report_id"),
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "unit": "seed-paired condition deltas",
        },
        "results_config": results.get("config", {}),
        "conditions": conditions,
        "scales": scales,
        "comparisons": comparisons,
        "decision": {
            "id": decision,
            "bounded_viability": bounded_viability,
            "current_path_pass_scales": current_path_pass_scales,
            "presettle_downstream_scales": presettle_downstream_scales,
            "presettle_novelty_scales": presettle_novelty_scales,
            "phase5_delta_e_run_authorized": False,
            "graduation_claim_authorized": False,
            "bridge_path_reopened": False,
        },
        "interpretation_bounds": [
            (
                "range_postsettle is the current production-compatible "
                "range-shaped replay path"
            ),
            (
                "range_presettle is a diagnostic positive control and cannot "
                "be treated as a production mechanism without a fresh precommit"
            ),
            (
                "candidate/provenance support is insufficient unless useful "
                "novelty and held-out retrieval also move"
            ),
        ],
        "analysis_complete": True,
    }


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=110)
    args = parser.parse_args()

    precommit_path = Path(args.precommit)
    results_path = Path(args.results)
    precommit = load_json(precommit_path)
    results = load_json(results_path)
    payload = analyze_payload(
        results=results,
        precommit=precommit,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    payload["input_artifacts"] = {
        "precommit": str(precommit_path),
        "precommit_sha256": sha256_file(precommit_path),
        "results": str(results_path),
        "results_sha256": sha256_file(results_path),
    }
    digest = write_json(Path(args.out), payload)
    print(f"wrote {args.out}")
    print(f"sha256 {digest}")
    print(f"decision {payload['decision']['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
