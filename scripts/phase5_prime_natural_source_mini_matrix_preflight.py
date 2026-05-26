"""Preflight-only planner for the next Phase 5' natural-source mini-matrix.

This script does not run scene-MHN retrieval, content cleanup, or any top1 gate.
It freezes the cleaned Report 092/093 natural-source protocol into a broader
static candidate/control plan so a later gate can run without selecting cells
adaptively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Sequence

from energy_memory.phase5.natural_source_protocol import (
    protocol_payload,
    source_with_protocol_plan,
    validate_cleanup_preflight,
)


SOURCE_NAME = "trajectory_native_provenance_context_trace"
SOURCE_FAMILY = "trajectory_derived_native"
SOURCE_KIND = "repo_sample_phase2_window_trace"
DEFAULT_CUE_NOISE = [0.0, 0.05, 0.10, 0.15]
DEFAULT_CONDITIONS = [
    "candidate",
    "random_role",
    "deranged_role",
    "fixedpoint_free_shuffled_role",
    "content_cleanup_positive",
    "bundle_positive",
    "perfect_cue",
    "no_scene_token_baseline",
]
CONDITIONS = set(DEFAULT_CONDITIONS)
ROLE_NEGATIVE_EXACT_KEYS = {
    "random_role": "random_exact_rate",
    "deranged_role": "deranged_exact_rate",
    "fixedpoint_free_shuffled_role": "fixedpoint_free_shuffled_exact_rate",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _stats(values: Sequence[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _validate_source(source: dict) -> None:
    framing = source.get("framing", {})
    if framing.get("source_name") != SOURCE_NAME:
        raise ValueError(f"source artifact is not {SOURCE_NAME}")
    if framing.get("source_family") != SOURCE_FAMILY:
        raise ValueError(f"source artifact is not {SOURCE_FAMILY}")
    if framing.get("source_kind") != SOURCE_KIND:
        raise ValueError(f"source artifact is not {SOURCE_KIND}")
    config = source.get("config", {})
    required = {
        "D": 4096,
        "N": 512,
        "K_roles": 16,
        "context_roles": 4,
        "cooccurrence": "repo_sample_natural",
        "token_weight": 0.25,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(
                f"source hard-cell mismatch for {key}: "
                f"{config.get(key)!r} != {expected!r}"
            )


def _validate_conditions(conditions: Sequence[str]) -> List[str]:
    unknown = sorted(set(conditions) - CONDITIONS)
    if unknown:
        raise ValueError(f"unknown mini-matrix preflight conditions: {unknown}")
    if len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be unique")
    return list(conditions)


def _validate_cue_noise(values: Sequence[float]) -> List[float]:
    out = [float(value) for value in values]
    if any(value < 0.0 for value in out):
        raise ValueError("cue noise values must be non-negative")
    if len(set(out)) != len(out):
        raise ValueError("cue noise values must be unique")
    return out


def _seed_plan_summary(source: dict, protocol: dict) -> List[dict]:
    planned = source_with_protocol_plan(source, protocol)
    rows_by_seed = planned["source_rows_by_seed"]
    plan_by_seed = planned["query_plan_by_seed"]
    k_roles = int(planned["config"]["K_roles"])
    n_rows = int(planned["config"]["N"])
    n_queries = int(protocol["required_queries_per_seed"])
    out = []
    for seed in [int(seed) for seed in planned["config"]["seeds"]]:
        seed_key = str(seed)
        rows = rows_by_seed.get(seed_key)
        query_plan = plan_by_seed.get(seed_key)
        if not isinstance(rows, list) or len(rows) != n_rows:
            raise ValueError(f"source rows for seed {seed} are missing or not N")
        if not isinstance(query_plan, list) or len(query_plan) != n_queries:
            raise ValueError(f"query plan for seed {seed} is missing or not n_queries")
        target_atoms = [int(item["target_atom"]) for item in query_plan]
        target_frequencies = [int(item["target_frequency"]) for item in query_plan]
        observed_counts = [len(item["observed_roles"]) for item in query_plan]
        for row in rows:
            if not isinstance(row, list) or len(row) != k_roles:
                raise ValueError(f"source row for seed {seed} is not K_roles long")
        for item in query_plan:
            scene = int(item["scene"])
            known_role = int(item["known_role"])
            query_role = int(item["query_role"])
            observed_roles = [int(role) for role in item["observed_roles"]]
            if scene < 0 or scene >= n_rows:
                raise ValueError(f"query plan for seed {seed} has invalid scene")
            if not (0 <= known_role < k_roles and 0 <= query_role < k_roles):
                raise ValueError(f"query plan for seed {seed} has invalid role")
            if known_role not in observed_roles:
                raise ValueError(f"query plan for seed {seed} omits known_role")
            if query_role in observed_roles:
                raise ValueError(f"query plan for seed {seed} leaks query_role")
        out.append({
            "seed": seed,
            "rows": len(rows),
            "selected_queries": len(query_plan),
            "distinct_target_atoms": len(set(target_atoms)),
            "target_frequency_mean": _stats([float(v) for v in target_frequencies])["mean"],
            "observed_roles_per_query": _stats([float(v) for v in observed_counts]),
            "query_plan_sha256": _stable_sha256(query_plan),
        })
    return out


def _opportunity_summary(protocol: dict) -> dict:
    per_seed = protocol["per_seed"]
    summary = {}
    for condition, exact_key in ROLE_NEGATIVE_EXACT_KEYS.items():
        values = [
            float(row["same_scene_exact_opportunity"][exact_key])
            for row in per_seed
        ]
        summary[condition] = {
            "same_scene_exact_rate": _stats(values),
            "passes_zero_exact_opportunity": all(value == 0.0 for value in values),
        }
    return summary


def _cell_for_condition(
    *,
    condition: str,
    cue_noise: float,
    config: dict,
    protocol_name: str,
    query_plan_sha256_by_seed: Dict[str, str],
) -> dict:
    scene_token = condition != "no_scene_token_baseline"
    token_weight = float(config["token_weight"]) if scene_token else 0.0
    token_source = SOURCE_NAME if scene_token else "none"
    run_condition = "candidate" if condition == "no_scene_token_baseline" else condition
    cell_id = (
        f"{condition}|protocol={protocol_name}|D={config['D']}|"
        f"K={config['K_roles']}|N={config['N']}|noise={cue_noise}|"
        f"scene_token={int(scene_token)}|token_weight={token_weight}|"
        f"token_source={token_source}|context_roles={config['context_roles']}|"
        f"cooc={config['cooccurrence']}"
    )
    exact_key = ROLE_NEGATIVE_EXACT_KEYS.get(condition)
    return {
        "cell_id": cell_id,
        "cell_kind": condition,
        "gate_condition": run_condition,
        "cue_noise": cue_noise,
        "scene_token": scene_token,
        "scene_token_weight": token_weight,
        "scene_token_source": token_source,
        "protocol_name": protocol_name,
        "conditions_are_static": True,
        "retrieval_run": False,
        "query_plan_sha256_by_seed": query_plan_sha256_by_seed,
        "same_scene_exact_opportunity_key": exact_key,
    }


def build_payload(
    *,
    source: dict,
    cleanup_preflight: dict,
    source_path: Path,
    cleanup_path: Path,
    prior_gate_path: Path,
    source_sha: str,
    cleanup_sha: str,
    prior_gate_sha: str,
    protocol_name: str | None,
    cue_noise: Sequence[float],
    conditions: Sequence[str],
) -> dict:
    _validate_source(source)
    validate_cleanup_preflight(
        cleanup_preflight,
        source_sha=source_sha,
        gate_sha=prior_gate_sha,
    )
    protocol = protocol_payload(cleanup_preflight, protocol_name)
    config = cleanup_preflight["config"]["source_config"]
    if config != source.get("config", {}):
        raise ValueError("cleanup preflight source_config does not match source config")
    noises = _validate_cue_noise(cue_noise)
    planned_conditions = _validate_conditions(conditions)
    seed_summary = _seed_plan_summary(source, protocol)
    query_plan_sha_by_seed = {
        str(row["seed"]): str(row["query_plan_sha256"])
        for row in seed_summary
    }
    planned_cells = [
        _cell_for_condition(
            condition=condition,
            cue_noise=noise,
            config=config,
            protocol_name=protocol["protocol_name"],
            query_plan_sha256_by_seed=query_plan_sha_by_seed,
        )
        for noise in noises
        for condition in planned_conditions
    ]
    opportunity_summary = _opportunity_summary(protocol)
    pass_criteria = {
        "preflight_only_no_retrieval": True,
        "cleanup_protocol_passes": bool(protocol["passes_all_criteria"]),
        "source_matches_fixed_hard_cell": True,
        "cue_noise_sweep_includes_0p05": 0.05 in noises,
        "planned_conditions_are_static": all(
            cell["conditions_are_static"] and not cell["retrieval_run"]
            for cell in planned_cells
        ),
        "role_negative_exact_opportunities_zero": all(
            item["passes_zero_exact_opportunity"]
            for item in opportunity_summary.values()
        ),
        "all_cells_share_seed_query_plans": len({
            _stable_sha256(cell["query_plan_sha256_by_seed"])
            for cell in planned_cells
        }) == 1,
    }
    return {
        "framing": {
            "phase": "5-prime natural-source mini-matrix preflight",
            "source_name": SOURCE_NAME,
            "source_family": SOURCE_FAMILY,
            "source_kind": SOURCE_KIND,
            "preflight_only": True,
            "analysis_only": True,
            "not_graduation": True,
            "no_delta_e_headline": True,
            "no_candidate_control_retrieval": True,
            "no_m2": True,
            "no_full_matrix": True,
            "anti_homunculus": (
                "fixed source rows, fixed cleaned query plan, fixed cue-noise "
                "sweep, fixed controls; no metric-triggered routing or "
                "best-of-N selection"
            ),
        },
        "source_manifest": {
            "source_artifact_path": str(source_path),
            "source_artifact_sha256": source_sha,
            "cleanup_preflight_artifact_path": str(cleanup_path),
            "cleanup_preflight_artifact_sha256": cleanup_sha,
            "prior_gate_artifact_path": str(prior_gate_path),
            "prior_gate_artifact_sha256": prior_gate_sha,
        },
        "config": {
            "D": int(config["D"]),
            "N": int(config["N"]),
            "K_roles": int(config["K_roles"]),
            "C_codebook": int(config["C_codebook"]),
            "context_roles": int(config["context_roles"]),
            "cooccurrence": str(config["cooccurrence"]),
            "corpus_source": str(config["corpus_source"]),
            "n_queries": int(protocol["required_queries_per_seed"]),
            "seeds": [int(seed) for seed in config["seeds"]],
            "cue_noise_sweep": noises,
            "scene_token_weight": float(config["token_weight"]),
            "conditions": planned_conditions,
        },
        "cleanup_protocol_summary": {
            "protocol_name": protocol["protocol_name"],
            "pass_criteria": protocol["pass_criteria"],
            "aggregate": protocol["aggregate"],
            "selected_query_plan_sha256": _stable_sha256(
                protocol["selected_query_plan_by_seed"]
            ),
        },
        "support_summary": {
            "per_seed": seed_summary,
            "selected_queries": _stats([
                float(row["selected_queries"]) for row in seed_summary
            ]),
            "distinct_target_atoms": _stats([
                float(row["distinct_target_atoms"]) for row in seed_summary
            ]),
            "target_frequency_mean": _stats([
                float(row["target_frequency_mean"]) for row in seed_summary
            ]),
        },
        "control_opportunity_summary": opportunity_summary,
        "planned_cells": planned_cells,
        "selected_query_plan_by_seed": protocol["selected_query_plan_by_seed"],
        "pass_criteria": pass_criteria,
        "passes_all_criteria": all(pass_criteria.values()),
        "decision_read": (
            "Preflight only. A passing artifact permits a separate fixed "
            "candidate/control mini-matrix gate precommit. It does not authorize "
            "a full matrix, M2, Phase 5 Delta E headline run, or graduation claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="reports/phase5_prime_nonsynthetic_native_context_source.json",
    )
    parser.add_argument(
        "--cleanup-preflight",
        default="reports/phase5_prime_natural_source_control_cleanup_preflight.json",
    )
    parser.add_argument(
        "--prior-gate",
        default="reports/phase5_prime_nonsynthetic_native_gate.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_natural_source_mini_matrix_preflight.json",
    )
    parser.add_argument("--protocol", default=None)
    parser.add_argument(
        "--cue-noise",
        nargs="+",
        type=float,
        default=DEFAULT_CUE_NOISE,
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    cleanup_path = Path(args.cleanup_preflight)
    prior_gate_path = Path(args.prior_gate)
    source = _load_json(source_path)
    cleanup_preflight = _load_json(cleanup_path)
    source_sha = _sha256(source_path)
    cleanup_sha = _sha256(cleanup_path)
    prior_gate_sha = _sha256(prior_gate_path)
    payload = build_payload(
        source=source,
        cleanup_preflight=cleanup_preflight,
        source_path=source_path,
        cleanup_path=cleanup_path,
        prior_gate_path=prior_gate_path,
        source_sha=source_sha,
        cleanup_sha=cleanup_sha,
        prior_gate_sha=prior_gate_sha,
        protocol_name=args.protocol,
        cue_noise=args.cue_noise,
        conditions=args.conditions,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out_path),
        "passes_all_criteria": payload["passes_all_criteria"],
        "protocol": payload["cleanup_protocol_summary"]["protocol_name"],
        "n_planned_cells": len(payload["planned_cells"]),
        "cue_noise_sweep": payload["config"]["cue_noise_sweep"],
        "conditions": payload["config"]["conditions"],
        "preflight_only": payload["framing"]["preflight_only"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
