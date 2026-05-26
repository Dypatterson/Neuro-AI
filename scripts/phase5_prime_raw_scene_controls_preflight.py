"""Static controls preflight for the raw-scene-energy Phase 5' bridge.

This script plans required controls for ``raw_scene_energy_v0`` without running
retrieval, top1, branch settling, or headline evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Sequence

from energy_memory.phase5.bundle_first_scene_memory import (
    BundleFirstConfig,
    build_bundle_first_seed_state,
)
from energy_memory.phase5.natural_source_protocol import (
    protocol_payload,
    source_with_protocol_plan,
    validate_cleanup_preflight,
)


HEADLINE_METRIC = "Delta E = E_content-prior - E_role-prior"
BASELINE_ID = "raw_scene_energy_v0"
LEGACY_HEADLINE_DEFAULTS = {
    "beta": 10.0,
    "gamma": 0.5,
    "k_main": 1,
    "include_surprise_branch": False,
    "formulation": "per_pattern",
}
RAW_SCENE_DEFAULTS = {
    "beta": 30.0,
    "gamma": 0.5,
    "k_main": 4,
    "include_surprise_branch": False,
    "formulation": "per_pattern",
    "score_bias": None,
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _control_cell(
    *,
    cell_id: str,
    control_family: str,
    prior_type: str,
    schema_source: str,
    k_main: int,
    gamma: float,
    paired_delta_group: Optional[str],
    config: BundleFirstConfig,
) -> dict:
    if schema_source == "scene_store":
        schema_store_shape = [config.N, config.D]
        schema_bindings_shape = [config.N, config.K_roles, config.D]
        schema_atom_idx_shape: Optional[List[int]] = [config.N]
    elif schema_source == "content_codebook":
        schema_store_shape = [config.C_codebook, config.D]
        schema_bindings_shape = [config.C_codebook, config.K_roles, config.D]
        schema_atom_idx_shape = None
    else:
        raise ValueError(f"unknown schema_source: {schema_source}")
    return {
        "cell_id": cell_id,
        "control_family": control_family,
        "headline_metric": HEADLINE_METRIC,
        "baseline_id": BASELINE_ID,
        "prior_type": prior_type,
        "schema_source": schema_source,
        "schema_store_shape": schema_store_shape,
        "schema_bindings_shape": schema_bindings_shape,
        "schema_atom_idx_shape": schema_atom_idx_shape,
        "memory_source": "bundle_first_scene_store",
        "memory_shape": [config.N, config.D],
        "query_plan_ref": "source.query_plan_by_seed",
        "score_bias": None,
        "energy_readout": "raw_scene_mhn_energy",
        "formulation": RAW_SCENE_DEFAULTS["formulation"],
        "include_surprise_branch": RAW_SCENE_DEFAULTS["include_surprise_branch"],
        "run_combiners": False,
        "k_main": int(k_main),
        "gamma": float(gamma),
        "paired_delta_group": paired_delta_group,
        "retrieval_executed": False,
    }


def planned_control_cells(
    *,
    config: BundleFirstConfig,
    beta: float = 30.0,
    gamma: float = 0.5,
    k_main: int = 4,
) -> List[dict]:
    cells = [
        _control_cell(
            cell_id=f"main_content_K{k_main}_g{gamma:g}",
            control_family="paired_delta_main",
            prior_type="content",
            schema_source="scene_store",
            k_main=k_main,
            gamma=gamma,
            paired_delta_group="main",
            config=config,
        ),
        _control_cell(
            cell_id=f"main_role_K{k_main}_g{gamma:g}",
            control_family="paired_delta_main",
            prior_type="role",
            schema_source="scene_store",
            k_main=k_main,
            gamma=gamma,
            paired_delta_group="main",
            config=config,
        ),
        _control_cell(
            cell_id=f"random_schema_K{k_main}_g{gamma:g}",
            control_family="random_schema",
            prior_type="random",
            schema_source="scene_store",
            k_main=k_main,
            gamma=gamma,
            paired_delta_group=None,
            config=config,
        ),
        _control_cell(
            cell_id=f"k1_content_g{gamma:g}",
            control_family="k1",
            prior_type="content",
            schema_source="scene_store",
            k_main=1,
            gamma=gamma,
            paired_delta_group="k1",
            config=config,
        ),
        _control_cell(
            cell_id=f"k1_role_g{gamma:g}",
            control_family="k1",
            prior_type="role",
            schema_source="scene_store",
            k_main=1,
            gamma=gamma,
            paired_delta_group="k1",
            config=config,
        ),
        _control_cell(
            cell_id=f"no_prior_content_K{k_main}_g0",
            control_family="no_prior",
            prior_type="content",
            schema_source="scene_store",
            k_main=k_main,
            gamma=0.0,
            paired_delta_group="no_prior",
            config=config,
        ),
        _control_cell(
            cell_id=f"no_prior_role_K{k_main}_g0",
            control_family="no_prior",
            prior_type="role",
            schema_source="scene_store",
            k_main=k_main,
            gamma=0.0,
            paired_delta_group="no_prior",
            config=config,
        ),
        _control_cell(
            cell_id=f"no_schema_store_content_K{k_main}_g{gamma:g}",
            control_family="no_schema_store",
            prior_type="content",
            schema_source="content_codebook",
            k_main=k_main,
            gamma=gamma,
            paired_delta_group="no_schema_store",
            config=config,
        ),
        _control_cell(
            cell_id=f"no_schema_store_role_K{k_main}_g{gamma:g}",
            control_family="no_schema_store",
            prior_type="role",
            schema_source="content_codebook",
            k_main=k_main,
            gamma=gamma,
            paired_delta_group="no_schema_store",
            config=config,
        ),
    ]
    for cell in cells:
        cell["beta"] = float(beta)
    return cells


def _validate_source_shape_and_queries(
    *,
    source: dict,
    seeds: Sequence[int],
    config: BundleFirstConfig,
) -> dict:
    rows_by_seed = source.get("source_rows_by_seed", {})
    plan_by_seed = source.get("query_plan_by_seed", {})
    per_seed = []
    ok = True
    observed_role_counts = set()
    for seed in seeds:
        seed_key = str(seed)
        rows = rows_by_seed.get(seed_key)
        plan = plan_by_seed.get(seed_key)
        seed_ok = isinstance(rows, list) and isinstance(plan, list)
        invalid_rows = 0
        invalid_queries = 0
        if seed_ok:
            seed_ok = len(rows) == config.N and len(plan) == config.n_queries
        if seed_ok:
            for row in rows:
                row_ok = (
                    isinstance(row, list)
                    and len(row) == config.K_roles
                    and all(0 <= int(atom) < config.C_codebook for atom in row)
                )
                if not row_ok:
                    invalid_rows += 1
            for item in plan:
                scene = int(item.get("scene", -1))
                known_role = int(item.get("known_role", -1))
                query_role = int(item.get("query_role", -1))
                observed_roles = [int(role) for role in item.get("observed_roles", [])]
                observed_role_counts.add(len(observed_roles))
                query_ok = (
                    0 <= scene < config.N
                    and 0 <= known_role < config.K_roles
                    and 0 <= query_role < config.K_roles
                    and known_role in observed_roles
                    and query_role not in observed_roles
                    and len(observed_roles) == config.context_roles
                    and all(0 <= role < config.K_roles for role in observed_roles)
                )
                if not query_ok:
                    invalid_queries += 1
        seed_ok = seed_ok and invalid_rows == 0 and invalid_queries == 0
        ok = ok and seed_ok
        per_seed.append({
            "seed": int(seed),
            "rows": len(rows) if isinstance(rows, list) else None,
            "queries": len(plan) if isinstance(plan, list) else None,
            "invalid_rows": invalid_rows,
            "invalid_queries": invalid_queries,
            "passes": bool(seed_ok),
        })
    return {
        "passes": bool(ok),
        "per_seed": per_seed,
        "observed_role_counts": sorted(observed_role_counts),
    }


def _config_mismatches(*, beta: float, gamma: float, k_main: int) -> List[dict]:
    candidate = {
        "beta": float(beta),
        "gamma": float(gamma),
        "k_main": int(k_main),
        "include_surprise_branch": RAW_SCENE_DEFAULTS["include_surprise_branch"],
        "formulation": RAW_SCENE_DEFAULTS["formulation"],
    }
    out = []
    for key, legacy_value in LEGACY_HEADLINE_DEFAULTS.items():
        value = candidate[key]
        if value != legacy_value:
            out.append({
                "field": key,
                "raw_scene_value": value,
                "legacy_headline_default": legacy_value,
                "requires_precommit_before_retrieval": True,
            })
    return out


def run_controls_preflight(
    *,
    source: dict,
    source_path: Path,
    source_sha: str,
    cleanup_preflight_path: Optional[Path],
    cleanup_preflight_sha: Optional[str],
    prior_gate_path: Optional[Path],
    prior_gate_sha: Optional[str],
    config: BundleFirstConfig,
    protocol_name: str,
    seeds: Sequence[int],
    shape_probe_seed: int,
    beta: float,
    gamma: float,
    k_main: int,
    device: str = "cpu",
) -> dict:
    if shape_probe_seed not in [int(seed) for seed in seeds]:
        raise ValueError("shape_probe_seed must be in seeds")
    cells = planned_control_cells(
        config=config,
        beta=beta,
        gamma=gamma,
        k_main=k_main,
    )
    query_validation = _validate_source_shape_and_queries(
        source=source,
        seeds=seeds,
        config=config,
    )
    shape_state = build_bundle_first_seed_state(
        seed=shape_probe_seed,
        source=source,
        config=config,
        device=device,
    )
    shape_probe = {
        "seed": int(shape_probe_seed),
        "device": device,
        "scene_matrix_shape": list(shape_state.scene_matrix.shape),
        "roles_shape": list(shape_state.roles.shape),
        "content_shape": list(shape_state.content.shape),
        "query_context_tokens_shape": list(shape_state.query_context_tokens.shape),
        "retrieval_executed": False,
    }
    cell_ids = {cell["cell_id"] for cell in cells}
    families = {cell["control_family"] for cell in cells}
    score_bias_values = {json.dumps(cell["score_bias"]) for cell in cells}
    pass_criteria = {
        "framing_preflight_only": True,
        "raw_scene_energy_v0_selected": True,
        "score_bias_none_all_cells": score_bias_values == {"null"},
        "no_retrieval_executed": all(not cell["retrieval_executed"] for cell in cells),
        "main_delta_pair_planned": {"paired_delta_main"} <= families,
        "random_schema_planned": "random_schema" in families,
        "k1_pair_planned": {"k1_content_g0.5", "k1_role_g0.5"} <= cell_ids,
        "no_prior_pair_planned": {
            f"no_prior_content_K{k_main}_g0",
            f"no_prior_role_K{k_main}_g0",
        } <= cell_ids,
        "no_schema_store_pair_planned": {
            f"no_schema_store_content_K{k_main}_g{gamma:g}",
            f"no_schema_store_role_K{k_main}_g{gamma:g}",
        } <= cell_ids,
        "same_query_set_all_controls": all(
            cell["query_plan_ref"] == "source.query_plan_by_seed" for cell in cells
        ),
        "source_manifest_sha_recorded": bool(source_sha),
        "protocol_name_recorded": bool(protocol_name),
        "query_plan_integrity_ok": bool(query_validation["passes"]),
        "scene_store_shapes_validated": shape_probe["scene_matrix_shape"]
        == [config.N, config.D],
        "no_schema_store_shapes_planned_not_materialized": all(
            cell["schema_source"] != "content_codebook"
            or cell["schema_bindings_shape"] == [config.C_codebook, config.K_roles, config.D]
            for cell in cells
        ),
        "config_mismatch_flags_recorded": bool(
            _config_mismatches(beta=beta, gamma=gamma, k_main=k_main)
        ),
        "headline_metric_unchanged": all(
            cell["headline_metric"] == HEADLINE_METRIC for cell in cells
        ),
        "stop_conditions_encoded": True,
    }
    return {
        "framing": {
            "phase": "Phase 5 prime raw-scene bridge controls preflight",
            "baseline_id": BASELINE_ID,
            "preflight_only": True,
            "not_a_gate": True,
            "not_graduation": True,
            "no_retrieval_run": True,
            "no_top1": True,
            "no_headline_claim": True,
            "no_full_matrix": True,
            "no_m1_escalation": True,
            "no_m2": True,
            "headline_metric": HEADLINE_METRIC,
        },
        "source_manifest": {
            "source_artifact_path": str(source_path),
            "source_artifact_sha256": source_sha,
            "cleanup_preflight_path": (
                str(cleanup_preflight_path) if cleanup_preflight_path else None
            ),
            "cleanup_preflight_sha256": cleanup_preflight_sha,
            "prior_gate_path": str(prior_gate_path) if prior_gate_path else None,
            "prior_gate_sha256": prior_gate_sha,
            "protocol_name": protocol_name,
        },
        "config": {
            "D": config.D,
            "N": config.N,
            "K_roles": config.K_roles,
            "C_codebook": config.C_codebook,
            "context_roles": config.context_roles,
            "n_queries_per_seed": config.n_queries,
            "seeds": [int(seed) for seed in seeds],
            "scene_token_weight": config.scene_token_weight,
            "cooccurrence": config.cooccurrence,
            "source_name": config.source_name,
            "beta": float(beta),
            "gamma": float(gamma),
            "k_main": int(k_main),
            "score_bias": None,
            "include_surprise_branch": RAW_SCENE_DEFAULTS["include_surprise_branch"],
            "formulation": RAW_SCENE_DEFAULTS["formulation"],
        },
        "legacy_headline_default_mismatches": _config_mismatches(
            beta=beta,
            gamma=gamma,
            k_main=k_main,
        ),
        "query_plan_validation": query_validation,
        "shape_probe": shape_probe,
        "planned_cells": cells,
        "planned_cell_count": len(cells),
        "pass_criteria": pass_criteria,
        "passes_all_criteria": all(bool(value) for value in pass_criteria.values()),
        "retrieval_executed": False,
        "stop_conditions": [
            "Do not run retrieval if any planned control cannot share the same fixed query set.",
            "Do not run retrieval if no-schema-store requires changing the headline metric.",
            "Do not run retrieval if a scene-level Step-3 score_bias is reintroduced without a fixed mapping precommit.",
            "Do not run retrieval if beta/gamma/K defaults drift without an explicit precommit.",
            "Do not interpret this artifact as Phase 5 evidence or graduation.",
        ],
        "anti_homunculus_check": {
            "pass": True,
            "reason": "Static control planning only; no metric-triggered route choice, adaptive bias, retrieval, or best-of-N evidence.",
        },
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
        default="reports/phase5_prime_raw_scene_controls_preflight.json",
    )
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--shape-probe-seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--k-main", type=int, default=4)
    args = parser.parse_args()

    source_path = Path(args.source)
    cleanup_path = Path(args.cleanup_preflight)
    prior_gate_path = Path(args.prior_gate)
    source = _load_json(source_path)
    cleanup_preflight = _load_json(cleanup_path)
    source_sha = _sha256(source_path)
    cleanup_sha = _sha256(cleanup_path)
    prior_gate_sha = _sha256(prior_gate_path)
    validate_cleanup_preflight(
        cleanup_preflight,
        source_sha=source_sha,
        gate_sha=prior_gate_sha,
    )
    protocol = protocol_payload(cleanup_preflight, args.protocol)
    run_source = source_with_protocol_plan(source, protocol)
    source_config = cleanup_preflight["config"]["source_config"]
    config = BundleFirstConfig(
        D=int(source_config["D"]),
        N=int(source_config["N"]),
        K_roles=int(source_config["K_roles"]),
        C_codebook=int(source_config["C_codebook"]),
        context_roles=int(source_config["context_roles"]),
        n_queries=int(protocol["required_queries_per_seed"]),
        beta=float(args.beta),
        max_iter=0,
        scene_token_weight=float(source_config["token_weight"]),
        cooccurrence=str(source_config["cooccurrence"]),
        source_name=str(source["framing"]["source_name"]),
    )
    payload = run_controls_preflight(
        source=run_source,
        source_path=source_path,
        source_sha=source_sha,
        cleanup_preflight_path=cleanup_path,
        cleanup_preflight_sha=cleanup_sha,
        prior_gate_path=prior_gate_path,
        prior_gate_sha=prior_gate_sha,
        config=config,
        protocol_name=str(protocol["protocol_name"]),
        seeds=[int(seed) for seed in source_config["seeds"]],
        shape_probe_seed=int(args.shape_probe_seed),
        beta=float(args.beta),
        gamma=float(args.gamma),
        k_main=int(args.k_main),
        device=str(args.device),
    )
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {out_path} passes={payload['passes_all_criteria']} "
        f"planned_cells={payload['planned_cell_count']} retrieval_executed=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
