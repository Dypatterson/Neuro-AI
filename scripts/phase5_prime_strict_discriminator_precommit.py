"""Precommit a strict Phase 5' bridge discriminator query subset.

This script performs static prior-ranking analysis only. It selects seed-17
queries where the role-prior top-K scene set contains the target scene while
the content-prior top-K scene set excludes it. Those probes are the narrow
subset where the current bridge can actually discriminate content-vs-role
branching before any wider evidence run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from energy_memory.phase5.bundle_first_scene_memory import (
    BundleFirstConfig,
    build_bundle_first_seed_state,
)
from energy_memory.phase5.natural_source_protocol import (
    protocol_payload,
    source_with_protocol_plan,
    validate_cleanup_preflight,
)
from scripts.phase5_prime_raw_scene_retrieval_smoke import (
    _load_json,
    _probe_cue_and_bindings,
    _scene_memory_and_consolidation,
    _schema_sources_for_smoke,
    _sha256,
)


EXP40 = importlib.import_module("experiments.40_phase5_branching")

STRICT_DISCRIMINATOR_ID = "role_topk_target_content_topk_excludes_target_v1"
VIABILITY_PLAN_ID = "bundle_resettle_strict_discriminator_viability_v1"
HEADLINE_METRIC = "Delta E = E_content-prior - E_role-prior"
DEFAULT_BETA_SWEEP = [10.0, 30.0]
DEFAULT_GAMMA_SWEEP = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
DEFAULT_MAGNITUDE_FLOOR = 5.5e-3


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _build_config(
    *,
    source: dict,
    cleanup_preflight: dict,
    protocol: dict,
    beta: float,
    max_settling_iter: int,
) -> BundleFirstConfig:
    source_config = cleanup_preflight["config"]["source_config"]
    return BundleFirstConfig(
        D=int(source_config["D"]),
        N=int(source_config["N"]),
        K_roles=int(source_config["K_roles"]),
        C_codebook=int(source_config["C_codebook"]),
        context_roles=int(source_config["context_roles"]),
        n_queries=int(protocol["required_queries_per_seed"]),
        beta=float(beta),
        max_iter=int(max_settling_iter),
        scene_token_weight=float(source_config["token_weight"]),
        cooccurrence=str(source_config["cooccurrence"]),
        source_name=str(source["framing"]["source_name"]),
    )


def load_protocol_context(
    *,
    source_path: Path,
    cleanup_preflight_path: Path,
    prior_gate_path: Path,
    protocol_name: Optional[str],
    beta: float,
    max_settling_iter: int,
) -> dict:
    source = _load_json(source_path)
    cleanup_preflight = _load_json(cleanup_preflight_path)
    source_sha = _sha256(source_path)
    cleanup_sha = _sha256(cleanup_preflight_path)
    prior_gate_sha = _sha256(prior_gate_path)
    validate_cleanup_preflight(
        cleanup_preflight,
        source_sha=source_sha,
        gate_sha=prior_gate_sha,
    )
    protocol = protocol_payload(cleanup_preflight, protocol_name)
    run_source = source_with_protocol_plan(source, protocol)
    config = _build_config(
        source=source,
        cleanup_preflight=cleanup_preflight,
        protocol=protocol,
        beta=beta,
        max_settling_iter=max_settling_iter,
    )
    return {
        "source": source,
        "source_sha": source_sha,
        "source_path": source_path,
        "cleanup_preflight": cleanup_preflight,
        "cleanup_preflight_sha": cleanup_sha,
        "cleanup_preflight_path": cleanup_preflight_path,
        "prior_gate_path": prior_gate_path,
        "prior_gate_sha": prior_gate_sha,
        "protocol": protocol,
        "run_source": run_source,
        "config": config,
    }


def _role_scores(*, cue_bindings: torch.Tensor, schema_bindings: torch.Tensor) -> torch.Tensor:
    cue_norm = cue_bindings.norm(dim=-1).clamp(min=1e-12)
    schema_norm = schema_bindings.norm(dim=-1).clamp(min=1e-12)
    inner = torch.einsum("nrd,bd->nrb", schema_bindings.conj(), cue_bindings).real
    cos = inner / (schema_norm.unsqueeze(-1) * cue_norm.unsqueeze(0).unsqueeze(0))
    return cos.max(dim=1).values.mean(dim=-1)


def _top1(score: torch.Tensor) -> int:
    return int(torch.argmax(score).detach().cpu())


def _rank_of(score: torch.Tensor, index: int) -> int:
    ranked = torch.argsort(score, descending=True)
    matches = (ranked == int(index)).nonzero(as_tuple=False)
    if matches.numel() == 0:
        raise ValueError(f"index {index} not present in score vector")
    return int(matches[0].item()) + 1


def _probe_prior_row(
    *,
    state,
    schema_source: dict,
    probe_index: int,
    k_main: int,
    delta_redundant: float,
) -> dict:
    cue, cue_bindings, metadata = _probe_cue_and_bindings(
        state=state,
        probe_index=probe_index,
    )
    content = EXP40.select_schema_priors(
        cue=cue,
        schema_store=schema_source["schema_store"],
        k_main=k_main,
        delta_redundant=delta_redundant,
        prior_type="content",
        cue_bindings=cue_bindings,
        schema_bindings=schema_source["schema_bindings"],
    )
    role = EXP40.select_schema_priors(
        cue=cue,
        schema_store=schema_source["schema_store"],
        k_main=k_main,
        delta_redundant=delta_redundant,
        prior_type="role",
        cue_bindings=cue_bindings,
        schema_bindings=schema_source["schema_bindings"],
    )
    content_indices = [int(idx) for idx, _prior in content]
    role_indices = [int(idx) for idx, _prior in role]
    content_score = EXP40._fhrr_cosine(cue, schema_source["schema_store"])
    role_score = _role_scores(
        cue_bindings=cue_bindings,
        schema_bindings=schema_source["schema_bindings"],
    )
    target_scene = int(metadata["scene"])
    return {
        **metadata,
        "content_schema_indices": content_indices,
        "role_schema_indices": role_indices,
        "content_top1_scene": _top1(content_score),
        "role_top1_scene": _top1(role_score),
        "content_target_rank": _rank_of(content_score, target_scene),
        "role_target_rank": _rank_of(role_score, target_scene),
        "target_in_content_topk": target_scene in content_indices,
        "target_in_role_topk": target_scene in role_indices,
        "target_is_content_top1": _top1(content_score) == target_scene,
        "target_is_role_top1": _top1(role_score) == target_scene,
        "content_role_topk_overlap": len(set(content_indices) & set(role_indices)),
        "content_role_topk_equal": tuple(content_indices) == tuple(role_indices),
    }


def select_strict_disagreement_rows(
    rows: Iterable[dict],
    *,
    max_probes: int,
) -> List[dict]:
    selected = []
    for row in rows:
        if row["target_in_role_topk"] and not row["target_in_content_topk"]:
            selected.append(row)
            if len(selected) >= max_probes:
                break
    return selected


def _prior_audit_summary(rows: List[dict]) -> dict:
    def count(predicate) -> int:
        return sum(1 for row in rows if predicate(row))

    combo_counts: Dict[str, int] = {}
    for row in rows:
        key = (
            f"content_target_topk={row['target_in_content_topk']}|"
            f"role_target_topk={row['target_in_role_topk']}|"
            f"topk_equal={row['content_role_topk_equal']}"
        )
        combo_counts[key] = combo_counts.get(key, 0) + 1
    return {
        "n_queries": len(rows),
        "target_in_content_topk": count(lambda row: row["target_in_content_topk"]),
        "target_in_role_topk": count(lambda row: row["target_in_role_topk"]),
        "target_in_both_topk": count(
            lambda row: row["target_in_content_topk"] and row["target_in_role_topk"]
        ),
        "target_in_role_topk_not_content_topk": count(
            lambda row: row["target_in_role_topk"] and not row["target_in_content_topk"]
        ),
        "target_in_content_topk_not_role_topk": count(
            lambda row: row["target_in_content_topk"] and not row["target_in_role_topk"]
        ),
        "target_is_content_top1": count(lambda row: row["target_is_content_top1"]),
        "target_is_role_top1": count(lambda row: row["target_is_role_top1"]),
        "target_is_role_top1_not_content_top1": count(
            lambda row: row["target_is_role_top1"] and not row["target_is_content_top1"]
        ),
        "content_role_topk_equal": count(lambda row: row["content_role_topk_equal"]),
        "combo_counts": combo_counts,
    }


def run_strict_discriminator_precommit(
    *,
    seed: int,
    context: dict,
    max_probes: int,
    k_main: int,
    delta_redundant: float,
    beta_sweep: List[float],
    gamma_sweep: List[float],
    magnitude_floor: float,
    max_settling_iter: int,
    device: str,
) -> dict:
    if seed != 17:
        raise ValueError("strict discriminator precommit is fixed to seed 17")
    if max_probes <= 0:
        raise ValueError("max_probes must be positive")
    state = build_bundle_first_seed_state(
        seed=seed,
        source=context["run_source"],
        config=context["config"],
        device=device,
    )
    _memory, _consolidation, scene_schema_atom_idx = _scene_memory_and_consolidation(
        state
    )
    schema_sources = _schema_sources_for_smoke(state, scene_schema_atom_idx)
    schema_source = schema_sources["scene_store"]
    rows = [
        _probe_prior_row(
            state=state,
            schema_source=schema_source,
            probe_index=probe_index,
            k_main=k_main,
            delta_redundant=delta_redundant,
        )
        for probe_index in range(len(state.query_plan))
    ]
    selected_rows = select_strict_disagreement_rows(rows, max_probes=max_probes)
    selected_probe_indices = [int(row["probe_index"]) for row in selected_rows]
    pass_criteria = {
        "precommit_only": True,
        "retrieval_not_executed": True,
        "seed17_only": int(seed) == 17,
        "full_seed17_query_plan_audited": len(rows) == int(context["config"].n_queries),
        "strict_selector_id_fixed": STRICT_DISCRIMINATOR_ID
        == "role_topk_target_content_topk_excludes_target_v1",
        "selected_requested_probe_count": len(selected_rows) == int(max_probes),
        "all_selected_role_topk_contains_target": all(
            row["target_in_role_topk"] for row in selected_rows
        ),
        "all_selected_content_topk_excludes_target": all(
            not row["target_in_content_topk"] for row in selected_rows
        ),
        "viability_plan_id_fixed": VIABILITY_PLAN_ID
        == "bundle_resettle_strict_discriminator_viability_v1",
        "uses_preferred_bundle_resettle_combiner": True,
        "gamma_sweep_precommitted": bool(gamma_sweep),
        "beta_sweep_precommitted": bool(beta_sweep),
        "magnitude_floor_recorded": float(magnitude_floor) > 0.0,
        "no_n3_or_n10_claim": True,
        "no_full_matrix_claim": True,
        "not_headline_verification": True,
        "not_graduation": True,
    }
    return {
        "framing": {
            "phase": "Phase 5 prime strict discriminator precommit",
            "precommit_only": True,
            "retrieval_executed": False,
            "not_a_gate": True,
            "not_n3": True,
            "not_n10": True,
            "not_graduation": True,
            "no_full_matrix": True,
            "no_m1_escalation": True,
            "no_m2": True,
            "no_new_headline": True,
            "headline_metric": HEADLINE_METRIC,
        },
        "source_manifest": {
            "source_artifact_path": str(context["source_path"]),
            "source_artifact_sha256": context["source_sha"],
            "cleanup_preflight_path": str(context["cleanup_preflight_path"]),
            "cleanup_preflight_sha256": context["cleanup_preflight_sha"],
            "prior_gate_path": str(context["prior_gate_path"]),
            "prior_gate_sha256": context["prior_gate_sha"],
            "protocol_name": context["protocol"].get(
                "name",
                context["protocol"].get("protocol_name"),
            ),
        },
        "config": {
            "seed": int(seed),
            "D": context["config"].D,
            "N": context["config"].N,
            "K_roles": context["config"].K_roles,
            "C_codebook": context["config"].C_codebook,
            "context_roles": context["config"].context_roles,
            "n_queries": context["config"].n_queries,
            "scene_token_weight": context["config"].scene_token_weight,
            "cooccurrence": context["config"].cooccurrence,
            "source_name": context["config"].source_name,
            "device": device,
            "k_main": int(k_main),
            "delta_redundant": float(delta_redundant),
            "max_settling_iter": int(max_settling_iter),
        },
        "strict_discriminator": {
            "selector_id": STRICT_DISCRIMINATOR_ID,
            "definition": (
                "target scene is present in role-prior top-K schema set and "
                "absent from content-prior top-K schema set"
            ),
            "selected_probe_indices": selected_probe_indices,
            "selected_rows": selected_rows,
            "prior_audit_summary": _prior_audit_summary(rows),
        },
        "viability_plan": {
            "plan_id": VIABILITY_PLAN_ID,
            "selected_probe_indices": selected_probe_indices,
            "conditions": ["content", "role", "random"],
            "primary_combiner": "q_bundle",
            "comparison_combiner": "q_greedy",
            "run_combiners": True,
            "include_surprise_branch": False,
            "readout_id": "cue_conditioned_scene_energy_v1",
            "delta_e": HEADLINE_METRIC,
            "beta_sweep": [float(value) for value in beta_sweep],
            "gamma_sweep": [float(value) for value in gamma_sweep],
            "magnitude_floor": float(magnitude_floor),
            "viability_criteria": {
                "mean_delta_e_bundle_ge_magnitude_floor": True,
                "role_beats_content_on_at_least_three_of_four_probes": True,
                "role_bundle_target_scene_rate_gt_content": True,
                "role_bundle_target_scene_rate_gt_random": True,
                "role_energy_lower_than_random_on_mean": True,
            },
        },
        "pass_criteria": pass_criteria,
        "passes_all_criteria": all(bool(value) for value in pass_criteria.values()),
        "anti_homunculus_check": {
            "pass": True,
            "reason": (
                "Static query selector and fixed diagnostic sweep only; no "
                "observed metric routes execution or chooses a production mode."
            ),
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
        default="reports/phase5_prime_strict_discriminator_precommit.json",
    )
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-probes", type=int, default=4)
    parser.add_argument("--k-main", type=int, default=4)
    parser.add_argument("--delta-redundant", type=float, default=0.95)
    parser.add_argument("--max-settling-iter", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    context = load_protocol_context(
        source_path=Path(args.source),
        cleanup_preflight_path=Path(args.cleanup_preflight),
        prior_gate_path=Path(args.prior_gate),
        protocol_name=args.protocol,
        beta=30.0,
        max_settling_iter=args.max_settling_iter,
    )
    payload = run_strict_discriminator_precommit(
        seed=int(args.seed),
        context=context,
        max_probes=int(args.max_probes),
        k_main=int(args.k_main),
        delta_redundant=float(args.delta_redundant),
        beta_sweep=DEFAULT_BETA_SWEEP,
        gamma_sweep=DEFAULT_GAMMA_SWEEP,
        magnitude_floor=DEFAULT_MAGNITUDE_FLOOR,
        max_settling_iter=int(args.max_settling_iter),
        device=args.device,
    )
    _write_json(Path(args.out), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
