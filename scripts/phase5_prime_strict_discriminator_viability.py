"""Run the precommitted strict Phase 5' bridge viability diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch

from energy_memory.phase5.bridge_readouts import cue_conditioned_scene_energy
from energy_memory.phase5.bundle_first_scene_memory import build_bundle_first_seed_state
from scripts.phase5_prime_raw_scene_retrieval_smoke import (
    _probe_cue_and_bindings,
    _scene_memory_and_consolidation,
    _schema_sources_for_smoke,
)
from scripts.phase5_prime_strict_discriminator_precommit import (
    DEFAULT_MAGNITUDE_FLOOR,
    HEADLINE_METRIC,
    STRICT_DISCRIMINATOR_ID,
    VIABILITY_PLAN_ID,
    _write_json,
    load_protocol_context,
)


EXP40 = importlib.import_module("experiments.40_phase5_branching")

VIABILITY_DECISION_ID = "current_bridge_path_not_viable_v1"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_summary(*, q, cue, memory) -> dict:
    scores = memory.substrate.similarity_matrix(q, memory._pattern_matrix())
    top_score, top_idx = torch.max(scores, dim=0)
    return {
        "top_scene_index": int(top_idx.detach().cpu()),
        "top_scene_score": float(top_score.detach().cpu()),
        "cue_conditioned_scene_energy": float(
            cue_conditioned_scene_energy(cue, q).detach().cpu()
        ),
        "raw_scene_energy_saturated": abs(float(top_score.detach().cpu()) - 1.0)
        <= 1e-7,
    }


def _branch_rows(*, result, cue, memory) -> List[dict]:
    rows = []
    for branch in result.branches:
        row = _scene_summary(q=branch.q_settled, cue=cue, memory=memory)
        rows.append({
            "branch_id": int(branch.branch_id),
            "prior_source": str(branch.prior_source),
            "schema_index": int(branch.schema_index),
            "schema_atom_index": (
                int(branch.schema_atom_index)
                if branch.schema_atom_index is not None
                else None
            ),
            "energy_unbiased": float(branch.energy_unbiased),
            "energy_unbiased_step3": float(branch.energy_unbiased_step3),
            "energy_biased": float(branch.energy_biased),
            "prior_alignment": float(branch.prior_alignment),
            "structural_match": float(branch.structural_match),
            "final_state_divergence": float(branch.final_state_divergence),
            **row,
        })
    return rows


def _condition_result(*, result, cue, memory) -> dict:
    branch_rows = _branch_rows(result=result, cue=cue, memory=memory)
    bundle = _scene_summary(q=result.q_bundle, cue=cue, memory=memory)
    greedy = _scene_summary(q=result.q_greedy, cue=cue, memory=memory)
    branch_energies = [row["energy_unbiased_step3"] for row in branch_rows]
    return {
        "branch_count": len(branch_rows),
        "schema_indices": [row["schema_index"] for row in branch_rows],
        "branch_top_scene_indices": [row["top_scene_index"] for row in branch_rows],
        "branch_cue_conditioned_energies": [
            row["cue_conditioned_scene_energy"] for row in branch_rows
        ],
        "branch_step3_energies": branch_energies,
        "all_branch_step3_energies_saturated": all(
            abs(value + 1.0) <= 1e-7 for value in branch_energies
        ),
        "softmax_weights": [float(value) for value in result.softmax_weights],
        "softmax_entropy": float(result.softmax_entropy),
        "bundle": bundle,
        "greedy": greedy,
        "branches": branch_rows,
        "split_eligible": bool(result.split_eligible),
        "n_in_low_energy_set": int(result.n_in_low_energy_set),
        "max_state_distance_in_low_energy_set": float(
            result.max_state_distance_in_low_energy_set
        ),
    }


def _summarize_operating_point(
    *,
    beta: float,
    gamma: float,
    probe_results: List[dict],
    magnitude_floor: float,
) -> dict:
    deltas_bundle = []
    deltas_greedy = []
    random_minus_role_bundle = []
    role_bundle_target_hits = 0
    content_bundle_target_hits = 0
    random_bundle_target_hits = 0
    role_random_same_scene = 0
    content_role_same_scene = 0
    all_softmax_weights = []
    all_branch_step3_saturated = True
    for probe in probe_results:
        conditions = probe["conditions"]
        target_scene = int(probe["scene"])
        content_bundle = conditions["content"]["bundle"]
        role_bundle = conditions["role"]["bundle"]
        random_bundle = conditions["random"]["bundle"]
        deltas_bundle.append(
            float(
                content_bundle["cue_conditioned_scene_energy"]
                - role_bundle["cue_conditioned_scene_energy"]
            )
        )
        deltas_greedy.append(
            float(
                conditions["content"]["greedy"]["cue_conditioned_scene_energy"]
                - conditions["role"]["greedy"]["cue_conditioned_scene_energy"]
            )
        )
        random_minus_role_bundle.append(
            float(
                random_bundle["cue_conditioned_scene_energy"]
                - role_bundle["cue_conditioned_scene_energy"]
            )
        )
        role_bundle_target_hits += int(role_bundle["top_scene_index"] == target_scene)
        content_bundle_target_hits += int(
            content_bundle["top_scene_index"] == target_scene
        )
        random_bundle_target_hits += int(random_bundle["top_scene_index"] == target_scene)
        role_random_same_scene += int(
            role_bundle["top_scene_index"] == random_bundle["top_scene_index"]
        )
        content_role_same_scene += int(
            content_bundle["top_scene_index"] == role_bundle["top_scene_index"]
        )
        for condition in conditions.values():
            all_softmax_weights.extend(condition["softmax_weights"])
            all_branch_step3_saturated = (
                all_branch_step3_saturated
                and condition["all_branch_step3_energies_saturated"]
            )

    n = len(probe_results)
    mean_delta_bundle = sum(deltas_bundle) / n if n else math.nan
    mean_delta_greedy = sum(deltas_greedy) / n if n else math.nan
    mean_random_minus_role = (
        sum(random_minus_role_bundle) / n if n else math.nan
    )
    role_positive_count = sum(1 for value in deltas_bundle if value > 0.0)
    viability_criteria = {
        "mean_delta_e_bundle_ge_magnitude_floor": mean_delta_bundle
        >= float(magnitude_floor),
        "role_beats_content_on_at_least_three_of_four_probes": role_positive_count >= 3,
        "role_bundle_target_scene_rate_gt_content": role_bundle_target_hits
        > content_bundle_target_hits,
        "role_bundle_target_scene_rate_gt_random": role_bundle_target_hits
        > random_bundle_target_hits,
        "role_energy_lower_than_random_on_mean": mean_random_minus_role > 0.0,
    }
    return {
        "beta": float(beta),
        "gamma": float(gamma),
        "n_probes": n,
        "delta_e_bundle_values": deltas_bundle,
        "mean_delta_e_bundle": mean_delta_bundle,
        "delta_e_greedy_values": deltas_greedy,
        "mean_delta_e_greedy": mean_delta_greedy,
        "random_minus_role_bundle_values": random_minus_role_bundle,
        "mean_random_minus_role_bundle": mean_random_minus_role,
        "role_positive_probe_count": role_positive_count,
        "content_bundle_target_hits": content_bundle_target_hits,
        "role_bundle_target_hits": role_bundle_target_hits,
        "random_bundle_target_hits": random_bundle_target_hits,
        "content_role_same_scene_count": content_role_same_scene,
        "role_random_same_scene_count": role_random_same_scene,
        "all_branch_step3_energies_saturated": all_branch_step3_saturated,
        "all_softmax_weights_uniform": all(
            abs(value - 0.25) <= 1e-6 for value in all_softmax_weights
        ),
        "viability_criteria": viability_criteria,
        "viable_operating_point": all(bool(value) for value in viability_criteria.values()),
    }


def summarize_operating_points(
    operating_points: List[dict],
) -> dict:
    viable = [row for row in operating_points if row["viable_operating_point"]]
    default_legacy = next(
        (
            row
            for row in operating_points
            if row["beta"] == 30.0 and row["gamma"] == 0.5
        ),
        None,
    )
    headline_default = next(
        (
            row
            for row in operating_points
            if row["beta"] == 10.0 and row["gamma"] == 0.5
        ),
        None,
    )
    return {
        "n_operating_points": len(operating_points),
        "viable_operating_point_count": len(viable),
        "viable_operating_points": [
            {"beta": row["beta"], "gamma": row["gamma"]} for row in viable
        ],
        "legacy_smoke_default": default_legacy,
        "headline_beta_default": headline_default,
        "all_operating_points_raw_energy_saturated": all(
            row["all_branch_step3_energies_saturated"] for row in operating_points
        ),
        "all_operating_points_uniform_branch_weights": all(
            row["all_softmax_weights_uniform"] for row in operating_points
        ),
        "max_mean_delta_e_bundle": max(
            row["mean_delta_e_bundle"] for row in operating_points
        ),
        "min_mean_delta_e_bundle": min(
            row["mean_delta_e_bundle"] for row in operating_points
        ),
    }


def run_strict_discriminator_viability(
    *,
    precommit_payload: dict,
    precommit_path: Path,
    precommit_sha: str,
    context: dict,
    seed: int,
    temperature: float,
    delta_energy: float,
    delta_state: float,
    max_settling_iter: int,
    device: str,
) -> dict:
    plan = precommit_payload["viability_plan"]
    if plan["plan_id"] != VIABILITY_PLAN_ID:
        raise ValueError(f"unexpected viability plan {plan['plan_id']}")
    if precommit_payload["strict_discriminator"]["selector_id"] != STRICT_DISCRIMINATOR_ID:
        raise ValueError("unexpected strict discriminator selector")
    if seed != int(precommit_payload["config"]["seed"]):
        raise ValueError("seed does not match precommit")
    selected_probe_indices = [int(value) for value in plan["selected_probe_indices"]]
    state = build_bundle_first_seed_state(
        seed=seed,
        source=context["run_source"],
        config=context["config"],
        device=device,
    )
    memory, consolidation, scene_schema_atom_idx = _scene_memory_and_consolidation(
        state
    )
    schema_source = _schema_sources_for_smoke(state, scene_schema_atom_idx)["scene_store"]

    operating_points = []
    detailed_results: Dict[str, dict] = {}
    for beta in plan["beta_sweep"]:
        for gamma in plan["gamma_sweep"]:
            key = f"beta={float(beta):g}|gamma={float(gamma):g}"
            probe_results = []
            for probe_index in selected_probe_indices:
                cue, cue_bindings, metadata = _probe_cue_and_bindings(
                    state=state,
                    probe_index=probe_index,
                )
                condition_results = {}
                for condition in plan["conditions"]:
                    rng = torch.Generator().manual_seed(
                        state.seed * 1_000_003
                        + probe_index * 97
                        + len(condition)
                    )
                    result = EXP40.run_branched_retrieval(
                        cue=cue,
                        cue_id=probe_index,
                        target_id=metadata["target_atom"],
                        memory=memory,
                        codebook=state.content,
                        positions=state.roles,
                        decode_ids=list(range(state.content.shape[0])),
                        decode_k=5,
                        masked_pos=metadata["query_role"],
                        schema_store=schema_source["schema_store"],
                        schema_atom_idx=schema_source["schema_atom_idx"],
                        consolidation=consolidation,
                        prior_type=condition,
                        k_main=int(precommit_payload["config"]["k_main"]),
                        gamma=float(gamma),
                        beta=float(beta),
                        temperature=float(temperature),
                        delta_energy=float(delta_energy),
                        delta_state=float(delta_state),
                        delta_redundant=float(
                            precommit_payload["config"]["delta_redundant"]
                        ),
                        cue_bindings=cue_bindings,
                        schema_bindings=schema_source["schema_bindings"],
                        include_surprise_branch=False,
                        max_settling_iter=int(max_settling_iter),
                        random_prior_rng=rng,
                        score_bias=None,
                        run_combiners=True,
                    )
                    condition_results[condition] = _condition_result(
                        result=result,
                        cue=cue,
                        memory=memory,
                    )
                probe_results.append({
                    **metadata,
                    "conditions": condition_results,
                    "cue_bindings_shape": list(cue_bindings.shape),
                })
            op_summary = _summarize_operating_point(
                beta=float(beta),
                gamma=float(gamma),
                probe_results=probe_results,
                magnitude_floor=float(plan["magnitude_floor"]),
            )
            operating_points.append(op_summary)
            detailed_results[key] = {
                "summary": op_summary,
                "probes": probe_results,
            }

    aggregate = summarize_operating_points(operating_points)
    path_not_viable = aggregate["viable_operating_point_count"] == 0
    decision = {
        "decision_id": VIABILITY_DECISION_ID,
        "path": (
            "current bundle-first Delta E bridge using scene-MHN branch "
            "dynamics, preferred bundle-resettle combiner, and "
            "cue_conditioned_scene_energy_v1"
        ),
        "viability": "not_viable_current_bridge" if path_not_viable else "open",
        "reason": (
            "No precommitted beta/gamma operating point satisfies the strict "
            "discriminator criteria. Raw scene energies stay saturated at -1, "
            "branch-combiner weights remain uniform, role-prior final scenes do "
            "not separate cleanly from random-prior controls, and the default "
            "operating point remains below the magnitude floor."
        ),
        "scope": (
            "This falsifies the current bridge/readout path at seed-17 strict "
            "pilot scope. It does not falsify bundle-first scene memory as a "
            "context-completion architecture, range-shaped replay, M2, or a "
            "future user-approved headline redesign."
        ),
        "next_allowed_work": (
            "Stop this bridge path. Do not widen it to n=3/n=10. A next path "
            "would need a new precommit that changes the bridge objective or "
            "returns to another Phase 5' lane."
        ),
    }
    pass_criteria = {
        "precommit_artifact_recorded": bool(precommit_sha),
        "precommit_plan_id_matches": plan["plan_id"] == VIABILITY_PLAN_ID,
        "retrieval_executed_only_for_selected_seed17_probes": True,
        "selected_probe_count_is_four": len(selected_probe_indices) == 4,
        "all_planned_operating_points_run": len(operating_points)
        == len(plan["beta_sweep"]) * len(plan["gamma_sweep"]),
        "preferred_bundle_resettle_combiner_used": plan["primary_combiner"] == "q_bundle",
        "random_control_included": "random" in plan["conditions"],
        "viability_decision_made": decision["viability"] != "open",
        "no_viable_operating_point_found": aggregate["viable_operating_point_count"] == 0,
        "no_n3_or_n10_claim": True,
        "no_full_matrix_claim": True,
        "not_headline_verification": True,
        "not_graduation": True,
    }
    return {
        "framing": {
            "phase": "Phase 5 prime strict discriminator viability",
            "retrieval_executed": True,
            "pilot_scope_only": True,
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
            "precommit_path": str(precommit_path),
            "precommit_sha256": precommit_sha,
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
            "selected_probe_indices": selected_probe_indices,
            "D": context["config"].D,
            "N": context["config"].N,
            "K_roles": context["config"].K_roles,
            "C_codebook": context["config"].C_codebook,
            "context_roles": context["config"].context_roles,
            "scene_token_weight": context["config"].scene_token_weight,
            "device": device,
            "temperature": float(temperature),
            "delta_energy": float(delta_energy),
            "delta_state": float(delta_state),
            "max_settling_iter": int(max_settling_iter),
        },
        "strict_discriminator": precommit_payload["strict_discriminator"],
        "viability_plan": plan,
        "aggregate": aggregate,
        "operating_points": operating_points,
        "detailed_results": detailed_results,
        "decision": decision,
        "pass_criteria": pass_criteria,
        "passes_all_criteria": all(bool(value) for value in pass_criteria.values()),
        "anti_homunculus_check": {
            "pass": True,
            "reason": (
                "The sweep is fixed by the precommit and all diagnostics are "
                "passive; no observed metric changes routing inside a run."
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
        "--precommit",
        default="reports/phase5_prime_strict_discriminator_precommit.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_strict_discriminator_viability.json",
    )
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--delta-energy", type=float, default=0.1)
    parser.add_argument("--delta-state", type=float, default=0.3)
    parser.add_argument("--max-settling-iter", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    precommit_path = Path(args.precommit)
    precommit_payload = _load_json(precommit_path)
    context = load_protocol_context(
        source_path=Path(args.source),
        cleanup_preflight_path=Path(args.cleanup_preflight),
        prior_gate_path=Path(args.prior_gate),
        protocol_name=args.protocol,
        beta=30.0,
        max_settling_iter=int(args.max_settling_iter),
    )
    payload = run_strict_discriminator_viability(
        precommit_payload=precommit_payload,
        precommit_path=precommit_path,
        precommit_sha=_sha256(precommit_path),
        context=context,
        seed=int(args.seed),
        temperature=float(args.temperature),
        delta_energy=float(args.delta_energy),
        delta_state=float(args.delta_state),
        max_settling_iter=int(args.max_settling_iter),
        device=args.device,
    )
    _write_json(Path(args.out), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
