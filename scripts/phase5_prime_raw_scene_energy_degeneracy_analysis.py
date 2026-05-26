"""Analyze raw-scene-energy degeneracy from the Report 103 smoke.

This reruns the exact seed-17/four-probe smoke and records score-level evidence
needed to explain why all paired raw-scene Delta E values collapse to zero.
It is analysis-only and does not widen evidence scale.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch

from energy_memory.phase5.bundle_first_scene_memory import BundleFirstConfig
from energy_memory.phase5.natural_source_protocol import (
    protocol_payload,
    source_with_protocol_plan,
    validate_cleanup_preflight,
)
from scripts.phase5_prime_raw_scene_controls_preflight import (
    BASELINE_ID,
    HEADLINE_METRIC,
    _config_mismatches,
    planned_control_cells,
)
from scripts.phase5_prime_raw_scene_retrieval_smoke import (
    _load_json,
    _probe_cue_and_bindings,
    _resolve_device,
    _scene_memory_and_consolidation,
    _schema_sources_for_smoke,
    _sha256,
    _stable_seed,
)
from energy_memory.phase5.bundle_first_scene_memory import (
    build_bundle_first_seed_state,
)


EXP40 = importlib.import_module("experiments.40_phase5_branching")


def _state_score_summary(*, branch, memory, beta: float, target_scene: int) -> dict:
    patterns = memory._pattern_matrix()
    scores = memory.substrate.similarity_matrix(branch.q_settled, patterns)
    top_score_t, top_idx_t = torch.max(scores, dim=0)
    max_logit = beta * top_score_t
    logsumexp = torch.logsumexp(beta * scores, dim=0)
    energy = -logsumexp / beta
    top_idx = int(top_idx_t.detach().cpu())
    top_score = float(top_score_t.detach().cpu())
    logsumexp_excess = float((logsumexp - max_logit).detach().cpu())
    energy_value = float(energy.detach().cpu())
    return {
        "branch_id": int(branch.branch_id),
        "prior_source": str(branch.prior_source),
        "schema_index": int(branch.schema_index),
        "schema_atom_index": (
            int(branch.schema_atom_index)
            if branch.schema_atom_index is not None
            else None
        ),
        "top_scene_index": top_idx,
        "target_scene": int(target_scene),
        "top_equals_target_scene": top_idx == int(target_scene),
        "top_score": top_score,
        "raw_energy_recomputed": energy_value,
        "raw_energy_from_branch": float(branch.energy_unbiased),
        "step3_energy_from_branch": float(branch.energy_unbiased_step3),
        "energy_abs_diff_recomputed_vs_branch": abs(
            energy_value - float(branch.energy_unbiased_step3)
        ),
        "gap_to_negative_one": abs(energy_value + 1.0),
        "logsumexp_excess_over_top_logit": logsumexp_excess,
        "top_score_saturated": top_score >= 1.0 - 1e-7,
        "energy_saturated_at_negative_one": abs(energy_value + 1.0) <= 1e-4,
        "step3_equals_raw": abs(
            float(branch.energy_unbiased_step3) - float(branch.energy_unbiased)
        )
        <= 1e-7,
    }


def _run_probe_cell_details(
    *,
    state,
    memory,
    consolidation,
    schema_sources: dict,
    cell: dict,
    probe_index: int,
    temperature: float,
    delta_energy: float,
    delta_state: float,
    delta_redundant: float,
    max_settling_iter: int,
) -> dict:
    cue, cue_bindings, metadata = _probe_cue_and_bindings(
        state=state,
        probe_index=probe_index,
    )
    source = schema_sources[cell["schema_source"]]
    rng = torch.Generator().manual_seed(
        _stable_seed(state.seed, probe_index, cell["cell_id"], "random-prior")
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
        schema_store=source["schema_store"],
        schema_atom_idx=source["schema_atom_idx"],
        consolidation=consolidation,
        prior_type=cell["prior_type"],
        k_main=int(cell["k_main"]),
        gamma=float(cell["gamma"]),
        beta=float(cell["beta"]),
        temperature=temperature,
        delta_energy=delta_energy,
        delta_state=delta_state,
        delta_redundant=delta_redundant,
        cue_bindings=cue_bindings,
        schema_bindings=source["schema_bindings"],
        include_surprise_branch=False,
        max_settling_iter=max_settling_iter,
        random_prior_rng=rng,
        score_bias=None,
        run_combiners=False,
    )
    branch_details = [
        _state_score_summary(
            branch=branch,
            memory=memory,
            beta=float(cell["beta"]),
            target_scene=metadata["scene"],
        )
        for branch in result.branches
    ]
    return {
        **metadata,
        "branch_count": len(result.branches),
        "min_energy_step3": min(
            row["step3_energy_from_branch"] for row in branch_details
        ),
        "min_recomputed_raw_energy": min(
            row["raw_energy_recomputed"] for row in branch_details
        ),
        "top_scene_indices": [row["top_scene_index"] for row in branch_details],
        "top_scores": [row["top_score"] for row in branch_details],
        "branch_details": branch_details,
    }


def _paired_delta_summary(cell_results: Dict[str, dict]) -> dict:
    by_group: Dict[str, Dict[str, dict]] = {}
    for cell_id, result in cell_results.items():
        group = result["paired_delta_group"]
        prior_type = result["prior_type"]
        if group is None or prior_type not in {"content", "role"}:
            continue
        by_group.setdefault(group, {})[prior_type] = result
    out = {}
    for group, pair in sorted(by_group.items()):
        values = []
        per_probe = []
        for content_probe, role_probe in zip(
            pair["content"]["probes"],
            pair["role"]["probes"],
        ):
            delta = float(
                content_probe["min_energy_step3"] - role_probe["min_energy_step3"]
            )
            values.append(delta)
            per_probe.append({
                "probe_index": int(content_probe["probe_index"]),
                "delta_e": delta,
                "content_min_energy_step3": content_probe["min_energy_step3"],
                "role_min_energy_step3": role_probe["min_energy_step3"],
            })
        out[group] = {
            "delta_e_values": values,
            "mean_delta_e": sum(values) / len(values) if values else None,
            "all_zero": all(abs(value) <= 1e-9 for value in values),
            "per_probe": per_probe,
        }
    return out


def run_raw_scene_energy_degeneracy_analysis(
    *,
    seed: int,
    source: dict,
    source_path: Path,
    source_sha: str,
    cleanup_preflight_path: Optional[Path],
    cleanup_preflight_sha: Optional[str],
    prior_gate_path: Optional[Path],
    prior_gate_sha: Optional[str],
    smoke_artifact_path: Optional[Path],
    smoke_artifact_sha: Optional[str],
    config: BundleFirstConfig,
    protocol_name: str,
    max_probes: int,
    beta: float,
    gamma: float,
    k_main: int,
    temperature: float,
    delta_energy: float,
    delta_state: float,
    delta_redundant: float,
    max_settling_iter: int,
    device: str,
) -> dict:
    if seed != 17:
        raise ValueError("degeneracy analysis is fixed to seed 17")
    if max_probes <= 0 or max_probes > 4:
        raise ValueError("degeneracy analysis is fixed to at most four probes")
    state = build_bundle_first_seed_state(
        seed=seed,
        source=source,
        config=config,
        device=device,
    )
    memory, consolidation, scene_schema_atom_idx = _scene_memory_and_consolidation(
        state
    )
    schema_sources = _schema_sources_for_smoke(state, scene_schema_atom_idx)
    cells = planned_control_cells(config=config, beta=beta, gamma=gamma, k_main=k_main)
    n_probe_cues = min(max_probes, len(state.query_plan))

    cell_results = {}
    for cell in cells:
        probes = [
            _run_probe_cell_details(
                state=state,
                memory=memory,
                consolidation=consolidation,
                schema_sources=schema_sources,
                cell=cell,
                probe_index=probe_index,
                temperature=temperature,
                delta_energy=delta_energy,
                delta_state=delta_state,
                delta_redundant=delta_redundant,
                max_settling_iter=max_settling_iter,
            )
            for probe_index in range(n_probe_cues)
        ]
        cell_results[cell["cell_id"]] = {
            **cell,
            "probes": probes,
            "retrieval_executed": True,
        }

    branch_rows = [
        branch
        for cell_result in cell_results.values()
        for probe in cell_result["probes"]
        for branch in probe["branch_details"]
    ]
    paired_delta = _paired_delta_summary(cell_results)
    top_indices = [int(row["top_scene_index"]) for row in branch_rows]
    pass_criteria = {
        "analysis_fixed_to_report103_scope": seed == 17 and n_probe_cues <= 4,
        "all_nine_cells_rerun": len(cell_results) == 9,
        "all_paired_delta_groups_zero": all(
            row["all_zero"] for row in paired_delta.values()
        ),
        "all_final_top_scores_saturated": all(
            row["top_score_saturated"] for row in branch_rows
        ),
        "all_energies_saturated_at_negative_one": all(
            row["energy_saturated_at_negative_one"] for row in branch_rows
        ),
        "all_step3_equals_raw": all(row["step3_equals_raw"] for row in branch_rows),
        "recomputed_energy_matches_branch_readback": all(
            row["energy_abs_diff_recomputed_vs_branch"] <= 1e-7
            for row in branch_rows
        ),
        "top_scene_indices_not_all_identical": len(set(top_indices)) > 1,
        "no_wider_evidence_claim": True,
    }
    return {
        "framing": {
            "phase": "Phase 5 prime raw-scene energy degeneracy analysis",
            "baseline_id": BASELINE_ID,
            "analysis_only": True,
            "not_a_gate": True,
            "not_n3": True,
            "not_n10": True,
            "not_graduation": True,
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
            "smoke_artifact_path": (
                str(smoke_artifact_path) if smoke_artifact_path else None
            ),
            "smoke_artifact_sha256": smoke_artifact_sha,
            "protocol_name": protocol_name,
        },
        "config": {
            "seed": int(seed),
            "D": config.D,
            "N": config.N,
            "K_roles": config.K_roles,
            "C_codebook": config.C_codebook,
            "context_roles": config.context_roles,
            "n_queries_available": config.n_queries,
            "n_probe_cues": n_probe_cues,
            "scene_token_weight": config.scene_token_weight,
            "cooccurrence": config.cooccurrence,
            "source_name": config.source_name,
            "device": device,
            "beta": float(beta),
            "gamma": float(gamma),
            "k_main": int(k_main),
            "temperature": float(temperature),
            "delta_energy": float(delta_energy),
            "delta_state": float(delta_state),
            "delta_redundant": float(delta_redundant),
            "max_settling_iter": int(max_settling_iter),
            "score_bias": None,
        },
        "legacy_headline_default_mismatches": _config_mismatches(
            beta=beta,
            gamma=gamma,
            k_main=k_main,
        ),
        "paired_delta": paired_delta,
        "aggregate": {
            "total_branches": len(branch_rows),
            "unique_top_scene_indices": sorted(set(top_indices)),
            "unique_top_scene_index_count": len(set(top_indices)),
            "top_score_min": min(row["top_score"] for row in branch_rows),
            "top_score_max": max(row["top_score"] for row in branch_rows),
            "max_gap_to_negative_one": max(
                row["gap_to_negative_one"] for row in branch_rows
            ),
            "max_logsumexp_excess_over_top_logit": max(
                row["logsumexp_excess_over_top_logit"] for row in branch_rows
            ),
            "target_scene_top_rate": sum(
                1 for row in branch_rows if row["top_equals_target_scene"]
            )
            / len(branch_rows),
        },
        "cell_results": cell_results,
        "pass_criteria": pass_criteria,
        "passes_all_criteria": all(bool(value) for value in pass_criteria.values()),
        "diagnosis": {
            "raw_scene_energy_v0_degenerate": all(
                bool(value) for value in pass_criteria.values()
            ),
            "reason": "Every branch settles exactly onto a normalized stored scene bundle, so raw softmax energy is saturated at the self-similarity ceiling E=-1 regardless of which scene is selected.",
            "implication": "The settled-state raw scene-MHN energy is not a discriminative Delta E readout for this bridge. Do not widen until a new fixed non-saturated bridge-energy readout is precommitted.",
        },
        "anti_homunculus_check": {
            "pass": True,
            "reason": "Analysis reruns fixed cells and logs passive score diagnostics only; no adaptive routing or best-condition selection is introduced.",
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
        "--smoke-artifact",
        default="reports/phase5_prime_raw_scene_retrieval_smoke_seed17.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_raw_scene_energy_degeneracy_analysis.json",
    )
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-probes", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--k-main", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--delta-energy", type=float, default=0.1)
    parser.add_argument("--delta-state", type=float, default=0.3)
    parser.add_argument("--delta-redundant", type=float, default=0.95)
    parser.add_argument("--max-settling-iter", type=int, default=10)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    source_path = Path(args.source)
    cleanup_path = Path(args.cleanup_preflight)
    prior_gate_path = Path(args.prior_gate)
    smoke_path = Path(args.smoke_artifact)
    source = _load_json(source_path)
    cleanup_preflight = _load_json(cleanup_path)
    source_sha = _sha256(source_path)
    cleanup_sha = _sha256(cleanup_path)
    prior_gate_sha = _sha256(prior_gate_path)
    smoke_sha = _sha256(smoke_path) if smoke_path.exists() else None
    validate_cleanup_preflight(
        cleanup_preflight,
        source_sha=source_sha,
        gate_sha=prior_gate_sha,
    )
    protocol = protocol_payload(cleanup_preflight, args.protocol)
    run_source = source_with_protocol_plan(source, protocol)
    source_config = cleanup_preflight["config"]["source_config"]
    if args.seed not in [int(seed) for seed in source_config["seeds"]]:
        raise ValueError(f"seed {args.seed} is not in the source config seed set")

    config = BundleFirstConfig(
        D=int(source_config["D"]),
        N=int(source_config["N"]),
        K_roles=int(source_config["K_roles"]),
        C_codebook=int(source_config["C_codebook"]),
        context_roles=int(source_config["context_roles"]),
        n_queries=int(protocol["required_queries_per_seed"]),
        beta=float(args.beta),
        max_iter=int(args.max_settling_iter),
        scene_token_weight=float(source_config["token_weight"]),
        cooccurrence=str(source_config["cooccurrence"]),
        source_name=str(source["framing"]["source_name"]),
    )
    payload = run_raw_scene_energy_degeneracy_analysis(
        seed=int(args.seed),
        source=run_source,
        source_path=source_path,
        source_sha=source_sha,
        cleanup_preflight_path=cleanup_path,
        cleanup_preflight_sha=cleanup_sha,
        prior_gate_path=prior_gate_path,
        prior_gate_sha=prior_gate_sha,
        smoke_artifact_path=smoke_path if smoke_path.exists() else None,
        smoke_artifact_sha=smoke_sha,
        config=config,
        protocol_name=str(protocol["protocol_name"]),
        max_probes=int(args.max_probes),
        beta=float(args.beta),
        gamma=float(args.gamma),
        k_main=int(args.k_main),
        temperature=float(args.temperature),
        delta_energy=float(args.delta_energy),
        delta_state=float(args.delta_state),
        delta_redundant=float(args.delta_redundant),
        max_settling_iter=int(args.max_settling_iter),
        device=device,
    )
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {out_path} passes={payload['passes_all_criteria']} "
        f"branches={payload['aggregate']['total_branches']} "
        f"unique_top_scene_indices={payload['aggregate']['unique_top_scene_index_count']} "
        f"device={device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
