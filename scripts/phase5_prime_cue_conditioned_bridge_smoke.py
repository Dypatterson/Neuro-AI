"""Pilot smoke for the cue-conditioned Phase 5' bridge readout.

This runs the exact Report 103 seed-17/four-probe/nine-cell surface, but scores
final scene states with ``cue_conditioned_scene_energy_v1`` from Report 105.
It is not n=3/n=10 evidence, a gate, a full matrix, or a graduation run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch

from energy_memory.phase5.bridge_readouts import (
    READOUT_ID_CUE_CONDITIONED_SCENE,
    cue_conditioned_scene_energy,
    delta_e_content_minus_role,
)
from energy_memory.phase5.bundle_first_scene_memory import BundleFirstConfig
from energy_memory.phase5.natural_source_protocol import (
    protocol_payload,
    source_with_protocol_plan,
    validate_cleanup_preflight,
)
from scripts.phase5_prime_raw_scene_controls_preflight import (
    HEADLINE_METRIC,
    _config_mismatches,
    planned_control_cells,
)
from scripts.phase5_prime_raw_scene_retrieval_smoke import (
    EXP40,
    _load_json,
    _probe_cue_and_bindings,
    _resolve_device,
    _scene_memory_and_consolidation,
    _schema_sources_for_smoke,
    _sha256,
    _stable_seed,
)
from energy_memory.phase5.bundle_first_scene_memory import build_bundle_first_seed_state


def _branch_bridge_summary(*, branch, cue, memory) -> dict:
    bridge_energy = cue_conditioned_scene_energy(cue, branch.q_settled)
    scene_scores = memory.substrate.similarity_matrix(
        branch.q_settled,
        memory._pattern_matrix(),
    )
    top_scene_score, top_scene_idx = torch.max(scene_scores, dim=0)
    return {
        "branch_id": int(branch.branch_id),
        "prior_source": str(branch.prior_source),
        "schema_index": int(branch.schema_index),
        "schema_atom_index": (
            int(branch.schema_atom_index)
            if branch.schema_atom_index is not None
            else None
        ),
        "cue_conditioned_scene_energy": float(bridge_energy.detach().cpu()),
        "top_scene_index": int(top_scene_idx.detach().cpu()),
        "top_scene_score": float(top_scene_score.detach().cpu()),
        "raw_scene_energy_step3": float(branch.energy_unbiased_step3),
        "raw_scene_energy": float(branch.energy_unbiased),
        "raw_scene_energy_saturated": abs(float(branch.energy_unbiased_step3) + 1.0)
        <= 1e-7,
        "bridge_energy_finite": math.isfinite(float(bridge_energy.detach().cpu())),
        "converged": bool(branch.converged),
        "structural_match": float(branch.structural_match),
    }


def _run_probe_cell(
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
    branches = [
        _branch_bridge_summary(branch=branch, cue=cue, memory=memory)
        for branch in result.branches
    ]
    bridge_values = [row["cue_conditioned_scene_energy"] for row in branches]
    raw_values = [row["raw_scene_energy_step3"] for row in branches]
    return {
        **metadata,
        "cue_bindings_shape": list(cue_bindings.shape),
        "retrieval_executed": True,
        "branch_count": len(branches),
        "schema_indices": [row["schema_index"] for row in branches],
        "min_bridge_energy": min(bridge_values) if bridge_values else None,
        "min_raw_scene_energy_step3": min(raw_values) if raw_values else None,
        "all_bridge_energies_finite": all(row["bridge_energy_finite"] for row in branches),
        "all_raw_scene_energies_saturated": all(
            row["raw_scene_energy_saturated"] for row in branches
        ),
        "branches": branches,
    }


def _paired_delta_summary(cell_results: Dict[str, dict]) -> dict:
    by_group: Dict[str, Dict[str, dict]] = {}
    for cell_result in cell_results.values():
        group = cell_result["paired_delta_group"]
        prior_type = cell_result["prior_type"]
        if group is None or prior_type not in {"content", "role"}:
            continue
        by_group.setdefault(group, {})[prior_type] = cell_result

    out = {}
    for group, pair in sorted(by_group.items()):
        if {"content", "role"} - set(pair):
            out[group] = {
                "delta_e_computable": False,
                "reason": "missing content or role cell",
            }
            continue
        rows = []
        for content_probe, role_probe in zip(
            pair["content"]["probes"],
            pair["role"]["probes"],
        ):
            c_energy = content_probe["min_bridge_energy"]
            r_energy = role_probe["min_bridge_energy"]
            delta = None
            if c_energy is not None and r_energy is not None:
                delta = float(
                    delta_e_content_minus_role(
                        torch.tensor(c_energy),
                        torch.tensor(r_energy),
                    )
                )
            rows.append({
                "probe_index": int(content_probe["probe_index"]),
                "content_min_bridge_energy": c_energy,
                "role_min_bridge_energy": r_energy,
                "delta_e_content_minus_role": delta,
                "delta_e_computable": delta is not None and math.isfinite(delta),
            })
        values = [
            row["delta_e_content_minus_role"]
            for row in rows
            if row["delta_e_content_minus_role"] is not None
        ]
        out[group] = {
            "delta_e_computable": all(row["delta_e_computable"] for row in rows),
            "delta_e_values": values,
            "mean_delta_e": sum(values) / len(values) if values else None,
            "all_zero": all(abs(value) <= 1e-9 for value in values),
            "per_probe": rows,
        }
    return out


def run_cue_conditioned_bridge_smoke(
    *,
    seed: int,
    source: dict,
    source_path: Path,
    source_sha: str,
    cleanup_preflight_path: Optional[Path],
    cleanup_preflight_sha: Optional[str],
    prior_gate_path: Optional[Path],
    prior_gate_sha: Optional[str],
    readout_precommit_path: Optional[Path],
    readout_precommit_sha: Optional[str],
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
        raise ValueError("cue-conditioned smoke is fixed to seed 17")
    if max_probes <= 0 or max_probes > 4:
        raise ValueError("cue-conditioned smoke is fixed to at most four probes")
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
            _run_probe_cell(
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
            "readout_id": READOUT_ID_CUE_CONDITIONED_SCENE,
            "retrieval_executed": True,
            "n_probe_cues": n_probe_cues,
            "probes": probes,
        }

    paired_delta = _paired_delta_summary(cell_results)
    all_probes = [
        probe
        for cell_result in cell_results.values()
        for probe in cell_result["probes"]
    ]
    all_branch_energies = [
        branch["cue_conditioned_scene_energy"]
        for probe in all_probes
        for branch in probe["branches"]
    ]
    raw_branch_energies = [
        branch["raw_scene_energy_step3"]
        for probe in all_probes
        for branch in probe["branches"]
    ]
    pass_criteria = {
        "readout_id_fixed": READOUT_ID_CUE_CONDITIONED_SCENE
        == "cue_conditioned_scene_energy_v1",
        "retrieval_executed": True,
        "seed17_only": int(seed) == 17,
        "pilot_probe_subset_only": n_probe_cues == max_probes and n_probe_cues <= 4,
        "planned_cell_count_is_nine": len(cells) == 9,
        "all_planned_cells_executed": set(cell_results) == {cell["cell_id"] for cell in cells},
        "same_query_subset_all_cells": all(
            [probe["probe_index"] for probe in cell_result["probes"]]
            == list(range(n_probe_cues))
            for cell_result in cell_results.values()
        ),
        "score_bias_none_all_cells": all(
            cell_result["score_bias"] is None for cell_result in cell_results.values()
        ),
        "all_cells_returned_branches": all(probe["branch_count"] > 0 for probe in all_probes),
        "all_bridge_energies_finite": all(
            probe["all_bridge_energies_finite"] for probe in all_probes
        ),
        "bridge_readout_not_all_negative_one": any(
            abs(value + 1.0) > 1e-7 for value in all_branch_energies
        ),
        "raw_scene_energy_still_recorded": len(raw_branch_energies) == len(all_branch_energies),
        "main_delta_e_computable": paired_delta.get("main", {}).get(
            "delta_e_computable", False
        ),
        "k1_delta_e_computable": paired_delta.get("k1", {}).get(
            "delta_e_computable", False
        ),
        "no_prior_delta_e_computable": paired_delta.get("no_prior", {}).get(
            "delta_e_computable", False
        ),
        "no_schema_store_delta_e_computable": paired_delta.get(
            "no_schema_store", {}
        ).get("delta_e_computable", False),
        "random_schema_cell_executed": any(
            cell_result["control_family"] == "random_schema"
            for cell_result in cell_results.values()
        ),
        "legacy_mismatch_accepted_for_smoke_only": bool(
            _config_mismatches(beta=beta, gamma=gamma, k_main=k_main)
        ),
        "no_n3_or_n10_claim": True,
        "no_full_matrix_claim": True,
        "not_headline_verification": True,
        "not_graduation": True,
    }
    return {
        "framing": {
            "phase": "Phase 5 prime cue-conditioned bridge retrieval smoke",
            "readout_id": READOUT_ID_CUE_CONDITIONED_SCENE,
            "retrieval_executed": True,
            "pilot_smoke_only": True,
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
            "source_artifact_path": str(source_path),
            "source_artifact_sha256": source_sha,
            "cleanup_preflight_path": (
                str(cleanup_preflight_path) if cleanup_preflight_path else None
            ),
            "cleanup_preflight_sha256": cleanup_preflight_sha,
            "prior_gate_path": str(prior_gate_path) if prior_gate_path else None,
            "prior_gate_sha256": prior_gate_sha,
            "readout_precommit_path": (
                str(readout_precommit_path) if readout_precommit_path else None
            ),
            "readout_precommit_sha256": readout_precommit_sha,
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
            "include_surprise_branch": False,
            "run_combiners": False,
            "planned_cell_count": len(cells),
        },
        "readout": {
            "readout_id": READOUT_ID_CUE_CONDITIONED_SCENE,
            "formula": "E_bridge(q_scene_final, cue) = -Re(<q_scene_final, cue>) / D",
            "condition_energy_summary": "minimum branch bridge energy within the fixed condition",
            "delta_e": "Delta E = E_content-prior - E_role-prior",
        },
        "legacy_headline_default_mismatches": _config_mismatches(
            beta=beta,
            gamma=gamma,
            k_main=k_main,
        ),
        "interface_shapes": {
            "scene_matrix": list(state.scene_matrix.shape),
            "scene_schema_bindings": list(schema_sources["scene_store"]["schema_bindings"].shape),
            "content_codebook": list(state.content.shape),
            "content_schema_bindings": list(
                schema_sources["content_codebook"]["schema_bindings"].shape
            ),
            "memory_stored_count": memory.stored_count,
        },
        "aggregate": {
            "bridge_energy_min": min(all_branch_energies),
            "bridge_energy_max": max(all_branch_energies),
            "raw_scene_energy_min": min(raw_branch_energies),
            "raw_scene_energy_max": max(raw_branch_energies),
            "total_branches": len(all_branch_energies),
        },
        "cell_results": cell_results,
        "paired_delta": paired_delta,
        "pass_criteria": pass_criteria,
        "passes_all_criteria": all(bool(value) for value in pass_criteria.values()),
        "interpretation": {
            "production_headline_ready": False,
            "major_test_run": False,
            "summary": "Pilot seed-17 cue-conditioned retrieval smoke only; results may debug the bridge readout but cannot verify Phase 5.",
        },
        "stop_conditions_for_followup": [
            "Stop if any required control cannot share the same fixed query set.",
            "Stop if no-schema-store requires changing the headline metric.",
            "Stop if cue-conditioned energy needs target labels or branch-prior energy.",
            "Stop if beta/gamma/K defaults drift without an explicit precommit.",
            "Do not interpret this artifact as n=3/n=10 evidence or Phase 5 graduation.",
        ],
        "anti_homunculus_check": {
            "pass": True,
            "reason": "Fixed cue-scene compatibility readout with passive diagnostics only; no metric-triggered route choice or best-of-N condition selection is introduced.",
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
        "--readout-precommit",
        default="reports/phase5_prime_bridge_readout_precommit.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_cue_conditioned_bridge_smoke_seed17.json",
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
    readout_precommit_path = Path(args.readout_precommit)
    source = _load_json(source_path)
    cleanup_preflight = _load_json(cleanup_path)
    source_sha = _sha256(source_path)
    cleanup_sha = _sha256(cleanup_path)
    prior_gate_sha = _sha256(prior_gate_path)
    readout_precommit_sha = _sha256(readout_precommit_path)
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
    payload = run_cue_conditioned_bridge_smoke(
        seed=int(args.seed),
        source=run_source,
        source_path=source_path,
        source_sha=source_sha,
        cleanup_preflight_path=cleanup_path,
        cleanup_preflight_sha=cleanup_sha,
        prior_gate_path=prior_gate_path,
        prior_gate_sha=prior_gate_sha,
        readout_precommit_path=readout_precommit_path,
        readout_precommit_sha=readout_precommit_sha,
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
        f"readout={payload['readout']['readout_id']} "
        f"n_probe_cues={payload['config']['n_probe_cues']} device={device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
