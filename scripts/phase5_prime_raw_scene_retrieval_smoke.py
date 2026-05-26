"""Pilot retrieval smoke for the raw-scene-energy Phase 5' bridge.

This is a narrow seed-17 smoke over the nine cells precommitted by Report 102.
It executes retrieval, so it is not a preflight artifact, but it is still not a
gate, n=3/n=10 evidence, a full matrix, or a Phase 5 graduation claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase4.consolidation import ConsolidationConfig, ConsolidationState
from energy_memory.phase5.bundle_first_scene_memory import (
    BundleFirstConfig,
    BundleFirstSeedState,
    build_bundle_first_seed_state,
)
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


EXP40 = importlib.import_module("experiments.40_phase5_branching")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_seed(*parts: object) -> int:
    data = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % (2**63)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _scene_memory_and_consolidation(
    state: BundleFirstSeedState,
) -> tuple[TorchHopfieldMemory, ConsolidationState, torch.Tensor]:
    memory = TorchHopfieldMemory(state.fhrr)
    consolidation = ConsolidationState(
        ConsolidationConfig(m=4, alpha=0.25),
        device=str(state.fhrr.device),
    )
    for idx, scene_state in enumerate(state.scene_matrix):
        memory.store(scene_state, label=idx)
        consolidation.add_pattern(novelty_strength=1.0)
    schema_atom_idx = torch.arange(
        state.scene_matrix.shape[0],
        dtype=torch.long,
        device=state.fhrr.device,
    )
    return memory, consolidation, schema_atom_idx


def _probe_cue_and_bindings(
    *,
    state: BundleFirstSeedState,
    probe_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    item = state.query_plan[probe_index]
    scene = int(item["scene"])
    known_role = int(item["known_role"])
    query_role = int(item["query_role"])
    observed_roles = [int(role) for role in item["observed_roles"]]
    if not observed_roles:
        raise ValueError(f"probe {probe_index} has no observed roles")

    known_atom = int(state.rows[scene, known_role].detach().cpu())
    target_atom = int(state.rows[scene, query_role].detach().cpu())
    cue = state.roles[known_role] * state.content[known_atom]
    cue = state.fhrr.normalize(
        cue + state.scene_token_weight * state.query_context_tokens[probe_index]
    )
    cue_bindings = torch.stack(
        [
            state.content[int(state.rows[scene, role].detach().cpu())]
            for role in observed_roles
        ],
        dim=0,
    )
    metadata = {
        "probe_index": int(probe_index),
        "scene": scene,
        "known_role": known_role,
        "query_role": query_role,
        "observed_roles": observed_roles,
        "known_atom": known_atom,
        "target_atom": target_atom,
    }
    return cue, cue_bindings, metadata


def _branch_summary(branch) -> dict:
    return {
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
        "energy_drop": float(branch.energy_drop),
        "prior_alignment": float(branch.prior_alignment),
        "score_entropy_initial": float(branch.score_entropy_initial),
        "score_entropy_final": float(branch.score_entropy_final),
        "entropy_collapse": float(branch.entropy_collapse),
        "final_state_divergence": float(branch.final_state_divergence),
        "recall_support": bool(branch.recall_support),
        "cap_coverage_t05": float(branch.cap_coverage_t05),
        "meta_stable": bool(branch.meta_stable),
        "structural_match": float(branch.structural_match),
        "converged": bool(branch.converged),
    }


def _result_summary(result) -> dict:
    step3 = [float(branch.energy_unbiased_step3) for branch in result.branches]
    raw = [float(branch.energy_unbiased) for branch in result.branches]
    finite = all(math.isfinite(value) for value in step3 + raw)
    step3_equals_raw = all(
        abs(step3_value - raw_value) <= 1e-7
        for step3_value, raw_value in zip(step3, raw)
    )
    return {
        "branch_count": len(result.branches),
        "schema_indices": [int(branch.schema_index) for branch in result.branches],
        "schema_atom_indices": [
            int(branch.schema_atom_index)
            if branch.schema_atom_index is not None
            else None
            for branch in result.branches
        ],
        "min_energy_step3": min(step3) if step3 else None,
        "min_energy_raw": min(raw) if raw else None,
        "all_branch_energies_finite": finite,
        "step3_energy_equals_raw": step3_equals_raw,
        "softmax_entropy": float(result.softmax_entropy),
        "split_eligible": bool(result.split_eligible),
        "n_in_low_energy_set": int(result.n_in_low_energy_set),
        "max_state_distance_in_low_energy_set": float(
            result.max_state_distance_in_low_energy_set
        ),
        "branches": [_branch_summary(branch) for branch in result.branches],
    }


def _schema_sources_for_smoke(
    state: BundleFirstSeedState,
    scene_schema_atom_idx: torch.Tensor,
) -> dict:
    scene_bindings = EXP40.compute_schema_bindings(
        substrate=state.fhrr,
        schemas=state.scene_matrix,
        positions=state.roles,
    )
    content_bindings = EXP40.compute_schema_bindings(
        substrate=state.fhrr,
        schemas=state.content,
        positions=state.roles,
    )
    return {
        "scene_store": {
            "schema_store": state.scene_matrix,
            "schema_atom_idx": scene_schema_atom_idx,
            "schema_bindings": scene_bindings,
        },
        "content_codebook": {
            "schema_store": state.content,
            "schema_atom_idx": None,
            "schema_bindings": content_bindings,
        },
    }


def _run_probe_cell(
    *,
    state: BundleFirstSeedState,
    memory: TorchHopfieldMemory,
    consolidation: ConsolidationState,
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
    random_prior_rng = torch.Generator().manual_seed(
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
        random_prior_rng=random_prior_rng,
        score_bias=None,
        run_combiners=False,
    )
    summary = _result_summary(result)
    return {
        **metadata,
        "cue_bindings_shape": list(cue_bindings.shape),
        "retrieval_executed": True,
        "result": summary,
    }


def _paired_delta_summary(cell_results: Dict[str, dict]) -> dict:
    by_group: Dict[str, Dict[str, dict]] = {}
    for cell_id, cell_result in cell_results.items():
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
        content = pair["content"]["probes"]
        role = pair["role"]["probes"]
        deltas = []
        for content_probe, role_probe in zip(content, role):
            c_energy = content_probe["result"]["min_energy_step3"]
            r_energy = role_probe["result"]["min_energy_step3"]
            delta = None
            if c_energy is not None and r_energy is not None:
                delta = float(c_energy - r_energy)
            deltas.append({
                "probe_index": int(content_probe["probe_index"]),
                "content_min_energy_step3": c_energy,
                "role_min_energy_step3": r_energy,
                "delta_e_content_minus_role_step3": delta,
                "delta_e_computable": delta is not None
                and math.isfinite(float(delta)),
            })
        values = [
            row["delta_e_content_minus_role_step3"]
            for row in deltas
            if row["delta_e_content_minus_role_step3"] is not None
        ]
        out[group] = {
            "delta_e_computable": all(row["delta_e_computable"] for row in deltas),
            "delta_e_values": values,
            "mean_delta_e": sum(values) / len(values) if values else None,
            "per_probe": deltas,
        }
    return out


def run_raw_scene_retrieval_smoke(
    *,
    seed: int,
    source: dict,
    source_path: Path,
    source_sha: str,
    cleanup_preflight_path: Optional[Path],
    cleanup_preflight_sha: Optional[str],
    prior_gate_path: Optional[Path],
    prior_gate_sha: Optional[str],
    controls_preflight_path: Optional[Path],
    controls_preflight_sha: Optional[str],
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
    if max_probes <= 0:
        raise ValueError("max_probes must be positive")
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

    cell_results: Dict[str, dict] = {}
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
    all_cells_returned_branches = all(
        probe["result"]["branch_count"] > 0 for probe in all_probes
    )
    all_branch_energies_finite = all(
        probe["result"]["all_branch_energies_finite"] for probe in all_probes
    )
    step3_equals_raw = all(
        probe["result"]["step3_energy_equals_raw"] for probe in all_probes
    )
    pass_criteria = {
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
        "all_cells_returned_branches": all_cells_returned_branches,
        "all_branch_energies_finite": all_branch_energies_finite,
        "step3_energy_equals_raw_because_score_bias_none": step3_equals_raw,
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
            "phase": "Phase 5 prime raw-scene bridge retrieval smoke",
            "baseline_id": BASELINE_ID,
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
            "controls_preflight_path": (
                str(controls_preflight_path) if controls_preflight_path else None
            ),
            "controls_preflight_sha256": controls_preflight_sha,
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
        "cell_results": cell_results,
        "paired_delta": paired_delta,
        "pass_criteria": pass_criteria,
        "passes_all_criteria": all(bool(value) for value in pass_criteria.values()),
        "interpretation": {
            "production_headline_ready": False,
            "major_test_run": False,
            "summary": "Pilot seed-17 retrieval plumbing smoke only; results may debug interfaces but cannot verify Phase 5.",
        },
        "stop_conditions_for_followup": [
            "Stop if any required control cannot share the same fixed query set.",
            "Stop if no-schema-store requires changing the headline metric.",
            "Stop if a scene-level Step-3 score_bias is reintroduced without a fixed mapping precommit.",
            "Stop if beta/gamma/K defaults drift without an explicit precommit.",
            "Do not interpret this artifact as n=3/n=10 evidence or Phase 5 graduation.",
        ],
        "anti_homunculus_check": {
            "pass": True,
            "reason": "Fixed scene states, fixed prior selectors, fixed raw scene energy, and passive diagnostics only.",
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
        "--controls-preflight",
        default="reports/phase5_prime_raw_scene_controls_preflight.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_raw_scene_retrieval_smoke_seed17.json",
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
    controls_preflight_path = Path(args.controls_preflight)
    source = _load_json(source_path)
    cleanup_preflight = _load_json(cleanup_path)
    source_sha = _sha256(source_path)
    cleanup_sha = _sha256(cleanup_path)
    prior_gate_sha = _sha256(prior_gate_path)
    controls_preflight_sha = _sha256(controls_preflight_path)
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
    payload = run_raw_scene_retrieval_smoke(
        seed=int(args.seed),
        source=run_source,
        source_path=source_path,
        source_sha=source_sha,
        cleanup_preflight_path=cleanup_path,
        cleanup_preflight_sha=cleanup_sha,
        prior_gate_path=prior_gate_path,
        prior_gate_sha=prior_gate_sha,
        controls_preflight_path=controls_preflight_path,
        controls_preflight_sha=controls_preflight_sha,
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
        f"cells={payload['config']['planned_cell_count']} "
        f"n_probe_cues={payload['config']['n_probe_cues']} device={device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
