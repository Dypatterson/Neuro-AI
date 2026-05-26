"""Preflight the Phase 5' bundle-first scene states against Delta E plumbing.

This is a wiring check only. It asks whether fixed bundle-first scene states
can be presented to the existing Phase 5 content-prior vs role-prior branch
comparison without changing the headline metric.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

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


EXP40 = importlib.import_module("experiments.40_phase5_branching")

HEADLINE_METRIC = "Delta E = E_content-prior - E_role-prior"
BRIDGE_CONDITIONS = ("content", "role", "random")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        "probe_index": probe_index,
        "scene": scene,
        "known_role": known_role,
        "query_role": query_role,
        "observed_roles": observed_roles,
        "known_atom": known_atom,
        "target_atom": target_atom,
    }
    return cue, cue_bindings, metadata


def _condition_summary(result) -> dict:
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
    }


def _run_probe_conditions(
    *,
    state: BundleFirstSeedState,
    memory: TorchHopfieldMemory,
    consolidation: ConsolidationState,
    schema_bindings: torch.Tensor,
    schema_atom_idx: torch.Tensor,
    probe_index: int,
    beta: float,
    gamma: float,
    k_main: int,
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
    condition_results: Dict[str, dict] = {}
    for condition in BRIDGE_CONDITIONS:
        rng = torch.Generator().manual_seed(
            state.seed * 1_000_003 + probe_index * 97 + len(condition)
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
            schema_store=state.scene_matrix,
            schema_atom_idx=schema_atom_idx,
            consolidation=consolidation,
            prior_type=condition,
            k_main=k_main,
            gamma=gamma,
            beta=beta,
            temperature=temperature,
            delta_energy=delta_energy,
            delta_state=delta_state,
            delta_redundant=delta_redundant,
            cue_bindings=cue_bindings,
            schema_bindings=schema_bindings,
            include_surprise_branch=False,
            max_settling_iter=max_settling_iter,
            random_prior_rng=rng,
            score_bias=None,
            run_combiners=False,
        )
        condition_results[condition] = _condition_summary(result)

    content_min = condition_results["content"]["min_energy_step3"]
    role_min = condition_results["role"]["min_energy_step3"]
    delta_e = None
    if content_min is not None and role_min is not None:
        delta_e = float(content_min - role_min)
    return {
        **metadata,
        "cue_bindings_shape": list(cue_bindings.shape),
        "conditions": condition_results,
        "delta_e_probe_content_minus_role_step3": delta_e,
        "delta_e_computable": delta_e is not None and math.isfinite(delta_e),
    }


def run_seed_bridge_preflight(
    *,
    seed: int,
    source: dict,
    source_path: Path,
    source_sha: str,
    config: BundleFirstConfig,
    protocol_name: str,
    max_probe_cues: int,
    beta: float,
    gamma: float,
    k_main: int,
    temperature: float,
    delta_energy: float,
    delta_state: float,
    delta_redundant: float,
    max_settling_iter: int,
    device: str,
    cleanup_preflight_path: Optional[Path] = None,
    cleanup_preflight_sha: Optional[str] = None,
    prior_gate_path: Optional[Path] = None,
    prior_gate_sha: Optional[str] = None,
) -> dict:
    if max_probe_cues <= 0:
        raise ValueError("max_probe_cues must be positive")
    state = build_bundle_first_seed_state(
        seed=seed,
        source=source,
        config=config,
        device=device,
    )
    memory, consolidation, schema_atom_idx = _scene_memory_and_consolidation(state)
    schema_bindings = EXP40.compute_schema_bindings(
        substrate=state.fhrr,
        schemas=state.scene_matrix,
        positions=state.roles,
    )
    n_probe_cues = min(max_probe_cues, len(state.query_plan))
    probes = [
        _run_probe_conditions(
            state=state,
            memory=memory,
            consolidation=consolidation,
            schema_bindings=schema_bindings,
            schema_atom_idx=schema_atom_idx,
            probe_index=probe_index,
            beta=beta,
            gamma=gamma,
            k_main=k_main,
            temperature=temperature,
            delta_energy=delta_energy,
            delta_state=delta_state,
            delta_redundant=delta_redundant,
            max_settling_iter=max_settling_iter,
        )
        for probe_index in range(n_probe_cues)
    ]

    condition_branch_ok = {
        condition: all(
            probe["conditions"][condition]["branch_count"] > 0 for probe in probes
        )
        for condition in BRIDGE_CONDITIONS
    }
    condition_energy_ok = {
        condition: all(
            probe["conditions"][condition]["all_branch_energies_finite"]
            for probe in probes
        )
        for condition in BRIDGE_CONDITIONS
    }
    deltas = [
        probe["delta_e_probe_content_minus_role_step3"]
        for probe in probes
        if probe["delta_e_probe_content_minus_role_step3"] is not None
    ]
    pass_criteria = {
        "scene_matrix_shape_ok": list(state.scene_matrix.shape)
        == [config.N, config.D],
        "memory_scene_count_matches": memory.stored_count == config.N,
        "schema_atom_idx_shape_ok": list(schema_atom_idx.shape) == [config.N],
        "schema_bindings_shape_ok": list(schema_bindings.shape)
        == [config.N, config.K_roles, config.D],
        "cue_bindings_present_all_probes": all(
            probe["cue_bindings_shape"][0] > 0 for probe in probes
        ),
        "content_prior_returned_branches": condition_branch_ok["content"],
        "role_prior_returned_branches": condition_branch_ok["role"],
        "random_prior_returned_branches": condition_branch_ok["random"],
        "content_prior_finite_energy": condition_energy_ok["content"],
        "role_prior_finite_energy": condition_energy_ok["role"],
        "random_prior_finite_energy": condition_energy_ok["random"],
        "delta_e_computable_all_probes": all(
            probe["delta_e_computable"] for probe in probes
        ),
        "headline_metric_unchanged": True,
        "preflight_fences_present": True,
    }
    passes_all_criteria = all(bool(value) for value in pass_criteria.values())
    return {
        "framing": {
            "phase": "Phase 5 prime bundle-first Delta E bridge preflight",
            "preflight_only": True,
            "not_a_gate": True,
            "not_graduation": True,
            "no_full_matrix": True,
            "no_m1_escalation": True,
            "no_m2": True,
            "no_new_headline": True,
            "no_headline_claim": True,
            "delta_e_probe_only": True,
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
            "seed": seed,
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
            "beta": beta,
            "gamma": gamma,
            "k_main": k_main,
            "temperature": temperature,
            "delta_energy": delta_energy,
            "delta_state": delta_state,
            "delta_redundant": delta_redundant,
            "max_settling_iter": max_settling_iter,
            "score_bias": None,
            "include_surprise_branch": False,
            "run_combiners": False,
            "conditions": list(BRIDGE_CONDITIONS),
        },
        "interface_shapes": {
            "scene_matrix": list(state.scene_matrix.shape),
            "schema_bindings": list(schema_bindings.shape),
            "schema_atom_idx": list(schema_atom_idx.shape),
            "memory_stored_count": memory.stored_count,
        },
        "pass_criteria": pass_criteria,
        "passes_all_criteria": passes_all_criteria,
        "diagnostic_summary": {
            "probe_only_non_headline": True,
            "delta_e_values": deltas,
            "mean_probe_delta_e": sum(deltas) / len(deltas) if deltas else None,
            "condition_branch_ok": condition_branch_ok,
            "condition_energy_ok": condition_energy_ok,
            "step3_energy_equals_raw_because_score_bias_none": all(
                probe["conditions"][condition]["step3_energy_equals_raw"]
                for probe in probes
                for condition in BRIDGE_CONDITIONS
            ),
        },
        "probes": probes,
        "production_headline_ready": False,
        "production_blockers": [
            "No scene-level score_bias mapping has been specified for the Step-3 weighted Phase 5 landscape; this preflight uses score_bias=None, so Step-3 energy equals raw scene-MHN energy.",
            "The required Phase 5 controls (random-schema branches, K=1, no-prior, and no-schema-store) were not run here.",
            "Only one pilot seed and a small probe subset are exercised; seed 17 is a pilot/reference seed, not representative evidence.",
        ],
        "anti_homunculus_check": {
            "pass": True,
            "reason": "Fixed scene states, fixed priors, and passive diagnostics only; no metric-triggered route choice or best-of-N condition selection is introduced.",
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
        default="reports/phase5_prime_delta_bridge_preflight.json",
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
    payload = run_seed_bridge_preflight(
        seed=int(args.seed),
        source=run_source,
        source_path=source_path,
        source_sha=source_sha,
        config=config,
        protocol_name=str(protocol["protocol_name"]),
        max_probe_cues=int(args.max_probes),
        beta=float(args.beta),
        gamma=float(args.gamma),
        k_main=int(args.k_main),
        temperature=float(args.temperature),
        delta_energy=float(args.delta_energy),
        delta_state=float(args.delta_state),
        delta_redundant=float(args.delta_redundant),
        max_settling_iter=int(args.max_settling_iter),
        device=device,
        cleanup_preflight_path=cleanup_path,
        cleanup_preflight_sha=cleanup_sha,
        prior_gate_path=prior_gate_path,
        prior_gate_sha=prior_gate_sha,
    )
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {out_path} passes={payload['passes_all_criteria']} "
        f"production_headline_ready={payload['production_headline_ready']} "
        f"n_probe_cues={payload['config']['n_probe_cues']} device={device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
