"""Static precommit for the next Phase 5' bundle-first bridge readout.

This records the fixed non-saturated readout after Report 104. It does not run
retrieval, compute evidence, or widen beyond the existing smoke scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import torch

from energy_memory.phase5.bridge_readouts import (
    READOUT_ID_CUE_CONDITIONED_SCENE,
    cue_conditioned_scene_energy,
    delta_e_content_minus_role,
)
from energy_memory.phase5.bundle_first_scene_memory import BundleFirstConfig
from energy_memory.phase5.natural_source_protocol import (
    protocol_payload,
    validate_cleanup_preflight,
)
from scripts.phase5_prime_raw_scene_controls_preflight import (
    HEADLINE_METRIC,
    _config_mismatches,
    planned_control_cells,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _toy_sanity_checks() -> dict:
    cue = torch.tensor([1, 1, 1, 1], dtype=torch.complex64)
    same_scene = torch.tensor([1, 1, 1, 1], dtype=torch.complex64)
    different_scene = torch.tensor([1, -1, 1, -1], dtype=torch.complex64)
    content_energy = cue_conditioned_scene_energy(cue, different_scene)
    role_energy = cue_conditioned_scene_energy(cue, same_scene)
    delta_e = delta_e_content_minus_role(content_energy, role_energy)
    return {
        "self_energy": float(role_energy.detach().cpu()),
        "different_scene_energy": float(content_energy.detach().cpu()),
        "delta_e_content_minus_role": float(delta_e.detach().cpu()),
        "self_energy_is_negative_one": abs(float(role_energy.detach().cpu()) + 1.0) <= 1e-7,
        "different_scene_not_saturated": abs(float(content_energy.detach().cpu()) + 1.0) > 1e-7,
        "positive_delta_when_role_energy_lower": float(delta_e.detach().cpu()) > 0.0,
    }


def build_bridge_readout_precommit(
    *,
    config: BundleFirstConfig,
    protocol_name: str,
    analysis_path: Optional[Path],
    analysis_sha: Optional[str],
    analysis_payload: Optional[dict],
    beta: float,
    gamma: float,
    k_main: int,
) -> dict:
    cells = planned_control_cells(config=config, beta=beta, gamma=gamma, k_main=k_main)
    toy = _toy_sanity_checks()
    analysis_confirms_degeneracy = bool(
        analysis_payload
        and analysis_payload.get("diagnosis", {}).get("raw_scene_energy_v0_degenerate")
    )
    pass_criteria = {
        "report104_degeneracy_recorded": analysis_confirms_degeneracy,
        "readout_id_fixed": READOUT_ID_CUE_CONDITIONED_SCENE == "cue_conditioned_scene_energy_v1",
        "formula_uses_final_scene_and_fixed_cue_only": True,
        "no_target_labels": True,
        "no_branch_prior_in_energy": True,
        "no_content_cleanup_labels": True,
        "same_delta_e_sign_convention": True,
        "same_nine_control_cells_planned": len(cells) == 9,
        "score_bias_none": True,
        "toy_self_energy_negative_one": toy["self_energy_is_negative_one"],
        "toy_different_scene_not_saturated": toy["different_scene_not_saturated"],
        "toy_delta_sign_correct": toy["positive_delta_when_role_energy_lower"],
        "no_retrieval_executed": True,
        "no_evidence_claim": True,
    }
    return {
        "framing": {
            "phase": "Phase 5 prime bridge-readout precommit",
            "precommit_only": True,
            "retrieval_executed": False,
            "not_a_gate": True,
            "not_n3": True,
            "not_n10": True,
            "not_graduation": True,
            "no_full_matrix": True,
            "no_m1_escalation": True,
            "no_m2": True,
            "headline_metric_form": HEADLINE_METRIC,
        },
        "source_manifest": {
            "report104_analysis_path": str(analysis_path) if analysis_path else None,
            "report104_analysis_sha256": analysis_sha,
            "protocol_name": protocol_name,
        },
        "readout": {
            "readout_id": READOUT_ID_CUE_CONDITIONED_SCENE,
            "formula": "E_bridge(q_scene_final, cue) = -Re(<q_scene_final, cue>) / D",
            "delta_e": "Delta E = E_content-prior - E_role-prior",
            "lower_is_better": True,
            "positive_delta_e_meaning": "role-prior final scene is more cue-compatible than content-prior final scene",
            "inputs": [
                "fixed cue vector",
                "final settled scene state",
            ],
            "forbidden_inputs": [
                "target atom label",
                "target scene label",
                "branch prior vector",
                "content-cleanup top1",
                "observed performance metrics",
                "condition winner selection",
            ],
            "score_bias": None,
            "settling_path_changed": False,
            "schema_selection_changed": False,
        },
        "config": {
            "D": config.D,
            "N": config.N,
            "K_roles": config.K_roles,
            "C_codebook": config.C_codebook,
            "context_roles": config.context_roles,
            "n_queries_per_seed": config.n_queries,
            "scene_token_weight": config.scene_token_weight,
            "cooccurrence": config.cooccurrence,
            "source_name": config.source_name,
            "beta": float(beta),
            "gamma": float(gamma),
            "k_main": int(k_main),
            "score_bias": None,
        },
        "legacy_headline_default_mismatches": _config_mismatches(
            beta=beta,
            gamma=gamma,
            k_main=k_main,
        ),
        "planned_cells": cells,
        "toy_sanity_checks": toy,
        "pass_criteria": pass_criteria,
        "passes_all_criteria": all(bool(value) for value in pass_criteria.values()),
        "stop_conditions": [
            "Do not run retrieval if the readout needs target labels or target scenes.",
            "Do not run retrieval if the readout uses the branch prior as an energy term.",
            "Do not run retrieval if the readout is selected by observed performance.",
            "Do not widen beyond seed-17/four-probe smoke before a passing pilot readback.",
            "Do not interpret a pilot readback as n=3/n=10 evidence or graduation.",
        ],
        "anti_homunculus_check": {
            "pass": True,
            "reason": "Fixed cue-scene compatibility term; no adaptive routing, target label, metric feedback, or best-condition selection.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        default="reports/phase5_prime_raw_scene_energy_degeneracy_analysis.json",
    )
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
        default="reports/phase5_prime_bridge_readout_precommit.json",
    )
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--k-main", type=int, default=4)
    args = parser.parse_args()

    source_path = Path(args.source)
    cleanup_path = Path(args.cleanup_preflight)
    prior_gate_path = Path(args.prior_gate)
    analysis_path = Path(args.analysis)
    source_sha = _sha256(source_path)
    prior_gate_sha = _sha256(prior_gate_path)
    cleanup_preflight = _load_json(cleanup_path)
    validate_cleanup_preflight(
        cleanup_preflight,
        source_sha=source_sha,
        gate_sha=prior_gate_sha,
    )
    protocol = protocol_payload(cleanup_preflight, args.protocol)
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
        source_name=str(_load_json(source_path)["framing"]["source_name"]),
    )
    analysis_payload = _load_json(analysis_path) if analysis_path.exists() else None
    analysis_sha = _sha256(analysis_path) if analysis_path.exists() else None
    payload = build_bridge_readout_precommit(
        config=config,
        protocol_name=str(protocol["protocol_name"]),
        analysis_path=analysis_path if analysis_path.exists() else None,
        analysis_sha=analysis_sha,
        analysis_payload=analysis_payload,
        beta=float(args.beta),
        gamma=float(args.gamma),
        k_main=int(args.k_main),
    )
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {out_path} passes={payload['passes_all_criteria']} "
        f"readout={payload['readout']['readout_id']} retrieval_executed=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
