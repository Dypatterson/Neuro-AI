"""Fixed gate for the Report 092 cleaned natural-source protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from energy_memory.phase5.natural_source_protocol import (
    fixedpoint_free_shuffle as _fixedpoint_free_shuffle_indices,
    protocol_payload as _protocol_payload,
    source_with_protocol_plan as _source_with_protocol_plan,
    validate_cleanup_preflight,
)
from scripts import phase5_prime_nonsynthetic_native_gate as base_gate


CONDITIONS = {
    "candidate",
    "random_role",
    "deranged_role",
    "fixedpoint_free_shuffled_role",
    "content_cleanup_positive",
}


def _fixedpoint_free_shuffle(seed: int, k_roles: int) -> torch.Tensor:
    return torch.tensor(
        _fixedpoint_free_shuffle_indices(seed, k_roles),
        dtype=torch.long,
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _run_fixedpoint_free_shuffled(
    *,
    seed: int,
    source: dict,
    source_path: Path,
    source_sha: str,
    D: int,
    N: int,
    K_roles: int,
    C_codebook: int,
    context_roles: int,
    cue_noise: float,
    scene_token_weight: float,
    cooccurrence: str,
    n_queries: int,
    beta: float,
    max_iter: int,
    device: str,
) -> base_gate.GateResult:
    return base_gate.run_bundle_first_seed_condition(
        condition="fixedpoint_free_shuffled_role",
        seed=seed,
        source=source,
        source_path=source_path,
        source_sha=source_sha,
        config=base_gate.BundleFirstConfig(
            D=D,
            N=N,
            K_roles=K_roles,
            C_codebook=C_codebook,
            context_roles=context_roles,
            n_queries=n_queries,
            beta=beta,
            max_iter=max_iter,
            scene_token_weight=scene_token_weight,
            cooccurrence=cooccurrence,
            source_name=base_gate.SOURCE_NAME,
        ),
        cue_noise=cue_noise,
        device=device,
    )


def _run_seed_condition(
    *,
    condition: str,
    seed: int,
    source: dict,
    source_path: Path,
    source_sha: str,
    D: int,
    N: int,
    K_roles: int,
    C_codebook: int,
    context_roles: int,
    cue_noise: float,
    scene_token_weight: float,
    cooccurrence: str,
    n_queries: int,
    beta: float,
    max_iter: int,
    device: str,
) -> base_gate.GateResult:
    if condition == "fixedpoint_free_shuffled_role":
        return _run_fixedpoint_free_shuffled(
            seed=seed,
            source=source,
            source_path=source_path,
            source_sha=source_sha,
            D=D,
            N=N,
            K_roles=K_roles,
            C_codebook=C_codebook,
            context_roles=context_roles,
            cue_noise=cue_noise,
            scene_token_weight=scene_token_weight,
            cooccurrence=cooccurrence,
            n_queries=n_queries,
            beta=beta,
            max_iter=max_iter,
            device=device,
        )
    return base_gate._run_seed_condition(
        condition=condition,
        seed=seed,
        source=source,
        source_path=source_path,
        source_sha=source_sha,
        D=D,
        N=N,
        K_roles=K_roles,
        C_codebook=C_codebook,
        context_roles=context_roles,
        cue_noise=cue_noise,
        scene_token_weight=scene_token_weight,
        cooccurrence=cooccurrence,
        n_queries=n_queries,
        beta=beta,
        max_iter=max_iter,
        device=device,
    )


def _validate_preflight(preflight: dict, source_sha: str, gate_sha: str) -> None:
    validate_cleanup_preflight(
        preflight,
        source_sha=source_sha,
        gate_sha=gate_sha,
    )


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
        default="reports/phase5_prime_natural_source_control_cleanup_gate.json",
    )
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "candidate",
            "random_role",
            "deranged_role",
            "fixedpoint_free_shuffled_role",
            "content_cleanup_positive",
        ],
    )
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    unknown = sorted(set(args.conditions) - CONDITIONS)
    if unknown:
        raise ValueError(f"unknown cleanup-gate conditions: {unknown}")

    source_path = Path(args.source)
    cleanup_path = Path(args.cleanup_preflight)
    prior_gate_path = Path(args.prior_gate)
    source = _load_json(source_path)
    cleanup_preflight = _load_json(cleanup_path)
    prior_gate = _load_json(prior_gate_path)
    source_sha = base_gate._sha256(source_path)
    prior_gate_sha = base_gate._sha256(prior_gate_path)
    _validate_preflight(cleanup_preflight, source_sha, prior_gate_sha)
    protocol = _protocol_payload(cleanup_preflight, args.protocol)
    run_source = _source_with_protocol_plan(source, protocol)
    config = cleanup_preflight["config"]["source_config"]
    n_queries = int(protocol["required_queries_per_seed"])
    seeds = [int(seed) for seed in config["seeds"]]
    cooccurrence = str(config["cooccurrence"])

    print(
        f"device={device} protocol={protocol['protocol_name']} "
        f"source={source_path} source_sha={source_sha} conditions={args.conditions}"
    )

    raw: List[base_gate.GateResult] = []
    aggregates: Dict[str, dict] = {}
    for condition in args.conditions:
        cell: List[base_gate.GateResult] = []
        for seed in seeds:
            result = _run_seed_condition(
                condition=condition,
                seed=seed,
                source=run_source,
                source_path=source_path,
                source_sha=source_sha,
                D=int(config["D"]),
                N=int(config["N"]),
                K_roles=int(config["K_roles"]),
                C_codebook=int(config["C_codebook"]),
                context_roles=int(config["context_roles"]),
                cue_noise=float(config["cue_noise"]),
                scene_token_weight=float(config["token_weight"]),
                cooccurrence=cooccurrence,
                n_queries=n_queries,
                beta=30.0,
                max_iter=10,
                device=device,
            )
            cell.append(result)
            raw.append(result)
        key = (
            f"{condition}|protocol={protocol['protocol_name']}|"
            f"D={config['D']}|K={config['K_roles']}|N={config['N']}|"
            f"noise={config['cue_noise']}|scene_token=1|"
            f"token_weight={config['token_weight']}|"
            f"token_source={base_gate.SOURCE_NAME}|"
            f"context_roles={config['context_roles']}|cooc={cooccurrence}"
        )
        agg = base_gate._aggregate(cell)
        aggregates[key] = agg
        loo_min = agg["leave_one_seed_out_top1_min"]
        loo_max = agg["leave_one_seed_out_top1_max"]
        loo_text = (
            f" loo=[{loo_min:.4f},{loo_max:.4f}]"
            if loo_min is not None and loo_max is not None
            else ""
        )
        print(
            f"{key} top1={agg['top1_mean']:.4f} "
            f"CI=[{agg['wilson_lo']:.4f},{agg['wilson_hi']:.4f}] "
            f"scene_tix={agg['scene_tix']}/{agg['n_total']} "
            f"content_tix={agg['content_tix']}/{agg['n_total']} "
            f"ent=({agg['mean_scene_entropy']:.3f},"
            f"{agg['mean_content_entropy']:.3f}) "
            f"margin=({agg['mean_scene_margin']:.4f},"
            f"{agg['mean_content_margin']:.4f}){loo_text}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "framing": {
            "phase": "5-prime natural source/control cleanup gate",
            "source_name": base_gate.SOURCE_NAME,
            "source_family": base_gate.SOURCE_FAMILY,
            "source_kind": base_gate.SOURCE_KIND,
            "cleanup_protocol": protocol["protocol_name"],
            "not_graduation": True,
            "no_full_matrix": True,
            "anti_homunculus": (
                "fixed source rows, fixed Report 092 query plan, fixed controls; "
                "no metric-triggered routing or best-of-N selection"
            ),
        },
        "source_manifest": {
            "source_artifact_path": str(source_path),
            "source_artifact_sha256": source_sha,
            "cleanup_preflight_artifact_path": str(cleanup_path),
            "cleanup_preflight_artifact_sha256": base_gate._sha256(cleanup_path),
            "prior_gate_artifact_path": str(prior_gate_path),
            "prior_gate_artifact_sha256": prior_gate_sha,
        },
        "cleanup_protocol_summary": {
            "protocol_name": protocol["protocol_name"],
            "pass_criteria": protocol["pass_criteria"],
            "aggregate": protocol["aggregate"],
        },
        "config": {
            **config,
            "n_queries": n_queries,
            "beta": 30.0,
            "max_iter": 10,
            "device": device,
            "conditions": args.conditions,
            "legacy_shuffled_role_excluded": True,
        },
        "aggregates": aggregates,
        "raw": [
            {
                "condition": r.condition,
                "D": r.D,
                "N": r.N,
                "K_roles": r.K_roles,
                "cue_noise": r.cue_noise,
                "scene_token_weight": r.scene_token_weight,
                "source_name": r.source_name,
                "context_roles": r.context_roles,
                "cooccurrence": r.cooccurrence,
                "seed": r.seed,
                "n_queries": r.n_queries,
                "n_correct": r.n_correct,
                "top1": r.top1,
                "scene_tix": r.scene_tix,
                "content_tix": r.content_tix,
                "scene_entropy": r.scene_entropy,
                "content_entropy": r.content_entropy,
                "scene_margin": r.scene_margin,
                "content_margin": r.content_margin,
                "source_rows_available": r.source_rows_available,
                "source_rows_used": r.source_rows_used,
                "source_rows_invalid": r.source_rows_invalid,
                "source_rows_too_short": r.source_rows_too_short,
                "source_artifact_path": r.source_artifact_path,
                "source_artifact_sha256": r.source_artifact_sha256,
            }
            for r in raw
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
