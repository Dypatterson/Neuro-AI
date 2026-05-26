"""Fixed candidate/control gate for the Phase 5' non-synthetic native source.

This script runs only after the Report 089 preflight artifact exists. It loads
the fixed repo-sample native provenance source artifact and uses its committed
rows, observed-role selections, and query plans for every condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import torch

from energy_memory.phase5.bundle_first_scene_memory import (
    EXP44,
    BundleFirstConfig,
    BundleFirstResult as GateResult,
    TorchFHRR,
    aggregate_bundle_first_results,
    build_native_roles,
    build_query_context_tokens,
    build_scene_matrix,
    run_bundle_first_seed_condition,
    wilson_ci,
)

SOURCE_NAME = "trajectory_native_provenance_context_trace"
SOURCE_FAMILY = "trajectory_derived_native"
SOURCE_KIND = "repo_sample_phase2_window_trace"
COOCCURRENCE = "repo_sample_natural"
CONDITIONS = {
    "candidate",
    "random_role",
    "deranged_role",
    "shuffled_role",
    "content_cleanup_positive",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _wilson(n_success: int, n_total: int, z: float = 1.96) -> dict:
    return wilson_ci(n_success, n_total, z=z)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _validate_source_artifact(source: dict, args: argparse.Namespace) -> None:
    framing = source.get("framing")
    if not isinstance(framing, dict):
        raise ValueError("source artifact missing framing")
    if framing.get("source_name") != SOURCE_NAME:
        raise ValueError(f"source artifact is not {SOURCE_NAME}")
    if framing.get("source_family") != SOURCE_FAMILY:
        raise ValueError(f"source artifact is not {SOURCE_FAMILY}")
    if framing.get("source_kind") != SOURCE_KIND:
        raise ValueError(f"source artifact is not {SOURCE_KIND}")

    config = source.get("config")
    if not isinstance(config, dict):
        raise ValueError("source artifact missing config")
    checks = {
        "D": args.D,
        "N": args.N,
        "K_roles": args.K_roles,
        "C_codebook": args.C_codebook,
        "context_roles": args.context_roles,
        "cooccurrence": args.cooccurrence,
        "n_queries": args.n_queries,
        "seeds": args.seeds,
    }
    for key, expected in checks.items():
        if config.get(key) != expected:
            raise ValueError(
                f"source artifact config mismatch for {key}: "
                f"{config.get(key)!r} != {expected!r}"
            )

    rows_by_seed = source.get("source_rows_by_seed")
    query_plan_by_seed = source.get("query_plan_by_seed")
    if not isinstance(rows_by_seed, dict):
        raise ValueError("source artifact missing source_rows_by_seed")
    if not isinstance(query_plan_by_seed, dict):
        raise ValueError("source artifact missing query_plan_by_seed")

    for seed in args.seeds:
        seed_key = str(seed)
        rows = rows_by_seed.get(seed_key)
        query_plan = query_plan_by_seed.get(seed_key)
        if not isinstance(rows, list) or len(rows) != args.N:
            raise ValueError(f"source rows for seed {seed} are missing or not N")
        if not isinstance(query_plan, list) or len(query_plan) != args.n_queries:
            raise ValueError(f"query plan for seed {seed} is missing or not n_queries")
        for row in rows:
            if not isinstance(row, list) or len(row) != args.K_roles:
                raise ValueError(f"source row for seed {seed} is not K_roles long")
            if any(int(atom) < 0 or int(atom) >= args.C_codebook for atom in row):
                raise ValueError(f"source row for seed {seed} has invalid atoms")
        for item in query_plan:
            scene = int(item["scene"])
            known_role = int(item["known_role"])
            query_role = int(item["query_role"])
            observed_roles = [int(role) for role in item["observed_roles"]]
            if scene < 0 or scene >= args.N:
                raise ValueError(f"query plan for seed {seed} has invalid scene")
            if not (0 <= known_role < args.K_roles and 0 <= query_role < args.K_roles):
                raise ValueError(f"query plan for seed {seed} has invalid role")
            if known_role not in observed_roles:
                raise ValueError(f"query plan for seed {seed} omits known_role")
            if query_role in observed_roles:
                raise ValueError(f"query plan for seed {seed} leaks query_role")


def _validate_preflight(preflight: dict, source_path: Path, source_sha: str) -> None:
    expected_sha = preflight.get("source_manifest", {}).get("source_artifact_sha256")
    if expected_sha != source_sha:
        raise ValueError(
            f"source SHA mismatch: preflight={expected_sha} artifact={source_sha}"
        )
    expected_path = preflight.get("source_manifest", {}).get("source_artifact_path")
    if expected_path and Path(expected_path) != source_path:
        raise ValueError(
            f"source path mismatch: preflight={expected_path} artifact={source_path}"
        )
    if preflight.get("framing", {}).get("source_name") != SOURCE_NAME:
        raise ValueError(f"preflight source is not {SOURCE_NAME}")
    if not all(preflight.get("pass_criteria", {}).values()):
        raise ValueError("preflight pass_criteria are not all true")


def _native_roles(fhrr: TorchFHRR, K_roles: int) -> torch.Tensor:
    return build_native_roles(fhrr, K_roles)


def _scene_bundles(
    fhrr: TorchFHRR,
    roles: torch.Tensor,
    content: torch.Tensor,
    rows: torch.Tensor,
    *,
    scene_token_weight: float,
) -> torch.Tensor:
    return build_scene_matrix(
        fhrr,
        roles,
        content,
        rows,
        scene_token_weight=scene_token_weight,
    )


def _query_context_tokens(
    fhrr: TorchFHRR,
    roles: torch.Tensor,
    content: torch.Tensor,
    rows: torch.Tensor,
    query_plan: list[dict],
) -> torch.Tensor:
    return build_query_context_tokens(fhrr, roles, content, rows, query_plan)


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
) -> GateResult:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    return run_bundle_first_seed_condition(
        condition=condition,
        seed=seed,
        source=source,
        source_path=source_path,
        source_sha=source_sha,
        config=BundleFirstConfig(
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
            source_name=SOURCE_NAME,
        ),
        cue_noise=cue_noise,
        device=device,
    )


def _aggregate(cell: list[GateResult]) -> dict:
    return aggregate_bundle_first_results(cell)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="reports/phase5_prime_nonsynthetic_native_context_source.json",
    )
    parser.add_argument(
        "--preflight",
        default="reports/phase5_prime_nonsynthetic_native_preflight.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_nonsynthetic_native_gate.json",
    )
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--K_roles", type=int, default=16)
    parser.add_argument("--C_codebook", type=int, default=2048)
    parser.add_argument("--context_roles", type=int, default=4)
    parser.add_argument("--cue_noise", type=float, default=0.15)
    parser.add_argument("--token_weight", type=float, default=0.25)
    parser.add_argument("--cooccurrence", choices=[COOCCURRENCE], default=COOCCURRENCE)
    parser.add_argument("--n_queries", type=int, default=512)
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[17, 11, 23, 1, 2, 3, 5, 7, 13, 29],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "candidate",
            "random_role",
            "deranged_role",
            "shuffled_role",
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

    unknown_conditions = sorted(set(args.conditions) - CONDITIONS)
    if unknown_conditions:
        raise ValueError(f"unknown conditions: {unknown_conditions}")
    if args.context_roles <= 0 or args.context_roles >= args.K_roles:
        raise ValueError("context_roles must be between 1 and K_roles - 1")
    if args.max_iter <= 0:
        raise ValueError("max_iter must be positive")

    source_path = Path(args.source)
    preflight_path = Path(args.preflight)
    source = _load_json(source_path)
    preflight = _load_json(preflight_path)
    _validate_source_artifact(source, args)
    source_sha = _sha256(source_path)
    _validate_preflight(preflight, source_path, source_sha)

    print(
        f"device={device} source={source_path} source_sha={source_sha} "
        f"D={args.D} N={args.N} K_roles={args.K_roles} "
        f"context_roles={args.context_roles} n_queries={args.n_queries} "
        f"seeds={args.seeds} conditions={args.conditions}"
    )

    raw: List[GateResult] = []
    aggregates: Dict[str, dict] = {}
    for condition in args.conditions:
        cell: List[GateResult] = []
        for seed in args.seeds:
            result = _run_seed_condition(
                condition=condition,
                seed=seed,
                source=source,
                source_path=source_path,
                source_sha=source_sha,
                D=args.D,
                N=args.N,
                K_roles=args.K_roles,
                C_codebook=args.C_codebook,
                context_roles=args.context_roles,
                cue_noise=args.cue_noise,
                scene_token_weight=args.token_weight,
                cooccurrence=args.cooccurrence,
                n_queries=args.n_queries,
                beta=args.beta,
                max_iter=args.max_iter,
                device=device,
            )
            cell.append(result)
            raw.append(result)
        key = (
            f"{condition}|D={args.D}|K={args.K_roles}|N={args.N}|"
            f"noise={args.cue_noise}|scene_token=1|token_weight={args.token_weight}|"
            f"token_source={SOURCE_NAME}|context_roles={args.context_roles}|"
            f"cooc={args.cooccurrence}"
        )
        agg = _aggregate(cell)
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
            "phase": "5-prime non-synthetic native provenance-source diagnostic",
            "source_name": SOURCE_NAME,
            "source_family": SOURCE_FAMILY,
            "source_kind": SOURCE_KIND,
            "not_graduation": True,
            "no_full_matrix": True,
            "anti_homunculus": (
                "fixed source rows, fixed query plan, fixed controls; no "
                "metric-triggered routing or best-of-N selection"
            ),
        },
        "source_manifest": {
            "source_artifact_path": str(source_path),
            "source_artifact_sha256": source_sha,
            "preflight_artifact_path": str(preflight_path),
        },
        "preflight_summary": {
            "pass_criteria": preflight.get("pass_criteria", {}),
            "source_observed_context_to_full_context": preflight.get(
                "aggregate", {}
            ).get("source_observed_context_to_full_context", {}),
            "support": preflight.get("aggregate", {}).get("support", {}),
        },
        "config": {
            "D": args.D,
            "N": args.N,
            "K_roles": args.K_roles,
            "C_codebook": args.C_codebook,
            "context_roles": args.context_roles,
            "cue_noise": args.cue_noise,
            "token_weight": args.token_weight,
            "cooccurrence": args.cooccurrence,
            "n_queries": args.n_queries,
            "beta": args.beta,
            "max_iter": args.max_iter,
            "device": device,
            "seeds": args.seeds,
            "conditions": args.conditions,
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
