"""Fixed candidate/control gate for the Phase 5' true provenance source.

This script runs only after the Report 087 preflight artifact exists. It loads
the committed native provenance source artifact and uses its fixed rows,
observed-role selections, and query plans for every condition.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from energy_memory.substrate.torch_fhrr import TorchFHRR


EXP44 = importlib.import_module("experiments.44_phase5_prime_bundle_first")


@dataclass(frozen=True)
class GateResult:
    condition: str
    D: int
    N: int
    K_roles: int
    cue_noise: float
    scene_token_weight: float
    source_name: str
    context_roles: int
    cooccurrence: str
    seed: int
    n_queries: int
    n_correct: int
    scene_tix: int
    content_tix: int
    scene_entropy: float
    content_entropy: float
    scene_margin: float
    content_margin: float
    source_rows_available: int
    source_rows_used: int
    source_rows_invalid: int
    source_rows_too_short: int
    source_artifact_path: str
    source_artifact_sha256: str

    @property
    def top1(self) -> float:
        return self.n_correct / self.n_queries if self.n_queries else 0.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _wilson(n_success: int, n_total: int, z: float = 1.96) -> dict:
    if n_total == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    p = n_success / n_total
    denom = 1.0 + z * z / n_total
    center = (p + z * z / (2 * n_total)) / denom
    half = (
        z
        * math.sqrt(p * (1.0 - p) / n_total + z * z / (4.0 * n_total * n_total))
        / denom
    )
    return {
        "mean": p,
        "lo": max(0.0, center - half),
        "hi": min(1.0, center + half),
        "n": n_total,
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _validate_source_artifact(source: dict, args: argparse.Namespace) -> None:
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
    if source.get("framing", {}).get("source_name") != "native_provenance_context_trace":
        raise ValueError("source artifact is not native_provenance_context_trace")


def _scene_bundles(
    fhrr: TorchFHRR,
    roles: torch.Tensor,
    content: torch.Tensor,
    rows: torch.Tensor,
    *,
    scene_token_weight: float,
) -> torch.Tensor:
    base_bundles = []
    for scene in range(rows.shape[0]):
        terms = [
            roles[role] * content[int(rows[scene, role].detach().cpu())]
            for role in range(rows.shape[1])
        ]
        full_context = fhrr.bundle(terms)
        base_bundles.append(fhrr.bundle([*terms, scene_token_weight * full_context]))
    return torch.stack(base_bundles, dim=0)


def _query_context_tokens(
    fhrr: TorchFHRR,
    roles: torch.Tensor,
    content: torch.Tensor,
    rows: torch.Tensor,
    query_plan: Sequence[dict],
) -> torch.Tensor:
    tokens = []
    for item in query_plan:
        scene = int(item["scene"])
        observed_roles = [int(role) for role in item["observed_roles"]]
        terms = [
            roles[role] * content[int(rows[scene, role].detach().cpu())]
            for role in observed_roles
        ]
        tokens.append(fhrr.bundle(terms))
    return torch.stack(tokens, dim=0)


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
    device: str,
) -> GateResult:
    if condition not in {
        "candidate",
        "random_role",
        "deranged_role",
        "shuffled_role",
        "content_cleanup_positive",
    }:
        raise ValueError(f"unknown condition: {condition}")

    seed_key = str(seed)
    rows_raw = source["source_rows_by_seed"][seed_key]
    query_plan = source["query_plan_by_seed"][seed_key]
    if len(rows_raw) != N:
        raise ValueError(f"source rows for seed {seed} have len {len(rows_raw)} != {N}")
    if len(query_plan) != n_queries:
        raise ValueError(
            f"query plan for seed {seed} has len {len(query_plan)} != {n_queries}"
        )

    fhrr = TorchFHRR(dim=D, seed=seed, device=device)
    roles = fhrr.random_vectors(K_roles)
    content = fhrr.random_vectors(C_codebook)
    rows = torch.tensor(rows_raw, dtype=torch.long, device=fhrr.device)
    scene_matrix = _scene_bundles(
        fhrr,
        roles,
        content,
        rows,
        scene_token_weight=scene_token_weight,
    )

    scene_idx = torch.tensor(
        [int(item["scene"]) for item in query_plan],
        dtype=torch.long,
        device=fhrr.device,
    )
    known_role = torch.tensor(
        [int(item["known_role"]) for item in query_plan],
        dtype=torch.long,
        device=fhrr.device,
    )
    query_role = torch.tensor(
        [int(item["query_role"]) for item in query_plan],
        dtype=torch.long,
        device=fhrr.device,
    )
    target_atom = rows[scene_idx, query_role]

    perm_generator = torch.Generator(device="cpu").manual_seed(seed * 4001 + K_roles)
    role_shuffle = torch.tensor(
        EXP44._role_permutation(K_roles, generator=perm_generator),
        dtype=torch.long,
        device=fhrr.device,
    )
    derange_generator = torch.Generator(device="cpu").manual_seed(seed * 5003 + K_roles)
    role_derangement = torch.tensor(
        EXP44._role_derangement(K_roles, generator=derange_generator),
        dtype=torch.long,
        device=fhrr.device,
    )

    cue_role = known_role
    unbind_role = query_role
    if condition == "random_role" and K_roles > 1:
        unbind_role = (query_role + 1) % K_roles
    elif condition == "shuffled_role":
        cue_role = role_shuffle[known_role]
        unbind_role = role_shuffle[query_role]
    elif condition == "deranged_role":
        cue_role = role_derangement[known_role]
        unbind_role = role_derangement[query_role]

    known_atom = rows[scene_idx, known_role]
    cue = roles[cue_role] * content[known_atom]
    query_tokens = _query_context_tokens(fhrr, roles, content, rows, query_plan)
    cue = fhrr.normalize(cue + scene_token_weight * query_tokens)
    cue = EXP44._perturb_batch(fhrr, cue, cue_noise)

    zero_stats = torch.zeros(n_queries, device=fhrr.device)
    if condition == "content_cleanup_positive":
        scene_state = None
        scene_top_index = scene_idx
        scene_entropy = zero_stats
        scene_margin = zero_stats
        content_query = EXP44._perturb_batch(fhrr, content[target_atom], cue_noise)
    else:
        scene_state, scene_top_index, scene_entropy, scene_margin = (
            EXP44._batched_hopfield_retrieve(
                fhrr,
                scene_matrix,
                cue,
                beta=beta,
            )
        )
        content_query = fhrr.normalize(fhrr.unbind(scene_state, roles[unbind_role]))

    scene_tix = int((scene_top_index == scene_idx).sum().detach().cpu())
    content_state, content_top_index, content_entropy, content_margin = (
        EXP44._batched_hopfield_retrieve(
            fhrr,
            content,
            content_query,
            beta=beta,
        )
    )
    content_tix = int((content_top_index == target_atom).sum().detach().cpu())
    pred = torch.argmax((content_state @ content.conj().T).real / content.shape[1], dim=1)
    n_correct = int((pred == target_atom).sum().detach().cpu())

    return GateResult(
        condition=condition,
        D=D,
        N=N,
        K_roles=K_roles,
        cue_noise=cue_noise,
        scene_token_weight=scene_token_weight,
        source_name="native_provenance_context_trace",
        context_roles=context_roles,
        cooccurrence=cooccurrence,
        seed=seed,
        n_queries=n_queries,
        n_correct=n_correct,
        scene_tix=scene_tix,
        content_tix=content_tix,
        scene_entropy=float(scene_entropy.mean().detach().cpu()),
        content_entropy=float(content_entropy.mean().detach().cpu()),
        scene_margin=float(scene_margin.mean().detach().cpu()),
        content_margin=float(content_margin.mean().detach().cpu()),
        source_rows_available=len(rows_raw),
        source_rows_used=N,
        source_rows_invalid=0,
        source_rows_too_short=0,
        source_artifact_path=str(source_path),
        source_artifact_sha256=source_sha,
    )


def _aggregate(cell: Sequence[GateResult]) -> dict:
    n_total = sum(r.n_queries for r in cell)
    n_correct = sum(r.n_correct for r in cell)
    ci = _wilson(n_correct, n_total)
    scene_tix = sum(r.scene_tix for r in cell)
    content_tix = sum(r.content_tix for r in cell)
    return {
        "top1_mean": ci["mean"],
        "wilson_lo": ci["lo"],
        "wilson_hi": ci["hi"],
        "n_total": n_total,
        "n_correct": n_correct,
        "per_seed_top1": [r.top1 for r in cell],
        "scene_tix": scene_tix,
        "content_tix": content_tix,
        "scene_tix_rate": scene_tix / n_total if n_total else 0.0,
        "content_tix_rate": content_tix / n_total if n_total else 0.0,
        "mean_scene_entropy": sum(r.scene_entropy for r in cell) / len(cell),
        "mean_content_entropy": sum(r.content_entropy for r in cell) / len(cell),
        "mean_scene_margin": sum(r.scene_margin for r in cell) / len(cell),
        "mean_content_margin": sum(r.content_margin for r in cell) / len(cell),
        "source_rows_available": [r.source_rows_available for r in cell],
        "source_rows_used": [r.source_rows_used for r in cell],
        "source_rows_invalid": [r.source_rows_invalid for r in cell],
        "source_rows_too_short": [r.source_rows_too_short for r in cell],
        "source_artifact_paths": sorted({r.source_artifact_path for r in cell}),
        "source_artifact_sha256": sorted({r.source_artifact_sha256 for r in cell}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="reports/phase5_prime_native_provenance_context_source.json",
    )
    parser.add_argument(
        "--preflight",
        default="reports/phase5_prime_true_provenance_preflight.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_true_provenance_gate.json",
    )
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--K_roles", type=int, default=16)
    parser.add_argument("--C_codebook", type=int, default=2048)
    parser.add_argument("--context_roles", type=int, default=4)
    parser.add_argument("--cue_noise", type=float, default=0.15)
    parser.add_argument("--token_weight", type=float, default=0.25)
    parser.add_argument("--cooccurrence", choices=["skewed", "uniform"], default="skewed")
    parser.add_argument("--n_queries", type=int, default=512)
    parser.add_argument("--beta", type=float, default=30.0)
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

    source_path = Path(args.source)
    preflight_path = Path(args.preflight)
    source = _load_json(source_path)
    preflight = _load_json(preflight_path)
    _validate_source_artifact(source, args)
    source_sha = _sha256(source_path)
    expected_sha = preflight.get("source_manifest", {}).get("source_artifact_sha256")
    if expected_sha != source_sha:
        raise ValueError(
            f"source SHA mismatch: preflight={expected_sha} artifact={source_sha}"
        )
    if not all(preflight.get("pass_criteria", {}).values()):
        raise ValueError("preflight pass_criteria are not all true")

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
                device=device,
            )
            cell.append(result)
            raw.append(result)
        key = (
            f"{condition}|D={args.D}|K={args.K_roles}|N={args.N}|"
            f"noise={args.cue_noise}|scene_token=1|token_weight={args.token_weight}|"
            f"token_source=native_provenance_context_trace|context_roles={args.context_roles}|"
            f"cooc={args.cooccurrence}"
        )
        agg = _aggregate(cell)
        aggregates[key] = agg
        print(
            f"{key} top1={agg['top1_mean']:.4f} "
            f"CI=[{agg['wilson_lo']:.4f},{agg['wilson_hi']:.4f}] "
            f"scene_tix={agg['scene_tix']}/{agg['n_total']} "
            f"content_tix={agg['content_tix']}/{agg['n_total']} "
            f"ent=({agg['mean_scene_entropy']:.3f},"
            f"{agg['mean_content_entropy']:.3f}) "
            f"margin=({agg['mean_scene_margin']:.4f},"
            f"{agg['mean_content_margin']:.4f})"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "framing": {
            "phase": "5-prime true provenance-source diagnostic",
            "source_name": "native_provenance_context_trace",
            "source_kind": "synthetic_controlled_native_provenance",
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
