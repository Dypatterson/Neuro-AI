"""Preflight diagnostics for the Phase 5' context-source v2 gate.

This script does not run candidate/control retrieval. It checks whether the
matched 4-role passive context-source gate is well-posed before any top1
experiment is launched.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence

import torch

from energy_memory.substrate.torch_fhrr import TorchFHRR


EXP44 = importlib.import_module("experiments.44_phase5_prime_bundle_first")


def _load_snapshot(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"snapshot is not a dict: {path}")
    return state


def _f(value: Any) -> float:
    return float(value.detach().cpu()) if hasattr(value, "detach") else float(value)


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _normalized_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return float(entropy / math.log(len(counts)))


def _similarity_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left @ right.conj().T).real / left.shape[1]


def _alignment_stats(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    sims = _similarity_matrix(left, right)
    if sims.shape[0] != sims.shape[1]:
        raise ValueError("alignment stats require square pairings")
    n = sims.shape[0]
    diag = sims.diag()
    mask = ~torch.eye(n, dtype=torch.bool, device=sims.device)
    off = sims[mask]
    ranks: List[int] = []
    top1 = 0
    for i in range(n):
        row = sims[i]
        diag_score = row[i]
        rank = int((row > diag_score).sum().item()) + 1
        ranks.append(rank)
        if int(torch.argmax(row).item()) == i:
            top1 += 1
    return {
        "diag_mean": _f(diag.mean()),
        "diag_std": _f(diag.std(unbiased=False)),
        "offdiag_mean": _f(off.mean()),
        "offdiag_std": _f(off.std(unbiased=False)),
        "diag_minus_offdiag_mean": _f(diag.mean() - off.mean()),
        "diag_top1_rate": top1 / n if n else 0.0,
        "diag_mean_rank": float(mean(ranks)) if ranks else 0.0,
        "diag_median_rank": float(sorted(ranks)[len(ranks) // 2]) if ranks else 0.0,
    }


def _nearest_neighbor_stats(vectors: torch.Tensor) -> Dict[str, float]:
    sims = _similarity_matrix(vectors, vectors)
    n = sims.shape[0]
    if n <= 1:
        return {"offdiag_mean": 0.0, "offdiag_std": 0.0, "nn_mean": 0.0, "nn_max": 0.0}
    eye = torch.eye(n, dtype=torch.bool, device=sims.device)
    off = sims[~eye]
    nn = sims.masked_fill(eye, -1e9).max(dim=1).values
    return {
        "offdiag_mean": _f(off.mean()),
        "offdiag_std": _f(off.std(unbiased=False)),
        "nn_mean": _f(nn.mean()),
        "nn_max": _f(nn.max()),
    }


def _role_atom_summary(
    rows: Sequence[Dict[int, int]],
    *,
    k_roles: int,
) -> dict:
    role_counts: Counter = Counter()
    role_set_counts: Counter = Counter()
    atom_counts: Counter = Counter()
    atom_counts_by_role: Dict[int, Counter] = {role: Counter() for role in range(k_roles)}
    for row in rows:
        role_set_counts[tuple(sorted(row.keys()))] += 1
        for role, atom in row.items():
            role_counts[int(role)] += 1
            atom_counts[int(atom)] += 1
            atom_counts_by_role[int(role)][int(atom)] += 1

    per_role = {}
    for role in range(k_roles):
        counts = atom_counts_by_role[role]
        per_role[str(role)] = {
            "count": int(sum(counts.values())),
            "distinct_atoms": int(len(counts)),
            "normalized_entropy": _normalized_entropy(counts),
            "top_atoms": [[int(atom), int(count)] for atom, count in counts.most_common(8)],
        }

    return {
        "role_counts": {str(role): int(role_counts[role]) for role in range(k_roles)},
        "role_set_counts": [
            {"roles": list(roles), "count": int(count)}
            for roles, count in role_set_counts.most_common()
        ],
        "atom_distinct": int(len(atom_counts)),
        "atom_normalized_entropy": _normalized_entropy(atom_counts),
        "atom_top": [[int(atom), int(count)] for atom, count in atom_counts.most_common(12)],
        "atom_fraction_ge_1024": (
            sum(count for atom, count in atom_counts.items() if int(atom) >= 1024)
            / max(1, sum(atom_counts.values()))
        ),
        "per_role": per_role,
    }


def _aggregate_metric(rows: Sequence[dict], key: str) -> Dict[str, float]:
    metrics = rows[0][key].keys()
    out: Dict[str, float] = {}
    for metric in metrics:
        vals = [float(row[key][metric]) for row in rows]
        out[f"{metric}_mean"] = float(mean(vals))
        out[f"{metric}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def _seed_preflight(
    *,
    seed: int,
    snapshot_path: Path,
    D: int,
    N: int,
    K_roles: int,
    C_codebook: int,
    context_roles: int,
    cooccurrence: str,
    n_queries: int,
) -> dict:
    fhrr = TorchFHRR(dim=D, seed=seed, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed * 1009 + N * 37 + K_roles)
    roles = fhrr.random_vectors(K_roles)
    content = fhrr.random_vectors(C_codebook)
    fillers = EXP44._filler_indices(
        N,
        K_roles,
        C_codebook,
        cooccurrence=cooccurrence,
        generator=generator,
    )
    passive_rows, support, learned_tokens = EXP44._load_passive_trace_context_rows(
        snapshot_path,
        N=N,
        K_roles=K_roles,
        C_codebook=C_codebook,
        context_roles=context_roles,
        generator=generator,
        return_pattern_tokens=True,
    )
    if learned_tokens is None:
        raise RuntimeError("expected learned snapshot pattern tokens")
    if int(learned_tokens.shape[1]) != D:
        raise ValueError(
            f"learned token dim mismatch: {int(learned_tokens.shape[1])} != {D}"
        )

    selected_roles = EXP44._apply_passive_trace_context_rows(
        fillers,
        passive_rows,
        context_roles=context_roles,
        generator=generator,
    )
    filler_tensor = torch.tensor(fillers, dtype=torch.long, device=fhrr.device)
    synthetic_full = torch.stack(
        [
            fhrr.bundle([
                roles[role] * content[int(atom)]
                for role, atom in enumerate(row)
            ])
            for row in fillers
        ],
        dim=0,
    )
    synthetic_partial = torch.stack(
        [
            fhrr.bundle([
                roles[role] * content[int(filler_tensor[scene, role].item())]
                for role in role_list
            ])
            for scene, role_list in enumerate(selected_roles)
        ],
        dim=0,
    )
    learned_tokens = learned_tokens.to(fhrr.device)

    # Match run_cell's generator consumption before passive query planning.
    EXP44._role_permutation(K_roles, generator=generator)
    plan, observed_role_plan = EXP44._query_plan_with_passive_context(
        N,
        K_roles,
        n_queries,
        selected_roles,
        generator=generator,
    )

    observed_role_counts = Counter(role for roles_i in observed_role_plan for role in roles_i)
    query_role_counts = Counter(query for _scene, _known, query in plan)
    query_in_own_observed = sum(
        1
        for (_scene, _known, query), observed_roles_i in zip(plan, observed_role_plan)
        if query in observed_roles_i
    )
    selected_row_summary = _role_atom_summary(passive_rows, k_roles=K_roles)
    role_set_counts = Counter(tuple(role_list) for role_list in selected_roles)

    return {
        "seed": seed,
        "support": support,
        "selected_row_summary": selected_row_summary,
        "selected_context_role_sets": [
            {"roles": list(roles_i), "count": int(count)}
            for roles_i, count in role_set_counts.most_common()
        ],
        "observed_role_counts": {
            str(role): int(observed_role_counts[role]) for role in range(K_roles)
        },
        "query_role_counts": {
            str(role): int(query_role_counts[role]) for role in range(K_roles)
        },
        "query_fraction_in_source_role_support": (
            sum(count for role, count in query_role_counts.items()
                if selected_row_summary["role_counts"].get(str(role), 0) > 0)
            / max(1, sum(query_role_counts.values()))
        ),
        "query_fraction_in_global_observed_role_set": (
            sum(count for role, count in query_role_counts.items()
                if observed_role_counts[role] > 0)
            / max(1, sum(query_role_counts.values()))
        ),
        "query_fraction_in_own_observed_context": (
            query_in_own_observed / max(1, len(plan))
        ),
        "partial_context_to_synthetic_full_context": _alignment_stats(
            synthetic_partial, synthetic_full,
        ),
        "learned_pattern_to_synthetic_full_context": _alignment_stats(
            learned_tokens, synthetic_full,
        ),
        "learned_pattern_to_partial_context": _alignment_stats(
            learned_tokens, synthetic_partial,
        ),
        "learned_pattern_pairwise": _nearest_neighbor_stats(learned_tokens),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        default="reports/phase5_m1_provenance_seed17/snapshots/phase3_phase4_w4_step1800.pt",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_context_source_v2_preflight.json",
    )
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--K_roles", type=int, default=4)
    parser.add_argument("--C_codebook", type=int, default=2048)
    parser.add_argument("--context_roles", type=int, default=2)
    parser.add_argument("--cue_noise", type=float, default=0.15)
    parser.add_argument("--token_weight", type=float, default=0.25)
    parser.add_argument("--cooccurrence", choices=["skewed", "uniform"], default="skewed")
    parser.add_argument("--n_queries", type=int, default=512)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[17, 11, 23, 1, 2, 3, 5, 7, 13, 29],
    )
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    snapshot = _load_snapshot(snapshot_path)
    raw_rows = snapshot.get("pattern_encoder_terms")
    if raw_rows is None:
        raise ValueError(f"snapshot has no pattern_encoder_terms: {snapshot_path}")

    per_seed = [
        _seed_preflight(
            seed=seed,
            snapshot_path=snapshot_path,
            D=args.D,
            N=args.N,
            K_roles=args.K_roles,
            C_codebook=args.C_codebook,
            context_roles=args.context_roles,
            cooccurrence=args.cooccurrence,
            n_queries=args.n_queries,
        )
        for seed in args.seeds
    ]

    support_keys = [
        "raw_rows",
        "eligible_rows",
        "required_rows",
        "used_rows",
        "rows_invalid",
        "rows_too_short",
    ]
    aggregate_support = {
        key: _stats([float(row["support"][key]) for row in per_seed])
        for key in support_keys
    }
    alignment_keys = [
        "partial_context_to_synthetic_full_context",
        "learned_pattern_to_synthetic_full_context",
        "learned_pattern_to_partial_context",
        "learned_pattern_pairwise",
    ]
    query_source = [row["query_fraction_in_source_role_support"] for row in per_seed]
    query_global_observed = [
        row["query_fraction_in_global_observed_role_set"] for row in per_seed
    ]
    query_own_observed = [
        row["query_fraction_in_own_observed_context"] for row in per_seed
    ]

    payload = {
        "framing": {
            "phase": "5-prime diagnostic preflight",
            "not_graduation": True,
            "preflight_only": True,
            "no_candidate_control_run": True,
        },
        "config": {
            "snapshot": str(snapshot_path),
            "D": args.D,
            "N": args.N,
            "K_roles": args.K_roles,
            "C_codebook": args.C_codebook,
            "context_roles": args.context_roles,
            "cue_noise": args.cue_noise,
            "token_weight": args.token_weight,
            "cooccurrence": args.cooccurrence,
            "n_queries": args.n_queries,
            "seeds": args.seeds,
        },
        "aggregate": {
            "support": aggregate_support,
            "query_fraction_in_source_role_support": _stats(query_source),
            "query_fraction_in_global_observed_role_set": _stats(query_global_observed),
            "query_fraction_in_own_observed_context": _stats(query_own_observed),
            **{
                key: _aggregate_metric(per_seed, key)
                for key in alignment_keys
            },
        },
        "per_seed": per_seed,
        "pass_criteria": {
            "eligible_rows_ge_N": min(
                row["support"]["eligible_rows"] for row in per_seed
            ) >= args.N,
            "used_rows_eq_N": min(row["support"]["used_rows"] for row in per_seed) == args.N,
            "no_invalid_rows": max(row["support"]["rows_invalid"] for row in per_seed) == 0,
            "no_too_short_rows": max(
                row["support"]["rows_too_short"] for row in per_seed
            ) == 0,
            "query_roles_in_source_support": min(query_source) == 1.0,
            "query_role_held_out_from_own_observed_context": (
                max(query_own_observed) == 0.0
            ),
            "geometry_reported": True,
        },
        "decision_read": (
            "preflight passes support/role-universe checks; inspect geometry before "
            "running matched candidate/control gate"
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out_path),
        "pass_criteria": payload["pass_criteria"],
        "query_fraction_in_source_role_support": payload["aggregate"][
            "query_fraction_in_source_role_support"
        ],
        "query_fraction_in_global_observed_role_set": payload["aggregate"][
            "query_fraction_in_global_observed_role_set"
        ],
        "query_fraction_in_own_observed_context": payload["aggregate"][
            "query_fraction_in_own_observed_context"
        ],
        "partial_to_full_diag_top1_mean": payload["aggregate"][
            "partial_context_to_synthetic_full_context"
        ]["diag_top1_rate_mean"],
        "learned_to_full_diag_top1_mean": payload["aggregate"][
            "learned_pattern_to_synthetic_full_context"
        ]["diag_top1_rate_mean"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
