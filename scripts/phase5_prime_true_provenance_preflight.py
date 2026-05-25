"""Preflight for the Phase 5' true provenance-source gate.

This script does not run candidate/control retrieval. It builds a fixed native
provenance source artifact in the same role universe as the diagnostic scene,
then reports support, query-schedule, atom, and geometry diagnostics required
before any top1 gate may be run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Sequence, Tuple

import torch

from energy_memory.substrate.torch_fhrr import TorchFHRR


EXP44 = importlib.import_module("experiments.44_phase5_prime_bundle_first")


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _role_atom_summary(
    rows: Sequence[Sequence[int]],
    *,
    k_roles: int,
) -> dict:
    role_counts: Counter = Counter()
    role_set_counts: Counter = Counter()
    atom_counts: Counter = Counter()
    atom_counts_by_role: Dict[int, Counter] = {role: Counter() for role in range(k_roles)}
    invalid = 0
    too_short = 0

    for row in rows:
        if len(row) < k_roles:
            too_short += 1
            continue
        role_set_counts[tuple(range(k_roles))] += 1
        for role, raw_atom in enumerate(row[:k_roles]):
            try:
                atom = int(raw_atom)
            except (TypeError, ValueError):
                invalid += 1
                continue
            role_counts[role] += 1
            atom_counts[atom] += 1
            atom_counts_by_role[role][atom] += 1

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
        "rows_invalid": int(invalid),
        "rows_too_short": int(too_short),
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


def _source_rows_for_seed(
    *,
    seed: int,
    N: int,
    K_roles: int,
    C_codebook: int,
    cooccurrence: str,
) -> List[List[int]]:
    generator = torch.Generator(device="cpu").manual_seed(
        seed * 1009 + N * 37 + K_roles
    )
    return EXP44._filler_indices(
        N,
        K_roles,
        C_codebook,
        cooccurrence=cooccurrence,
        generator=generator,
    )


def _selected_context_roles_for_seed(
    *,
    seed: int,
    N: int,
    K_roles: int,
    context_roles: int,
) -> List[List[int]]:
    generator = torch.Generator(device="cpu").manual_seed(
        seed * 2003 + N * 19 + context_roles * 97 + K_roles
    )
    selected: List[List[int]] = []
    for _ in range(N):
        order = torch.randperm(K_roles, generator=generator).tolist()
        selected.append(sorted(int(role) for role in order[:context_roles]))
    return selected


def _query_plan_for_seed(
    *,
    seed: int,
    N: int,
    K_roles: int,
    n_queries: int,
    selected_context_roles: Sequence[Sequence[int]],
) -> Tuple[List[Tuple[int, int, int]], List[List[int]]]:
    generator = torch.Generator(device="cpu").manual_seed(
        seed * 3001 + N * 29 + n_queries * 13 + K_roles
    )
    return EXP44._query_plan_with_passive_context(
        N,
        K_roles,
        n_queries,
        selected_context_roles,
        generator=generator,
    )


def _context_matrices(
    *,
    fhrr: TorchFHRR,
    roles: torch.Tensor,
    content: torch.Tensor,
    rows: Sequence[Sequence[int]],
    selected_context_roles: Sequence[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    full_context = torch.stack(
        [
            fhrr.bundle([
                roles[role] * content[int(atom)]
                for role, atom in enumerate(row)
            ])
            for row in rows
        ],
        dim=0,
    )
    partial_context = torch.stack(
        [
            fhrr.bundle([
                roles[int(role)] * content[int(rows[scene][int(role)])]
                for role in role_list
            ])
            for scene, role_list in enumerate(selected_context_roles)
        ],
        dim=0,
    )
    return full_context, partial_context


def _seed_preflight(
    *,
    seed: int,
    D: int,
    N: int,
    K_roles: int,
    C_codebook: int,
    context_roles: int,
    cooccurrence: str,
    n_queries: int,
) -> dict:
    rows = _source_rows_for_seed(
        seed=seed,
        N=N,
        K_roles=K_roles,
        C_codebook=C_codebook,
        cooccurrence=cooccurrence,
    )
    selected_context_roles = _selected_context_roles_for_seed(
        seed=seed,
        N=N,
        K_roles=K_roles,
        context_roles=context_roles,
    )
    plan, observed_role_plan = _query_plan_for_seed(
        seed=seed,
        N=N,
        K_roles=K_roles,
        n_queries=n_queries,
        selected_context_roles=selected_context_roles,
    )

    summary = _role_atom_summary(rows, k_roles=K_roles)
    rows_invalid = int(summary["rows_invalid"])
    rows_too_short = int(summary["rows_too_short"])
    eligible_rows = sum(1 for row in rows if len(row) >= K_roles) - rows_too_short
    used_rows = min(N, eligible_rows)

    observed_role_counts = Counter(role for roles_i in observed_role_plan for role in roles_i)
    query_role_counts = Counter(query for _scene, _known, query in plan)
    source_supported_roles = {
        int(role)
        for role, count in summary["role_counts"].items()
        if int(count) > 0
    }
    query_in_own_observed = sum(
        1
        for (_scene, _known, query), observed_roles_i in zip(plan, observed_role_plan)
        if query in observed_roles_i
    )
    selected_context_role_set_counts = Counter(
        tuple(role_list) for role_list in selected_context_roles
    )

    fhrr = TorchFHRR(dim=D, seed=seed, device="cpu")
    roles = fhrr.random_vectors(K_roles)
    content = fhrr.random_vectors(C_codebook)
    full_context, partial_context = _context_matrices(
        fhrr=fhrr,
        roles=roles,
        content=content,
        rows=rows,
        selected_context_roles=selected_context_roles,
    )

    return {
        "seed": seed,
        "support": {
            "raw_rows": len(rows),
            "eligible_rows": eligible_rows,
            "required_rows": N,
            "used_rows": used_rows,
            "rows_invalid": rows_invalid,
            "rows_too_short": rows_too_short,
            "context_roles": context_roles,
        },
        "selected_row_summary": summary,
        "selected_context_role_sets": [
            {"roles": list(roles_i), "count": int(count)}
            for roles_i, count in selected_context_role_set_counts.most_common()
        ],
        "observed_role_counts": {
            str(role): int(observed_role_counts[role]) for role in range(K_roles)
        },
        "query_role_counts": {
            str(role): int(query_role_counts[role]) for role in range(K_roles)
        },
        "query_fraction_in_source_role_support": (
            sum(count for role, count in query_role_counts.items()
                if int(role) in source_supported_roles)
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
        "source_observed_context_to_full_context": _alignment_stats(
            partial_context, full_context,
        ),
    }


def _source_artifact_payload(
    *,
    args: argparse.Namespace,
    per_seed_rows: Dict[str, List[List[int]]],
    per_seed_context_roles: Dict[str, List[List[int]]],
    per_seed_query_plan: Dict[str, List[dict]],
) -> dict:
    return {
        "framing": {
            "phase": "5-prime true provenance-source diagnostic",
            "source_name": "native_provenance_context_trace",
            "source_kind": "synthetic_controlled_native_provenance",
            "not_graduation": True,
            "source_artifact_only": True,
            "no_candidate_control_run": True,
            "description": (
                "Encoder-term rows are generated directly in the diagnostic "
                "role universe before retrieval evaluation. Query plans hold "
                "out the target role from each query's observed context."
            ),
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
            "seeds": args.seeds,
            "source_row_seed_formula": "seed * 1009 + N * 37 + K_roles",
            "context_role_seed_formula": (
                "seed * 2003 + N * 19 + context_roles * 97 + K_roles"
            ),
            "query_plan_seed_formula": (
                "seed * 3001 + N * 29 + n_queries * 13 + K_roles"
            ),
        },
        "role_universe": list(range(args.K_roles)),
        "source_rows_by_seed": per_seed_rows,
        "selected_context_roles_by_seed": per_seed_context_roles,
        "query_plan_by_seed": per_seed_query_plan,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-out",
        default="reports/phase5_prime_native_provenance_context_source.json",
    )
    parser.add_argument(
        "--preflight-out",
        default="reports/phase5_prime_true_provenance_preflight.json",
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
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[17, 11, 23, 1, 2, 3, 5, 7, 13, 29],
    )
    args = parser.parse_args()

    if args.context_roles <= 0:
        raise ValueError("context_roles must be positive")
    if args.context_roles >= args.K_roles:
        raise ValueError("context_roles must be less than K_roles")

    per_seed_rows: Dict[str, List[List[int]]] = {}
    per_seed_context_roles: Dict[str, List[List[int]]] = {}
    per_seed_query_plan: Dict[str, List[dict]] = {}
    for seed in args.seeds:
        rows = _source_rows_for_seed(
            seed=seed,
            N=args.N,
            K_roles=args.K_roles,
            C_codebook=args.C_codebook,
            cooccurrence=args.cooccurrence,
        )
        selected_roles = _selected_context_roles_for_seed(
            seed=seed,
            N=args.N,
            K_roles=args.K_roles,
            context_roles=args.context_roles,
        )
        plan, observed = _query_plan_for_seed(
            seed=seed,
            N=args.N,
            K_roles=args.K_roles,
            n_queries=args.n_queries,
            selected_context_roles=selected_roles,
        )
        per_seed_rows[str(seed)] = rows
        per_seed_context_roles[str(seed)] = selected_roles
        per_seed_query_plan[str(seed)] = [
            {
                "scene": int(scene),
                "known_role": int(known),
                "query_role": int(query),
                "observed_roles": [int(role) for role in observed_roles],
            }
            for (scene, known, query), observed_roles in zip(plan, observed)
        ]

    source_path = Path(args.source_out)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_payload = _source_artifact_payload(
        args=args,
        per_seed_rows=per_seed_rows,
        per_seed_context_roles=per_seed_context_roles,
        per_seed_query_plan=per_seed_query_plan,
    )
    source_path.write_text(json.dumps(source_payload, indent=2, sort_keys=True) + "\n")
    source_sha = _sha256(source_path)

    per_seed = [
        _seed_preflight(
            seed=seed,
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
    query_source = [row["query_fraction_in_source_role_support"] for row in per_seed]
    query_global_observed = [
        row["query_fraction_in_global_observed_role_set"] for row in per_seed
    ]
    query_own_observed = [
        row["query_fraction_in_own_observed_context"] for row in per_seed
    ]
    atom_distinct = [
        float(row["selected_row_summary"]["atom_distinct"]) for row in per_seed
    ]
    atom_entropy = [
        float(row["selected_row_summary"]["atom_normalized_entropy"])
        for row in per_seed
    ]
    atom_fraction_ge_1024 = [
        float(row["selected_row_summary"]["atom_fraction_ge_1024"])
        for row in per_seed
    ]

    payload = {
        "framing": {
            "phase": "5-prime true provenance-source diagnostic preflight",
            "source_name": "native_provenance_context_trace",
            "source_kind": "synthetic_controlled_native_provenance",
            "not_graduation": True,
            "preflight_only": True,
            "no_candidate_control_run": True,
        },
        "source_manifest": {
            "source_name": "native_provenance_context_trace",
            "source_build_command": (
                "PYTHONPATH=src:. .venv/bin/python "
                "scripts/phase5_prime_true_provenance_preflight.py"
            ),
            "source_artifact_path": str(source_path),
            "source_artifact_sha256": source_sha,
            "D": args.D,
            "K_roles": args.K_roles,
            "C_codebook": args.C_codebook,
            "N_raw_rows": args.N,
            "role_universe": list(range(args.K_roles)),
            "pattern_token_shape": None,
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
            "seeds": args.seeds,
        },
        "aggregate": {
            "support": aggregate_support,
            "query_fraction_in_source_role_support": _stats(query_source),
            "query_fraction_in_global_observed_role_set": _stats(query_global_observed),
            "query_fraction_in_own_observed_context": _stats(query_own_observed),
            "atom_distinct": _stats(atom_distinct),
            "atom_normalized_entropy": _stats(atom_entropy),
            "atom_fraction_ge_1024": _stats(atom_fraction_ge_1024),
            "source_observed_context_to_full_context": _aggregate_metric(
                per_seed, "source_observed_context_to_full_context"
            ),
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
            "source_artifact_sha256_recorded": bool(source_sha),
            "geometry_reported": True,
        },
        "decision_read": (
            "preflight only; support/query checks pass if all criteria are true. "
            "Inspect source geometry before any native provenance candidate/control gate."
        ),
    }

    preflight_path = Path(args.preflight_out)
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "source_out": str(source_path),
        "source_sha256": source_sha,
        "preflight_out": str(preflight_path),
        "pass_criteria": payload["pass_criteria"],
        "query_fraction_in_source_role_support": payload["aggregate"][
            "query_fraction_in_source_role_support"
        ],
        "query_fraction_in_own_observed_context": payload["aggregate"][
            "query_fraction_in_own_observed_context"
        ],
        "source_observed_to_full_diag_top1_mean": payload["aggregate"][
            "source_observed_context_to_full_context"
        ]["diag_top1_rate_mean"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
