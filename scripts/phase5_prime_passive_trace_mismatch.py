"""Analyze the Phase 5' passive trace context-source mismatch.

This is analysis-only follow-up to Reports 082/083. It does not run a new
condition grid or alter the hard-cell contract. It reconstructs the fixed
passive-source row selection and compares:

- observed passive row role/atom support;
- query-role coverage implied by those passive rows;
- synthetic re-encoded context geometry used by Report 082;
- learned snapshot-pattern context geometry used by Report 083.
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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_snapshot(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"snapshot is not a dict: {path}")
    return state


def _f(v: Any) -> float:
    return float(v.detach().cpu()) if hasattr(v, "detach") else float(v)


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
    n = sims.shape[0]
    if n != sims.shape[1]:
        raise ValueError("alignment stats require square pairings")
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
    mask = ~torch.eye(n, dtype=torch.bool, device=sims.device)
    off = sims[mask]
    masked = sims.masked_fill(torch.eye(n, dtype=torch.bool, device=sims.device), -1e9)
    nn = masked.max(dim=1).values
    return {
        "offdiag_mean": _f(off.mean()),
        "offdiag_std": _f(off.std(unbiased=False)),
        "nn_mean": _f(nn.mean()),
        "nn_max": _f(nn.max()),
    }


def _aggregate_alignment(rows: Sequence[dict], key: str) -> Dict[str, float]:
    metric_keys = rows[0][key].keys()
    out: Dict[str, float] = {}
    for metric in metric_keys:
        vals = [float(row[key][metric]) for row in rows]
        out[f"{metric}_mean"] = float(mean(vals))
        out[f"{metric}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def _top_seed_results(raw_json: dict, condition: str) -> Dict[int, float]:
    return {
        int(row["seed"]): float(row["top1"])
        for row in raw_json["raw"]
        if row["condition"] == condition
    }


def _role_atom_snapshot_summary(
    raw_rows: Sequence[Optional[Sequence[Sequence[int]]]],
    *,
    k_roles: int,
    c_codebook: int,
    context_roles: int,
) -> dict:
    role_counts: Counter = Counter()
    role_set_counts: Counter = Counter()
    atom_counts: Counter = Counter()
    atom_counts_by_role: Dict[int, Counter] = {role: Counter() for role in range(k_roles)}
    eligible = 0
    too_short = 0
    invalid_rows = 0
    for raw_terms in raw_rows:
        if raw_terms is None:
            too_short += 1
            continue
        row_invalid = False
        role_to_atom: Dict[int, int] = {}
        for raw_role, raw_atom in raw_terms:
            role = int(raw_role)
            atom = int(raw_atom)
            if role < 0 or role >= k_roles or atom < 0 or atom >= c_codebook:
                row_invalid = True
                continue
            role_to_atom.setdefault(role, atom)
        if row_invalid:
            invalid_rows += 1
        if len(role_to_atom) < context_roles:
            too_short += 1
            continue
        eligible += 1
        role_set_counts[tuple(sorted(role_to_atom.keys()))] += 1
        for role, atom in role_to_atom.items():
            role_counts[role] += 1
            atom_counts[atom] += 1
            atom_counts_by_role[role][atom] += 1

    per_role = {}
    for role in range(k_roles):
        counts = atom_counts_by_role[role]
        total = sum(counts.values())
        per_role[str(role)] = {
            "count": total,
            "distinct_atoms": len(counts),
            "normalized_entropy": _normalized_entropy(counts),
            "top_atoms": [[int(atom), int(count)] for atom, count in counts.most_common(8)],
        }
    return {
        "raw_rows": len(raw_rows),
        "eligible_rows": eligible,
        "rows_too_short": too_short,
        "rows_invalid": invalid_rows,
        "role_counts": {str(role): int(role_counts[role]) for role in range(k_roles)},
        "role_set_counts": [
            {"roles": list(roles), "count": int(count)}
            for roles, count in role_set_counts.most_common()
        ],
        "atom_distinct": len(atom_counts),
        "atom_normalized_entropy": _normalized_entropy(atom_counts),
        "atom_top": [[int(atom), int(count)] for atom, count in atom_counts.most_common(12)],
        "atom_fraction_ge_1024": (
            sum(count for atom, count in atom_counts.items() if int(atom) >= 1024)
            / max(1, sum(atom_counts.values()))
        ),
        "per_role": per_role,
    }


def _seed_geometry(
    *,
    seed: int,
    snapshot_path: Path,
    D: int,
    N: int,
    K_roles: int,
    C_codebook: int,
    context_roles: int,
    cue_noise: float,
    cooccurrence: str,
    token_weight: float,
    n_queries: int,
    report082_top1: Dict[int, float],
    report083_top1: Dict[int, float],
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
        raise RuntimeError("expected learned pattern tokens")
    selected_roles = EXP44._apply_passive_trace_context_rows(
        fillers,
        passive_rows,
        context_roles=context_roles,
        generator=generator,
    )

    synthetic_full_tokens = torch.stack(
        [
            fhrr.bundle([
                roles[role] * content[int(atom)]
                for role, atom in enumerate(row)
            ])
            for row in fillers
        ],
        dim=0,
    )
    filler_tensor = torch.tensor(fillers, dtype=torch.long, device=fhrr.device)
    synthetic_partial_tokens = torch.stack(
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

    scene_matrix_082 = torch.stack(
        EXP44._build_scene_bundles(
            fhrr,
            roles,
            content,
            fillers,
            scene_tokens=synthetic_full_tokens,
            scene_token_weight=token_weight,
        ),
        dim=0,
    )
    scene_matrix_083 = torch.stack(
        EXP44._build_scene_bundles(
            fhrr,
            roles,
            content,
            fillers,
            scene_tokens=learned_tokens,
            scene_token_weight=token_weight,
        ),
        dim=0,
    )

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

    abs_values = learned_tokens.abs()
    role_set_counts = Counter(tuple(role_list) for role_list in selected_roles)

    return {
        "seed": seed,
        "report082_candidate_top1": report082_top1.get(seed),
        "report083_candidate_top1": report083_top1.get(seed),
        "support": support,
        "selected_role_sets": [
            {"roles": list(roles), "count": int(count)}
            for roles, count in role_set_counts.most_common()
        ],
        "observed_role_counts": {
            str(role): int(observed_role_counts[role]) for role in range(K_roles)
        },
        "query_role_counts": {
            str(role): int(query_role_counts[role]) for role in range(K_roles)
        },
        "query_fraction_in_observed_role_support": (
            sum(count for role, count in query_role_counts.items() if observed_role_counts[role] > 0)
            / max(1, sum(query_role_counts.values()))
        ),
        "learned_token_abs_mean": _f(abs_values.mean()),
        "learned_token_abs_max_deviation_from_one": _f((abs_values - 1.0).abs().max()),
        "partial_context_to_synthetic_full_context": _alignment_stats(
            synthetic_partial_tokens, synthetic_full_tokens,
        ),
        "learned_pattern_to_synthetic_full_context": _alignment_stats(
            learned_tokens, synthetic_full_tokens,
        ),
        "learned_pattern_to_partial_context": _alignment_stats(
            learned_tokens, synthetic_partial_tokens,
        ),
        "partial_context_to_scene_matrix_082": _alignment_stats(
            synthetic_partial_tokens, scene_matrix_082,
        ),
        "learned_pattern_to_scene_matrix_083": _alignment_stats(
            learned_tokens, scene_matrix_083,
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
        "--report082-json",
        default="reports/phase5_prime_replay_observed_context_hard_cell.json",
    )
    parser.add_argument(
        "--report083-json",
        default="reports/phase5_prime_replay_observed_pattern_context_hard_cell.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_passive_trace_mismatch_analysis.json",
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

    snapshot_path = Path(args.snapshot)
    report082 = _load_json(Path(args.report082_json))
    report083 = _load_json(Path(args.report083_json))
    snapshot = _load_snapshot(snapshot_path)
    raw_rows = snapshot.get("pattern_encoder_terms")
    if raw_rows is None:
        raise ValueError(f"snapshot has no pattern_encoder_terms: {snapshot_path}")

    report082_top1 = _top_seed_results(report082, "candidate")
    report083_top1 = _top_seed_results(report083, "candidate")
    per_seed = [
        _seed_geometry(
            seed=seed,
            snapshot_path=snapshot_path,
            D=args.D,
            N=args.N,
            K_roles=args.K_roles,
            C_codebook=args.C_codebook,
            context_roles=args.context_roles,
            cue_noise=args.cue_noise,
            cooccurrence=args.cooccurrence,
            token_weight=args.token_weight,
            n_queries=args.n_queries,
            report082_top1=report082_top1,
            report083_top1=report083_top1,
        )
        for seed in args.seeds
    ]

    alignment_keys = [
        "partial_context_to_synthetic_full_context",
        "learned_pattern_to_synthetic_full_context",
        "learned_pattern_to_partial_context",
        "partial_context_to_scene_matrix_082",
        "learned_pattern_to_scene_matrix_083",
        "learned_pattern_pairwise",
    ]

    query_in_observed = [row["query_fraction_in_observed_role_support"] for row in per_seed]
    learned_to_full_diag_top1_mean = mean([
        row["learned_pattern_to_synthetic_full_context"]["diag_top1_rate"]
        for row in per_seed
    ])
    payload = {
        "framing": {
            "phase": "5-prime diagnostic analysis",
            "not_graduation": True,
            "analysis_only": True,
            "no_new_matrix": True,
        },
        "config": {
            "snapshot": str(snapshot_path),
            "report082_json": args.report082_json,
            "report083_json": args.report083_json,
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
        "snapshot_summary": _role_atom_snapshot_summary(
            raw_rows,
            k_roles=args.K_roles,
            c_codebook=args.C_codebook,
            context_roles=args.context_roles,
        ),
        "aggregate": {
            "report082_candidate_top1": _stats(list(report082_top1.values())),
            "report083_candidate_top1": _stats(list(report083_top1.values())),
            "query_fraction_in_observed_role_support": _stats(query_in_observed),
            **{
                key: _aggregate_alignment(per_seed, key)
                for key in alignment_keys
            },
        },
        "per_seed": per_seed,
        "decision_read": {
            "passive_rows_cover_only_roles_0_to_3": True,
            "hard_cell_query_roles_are_outside_passive_row_role_support": (
                max(query_in_observed) == 0.0
            ),
            "learned_snapshot_tokens_align_to_synthetic_context_at_chance": (
                learned_to_full_diag_top1_mean < 0.01
            ),
            "learned_snapshot_token_diag_top1_rate_mean": learned_to_full_diag_top1_mean,
            "next_step": (
                "analyze source-provenance mismatch; do not run full matrix or M2 "
                "from Reports 082-084"
            ),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out_path),
        "snapshot_eligible_rows": payload["snapshot_summary"]["eligible_rows"],
        "query_fraction_in_observed_role_support": payload["aggregate"][
            "query_fraction_in_observed_role_support"
        ],
        "partial_to_full_diag_top1_mean": payload["aggregate"][
            "partial_context_to_synthetic_full_context"
        ]["diag_top1_rate_mean"],
        "learned_to_full_diag_top1_mean": payload["aggregate"][
            "learned_pattern_to_synthetic_full_context"
        ]["diag_top1_rate_mean"],
        "learned_to_scene083_diag_top1_mean": payload["aggregate"][
            "learned_pattern_to_scene_matrix_083"
        ]["diag_top1_rate_mean"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
