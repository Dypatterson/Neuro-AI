"""Audit Phase 5 M1 row-domain provenance in substrate snapshots.

This script checks whether a frozen Phase 4 snapshot carries per-pattern
encoder provenance that can be converted into M1 row-role weights over the
exact pattern rows M1 retrieves from. Old snapshots are expected to fail with
``missing_pattern_encoder_terms``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from energy_memory.phase4.snapshot import load_substrate_snapshot  # noqa: E402
from energy_memory.phase5.m1_role_energy import RoleBindingStats  # noqa: E402
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402


MEAN_ENTROPY_FAIL_THRESHOLD = 0.95
UNIFORM_ROW_ENTROPY_THRESHOLD = 0.99
UNIFORM_ROW_FRACTION_FAIL_THRESHOLD = 0.95
MIN_ROLE_FRACTION = 1e-6
EMPTY_ROW_FRACTION_FAIL_THRESHOLD = 0.5


def _load_snapshot(path: Path, device: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    patterns = state["patterns"]
    if patterns.numel() == 0:
        raise ValueError(f"snapshot {path} has zero stored patterns")
    substrate = TorchFHRR(dim=int(patterns.shape[-1]), device=device)
    mem, cons, info = load_substrate_snapshot(
        path=path, substrate=substrate, device=device,
    )
    return mem, cons, info


def _normalize_terms(
    raw_terms: Optional[Sequence[Any]],
) -> Optional[List[Tuple[int, int]]]:
    if raw_terms is None:
        return None
    return [(int(role_index), int(token_id)) for role_index, token_id in raw_terms]


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def _entropy_summary(counts: "torch.Tensor") -> Dict[str, Any]:
    n_roles = int(counts.shape[1])
    n_rows = int(counts.shape[0])
    if n_roles <= 1:
        return {
            "mean_normalized": float("nan"),
            "median_normalized": float("nan"),
            "max_normalized": float("nan"),
            "uniform_row_fraction": float("nan"),
            "uniform_rows": 0,
            "empty_row_fraction": float("nan"),
            "empty_rows": 0,
        }
    totals = counts.sum(dim=1, keepdim=True)
    nonempty = totals.squeeze(1) > 0
    probs = torch.where(totals > 0, counts / totals.clamp(min=1e-12), counts)
    safe = probs.clamp(min=1e-12)
    entropy = -(safe * safe.log()).sum(dim=1) / math.log(n_roles)
    values = [
        float(x)
        for x in entropy[nonempty].detach().cpu().tolist()
    ]
    uniform_rows = sum(1 for x in values if x >= UNIFORM_ROW_ENTROPY_THRESHOLD)
    empty_rows = int((~nonempty).sum().detach().cpu())
    return {
        "mean_normalized": sum(values) / len(values) if values else float("nan"),
        "median_normalized": _quantile(values, 0.5),
        "max_normalized": max(values) if values else float("nan"),
        "uniform_row_fraction": uniform_rows / len(values) if values else float("nan"),
        "uniform_rows": uniform_rows,
        "empty_row_fraction": empty_rows / n_rows if n_rows else float("nan"),
        "empty_rows": empty_rows,
    }


def _role_summary(matrix: "torch.Tensor") -> Dict[str, Any]:
    totals = matrix.sum(dim=0)
    total = float(totals.sum().detach().cpu())
    role_counts = [float(x) for x in totals.detach().cpu().tolist()]
    role_fractions = [
        (count / total if total > 0.0 else 0.0)
        for count in role_counts
    ]
    return {
        "role_counts": role_counts,
        "role_fractions": role_fractions,
        "role_coverage": sum(1 for count in role_counts if count > 0.0),
        "entropy": _entropy_summary(matrix),
    }


def _geometric_score_summary(scores: "torch.Tensor") -> Dict[str, Any]:
    row_spreads = scores.max(dim=1).values - scores.min(dim=1).values
    spread_values = [
        float(x)
        for x in row_spreads.detach().cpu().tolist()
    ]
    return {
        "min": float(scores.min().detach().cpu()),
        "max": float(scores.max().detach().cpu()),
        "mean": float(scores.mean().detach().cpu()),
        "row_spread_mean": (
            sum(spread_values) / len(spread_values)
            if spread_values else float("nan")
        ),
        "row_spread_median": _quantile(spread_values, 0.5),
        "row_spread_max": max(spread_values) if spread_values else float("nan"),
    }


def _degeneracy_reasons(
    *,
    entropy: Dict[str, Any],
    role_coverage: int,
    role_fractions: Sequence[float],
    n_roles: int,
    prefix: str = "",
) -> List[str]:
    reasons: List[str] = []
    mean_normalized = entropy.get("mean_normalized", float("nan"))
    uniform_fraction = entropy.get("uniform_row_fraction", float("nan"))
    empty_fraction = entropy.get("empty_row_fraction", 0.0)
    if not math.isnan(mean_normalized) and mean_normalized > MEAN_ENTROPY_FAIL_THRESHOLD:
        reasons.append(f"{prefix}mean_role_entropy_degenerate")
    if (
        not math.isnan(uniform_fraction)
        and uniform_fraction >= UNIFORM_ROW_FRACTION_FAIL_THRESHOLD
    ):
        reasons.append(f"{prefix}uniform_role_rows_degenerate")
    if (
        not math.isnan(empty_fraction)
        and empty_fraction > EMPTY_ROW_FRACTION_FAIL_THRESHOLD
    ):
        reasons.append(f"{prefix}empty_role_rows_degenerate")
    if role_coverage < n_roles or any(
        fraction <= MIN_ROLE_FRACTION for fraction in role_fractions
    ):
        reasons.append(f"{prefix}near_zero_role_utilization")
    return reasons


def audit_snapshot(
    snapshot_path: str | Path,
    *,
    device: str = "cpu",
    mask_token_id: Optional[int] = None,
    weight_source: str = "count",
    geometric_mode: str = "unbind_density",
    geometric_neighbor_k: int = 8,
    geometric_laplace: float = 1e-6,
    geometric_temperature: Optional[float] = 0.05,
) -> Dict[str, Any]:
    if weight_source not in {"count", "geometric"}:
        raise ValueError("weight_source must be 'count' or 'geometric'")
    if geometric_temperature is not None and geometric_temperature <= 0.0:
        raise ValueError("geometric_temperature must be positive")
    path = Path(snapshot_path)
    mem, cons, info = _load_snapshot(path, device=device)
    n_atoms = mem.stored_count
    metadata = info.get("metadata") or {}
    positions = info.get("positions")
    raw_terms = info.get("pattern_encoder_terms")
    raw_kinds = info.get("pattern_encoder_term_kinds")
    if mask_token_id is None and metadata.get("mask_token_id") is not None:
        mask_token_id = int(metadata["mask_token_id"])

    failures: List[str] = []
    warnings: List[str] = []
    if raw_terms is None or raw_kinds is None:
        failures.append("missing_pattern_encoder_terms")
        return {
            "status": "fail",
            "snapshot": str(path),
            "failure_reasons": failures,
            "warnings": warnings,
            "n_atoms": n_atoms,
            "n_roles": None,
            "metadata": metadata,
            "row_count_aligned": False,
            "pattern_encoder_terms_present": raw_terms is not None,
            "pattern_encoder_term_kinds_present": raw_kinds is not None,
        }

    row_count_aligned = len(raw_terms) == n_atoms and len(raw_kinds) == n_atoms
    if not row_count_aligned:
        failures.append("pattern_encoder_terms_row_count_mismatch")

    if positions is None:
        warnings.append("missing_positions; inferred n_roles from provenance")
        max_role = -1
        for terms in raw_terms:
            if terms is None:
                continue
            for role_index, _token_id in terms:
                max_role = max(max_role, int(role_index))
        n_roles = max_role + 1
    else:
        n_roles = int(positions.shape[0])
    if n_roles <= 0:
        failures.append("invalid_role_count")

    normalized_terms: List[Optional[List[Tuple[int, int]]]] = []
    malformed_terms = 0
    role_out_of_bounds = 0
    negative_token_ids = 0
    missing_rows = 0
    empty_rows = 0
    rows_with_mask = 0
    mask_terms = 0
    kind_counts = Counter(None if kind is None else str(kind) for kind in raw_kinds)

    for terms in raw_terms:
        if terms is None:
            missing_rows += 1
            normalized_terms.append(None)
            continue
        try:
            clean_terms = _normalize_terms(terms)
        except (TypeError, ValueError):
            malformed_terms += 1
            normalized_terms.append([])
            continue
        if not clean_terms:
            empty_rows += 1
        row_has_mask = False
        for role_index, token_id in clean_terms:
            if not 0 <= role_index < n_roles:
                role_out_of_bounds += 1
            if token_id < 0:
                negative_token_ids += 1
            if mask_token_id is not None and token_id == mask_token_id:
                mask_terms += 1
                row_has_mask = True
        if row_has_mask:
            rows_with_mask += 1
        normalized_terms.append(clean_terms)

    if missing_rows:
        failures.append("missing_pattern_encoder_terms_rows")
    if malformed_terms:
        failures.append("malformed_pattern_encoder_terms")
    if role_out_of_bounds:
        failures.append("role_index_out_of_bounds")
    if negative_token_ids:
        failures.append("negative_token_ids")
    if empty_rows:
        warnings.append("empty_pattern_encoder_terms_rows")
    if rows_with_mask:
        warnings.append("mask_token_participates_in_row_provenance")

    stats = None
    entropy = {
        "mean_normalized": float("nan"),
        "median_normalized": float("nan"),
        "max_normalized": float("nan"),
        "uniform_row_fraction": float("nan"),
        "uniform_rows": 0,
        "empty_row_fraction": float("nan"),
        "empty_rows": 0,
    }
    count_summary = {
        "role_counts": [],
        "role_fractions": [],
        "role_coverage": 0,
        "entropy": entropy,
    }
    count_degeneracy_reasons: List[str] = []
    count_blockers = (
        "pattern_encoder_terms_row_count_mismatch",
        "malformed_pattern_encoder_terms",
        "role_index_out_of_bounds",
        "invalid_role_count",
    )
    if not any(reason in failures for reason in count_blockers):
        stats = RoleBindingStats.from_pattern_encoder_terms(
            normalized_terms,
            n_roles=n_roles,
            device=device,
            require_complete=False,
        )
        count_summary = _role_summary(stats.counts)
        count_degeneracy_reasons = _degeneracy_reasons(
            entropy=count_summary["entropy"],
            role_coverage=count_summary["role_coverage"],
            role_fractions=count_summary["role_fractions"],
            n_roles=n_roles,
        )
        if weight_source == "count":
            failures.extend(count_degeneracy_reasons)

    geometric_summary = None
    geometric_scores = None
    geometric_degeneracy_reasons: List[str] = []
    geometric_blockers = (
        "pattern_encoder_terms_row_count_mismatch",
        "malformed_pattern_encoder_terms",
        "invalid_role_count",
    )
    if not any(reason in failures for reason in geometric_blockers):
        if positions is None:
            if weight_source == "geometric":
                failures.append("missing_positions_for_geometric_weights")
            else:
                warnings.append("missing_positions_for_geometric_weights")
        else:
            patterns = mem._pattern_matrix()
            role_vectors = [positions[i] for i in range(n_roles)]
            geometric_scores = RoleBindingStats.geometric_row_role_scores(
                mem.substrate,
                patterns,
                role_vectors,
                mode=geometric_mode,
                neighbor_k=geometric_neighbor_k,
            )
            if geometric_temperature is None:
                geometric_weights = geometric_scores + float(geometric_laplace)
                geometric_weights = geometric_weights / geometric_weights.sum(
                    dim=1, keepdim=True,
                ).clamp(min=1e-12)
            else:
                geometric_weights = torch.softmax(
                    geometric_scores / float(geometric_temperature),
                    dim=1,
                )
            geometric_summary = _role_summary(geometric_weights)
            geometric_degeneracy_reasons = _degeneracy_reasons(
                entropy=geometric_summary["entropy"],
                role_coverage=geometric_summary["role_coverage"],
                role_fractions=geometric_summary["role_fractions"],
                n_roles=n_roles,
                prefix="geometric_",
            )
            if weight_source == "geometric":
                failures.extend(geometric_degeneracy_reasons)

    if weight_source == "geometric" and count_degeneracy_reasons:
        warnings.append("count_role_weights_degenerate")
    if weight_source == "count" and geometric_degeneracy_reasons:
        warnings.append("geometric_role_weights_degenerate")

    active_summary = geometric_summary if weight_source == "geometric" else count_summary
    if active_summary is None:
        active_summary = count_summary

    status = "pass" if not failures else "fail"
    return {
        "status": status,
        "snapshot": str(path),
        "failure_reasons": failures,
        "warnings": warnings,
        "n_atoms": n_atoms,
        "n_roles": n_roles,
        "metadata": metadata,
        "weight_source": weight_source,
        "geometric_config": {
            "mode": geometric_mode,
            "neighbor_k": geometric_neighbor_k,
            "laplace": geometric_laplace,
            "temperature": geometric_temperature,
        },
        "row_count_aligned": row_count_aligned,
        "kind_counts": {str(k): int(v) for k, v in kind_counts.items()},
        "term_checks": {
            "missing_rows": missing_rows,
            "empty_rows": empty_rows,
            "malformed_terms": malformed_terms,
            "role_out_of_bounds": role_out_of_bounds,
            "negative_token_ids": negative_token_ids,
            "mask_token_id": mask_token_id,
            "rows_with_mask_token": rows_with_mask,
            "mask_token_terms": mask_terms,
        },
        "role_counts": active_summary["role_counts"],
        "role_fractions": active_summary["role_fractions"],
        "role_coverage": active_summary["role_coverage"],
        "entropy": active_summary["entropy"],
        "count_role_counts": count_summary["role_counts"],
        "count_role_fractions": count_summary["role_fractions"],
        "count_role_coverage": count_summary["role_coverage"],
        "count_entropy": count_summary["entropy"],
        "count_degeneracy_reasons": count_degeneracy_reasons,
        "geometric_role_counts": (
            None if geometric_summary is None else geometric_summary["role_counts"]
        ),
        "geometric_role_fractions": (
            None if geometric_summary is None else geometric_summary["role_fractions"]
        ),
        "geometric_role_coverage": (
            None if geometric_summary is None else geometric_summary["role_coverage"]
        ),
        "geometric_entropy": (
            None if geometric_summary is None else geometric_summary["entropy"]
        ),
        "geometric_degeneracy_reasons": geometric_degeneracy_reasons,
        "geometric_score_range": (
            None if geometric_scores is None else _geometric_score_summary(geometric_scores)
        ),
        "thresholds": {
            "mean_entropy_fail": MEAN_ENTROPY_FAIL_THRESHOLD,
            "uniform_row_entropy": UNIFORM_ROW_ENTROPY_THRESHOLD,
            "uniform_row_fraction_fail": UNIFORM_ROW_FRACTION_FAIL_THRESHOLD,
            "min_role_fraction": MIN_ROLE_FRACTION,
            "empty_row_fraction_fail": EMPTY_ROW_FRACTION_FAIL_THRESHOLD,
        },
    }


def write_report(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 5 M1 Provenance Audit",
        "",
        f"Status: **{payload['status'].upper()}**",
        "",
        f"- Snapshot: `{payload['snapshot']}`",
        f"- Rows: {payload.get('n_atoms')}",
        f"- Roles: {payload.get('n_roles')}",
        f"- Weight source: {payload.get('weight_source')}",
        f"- Row alignment: {payload.get('row_count_aligned')}",
        f"- Role coverage: {payload.get('role_coverage')}/{payload.get('n_roles')}",
    ]
    entropy = payload.get("entropy") or {}
    if entropy:
        lines.extend([
            f"- Mean normalized row entropy: {entropy.get('mean_normalized')}",
            f"- Uniform row fraction: {entropy.get('uniform_row_fraction')}",
        ])
    if payload.get("failure_reasons"):
        lines.extend(["", "## Failure Reasons"])
        lines.extend(f"- {reason}" for reason in payload["failure_reasons"])
    if payload.get("warnings"):
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    lines.extend(["", "## Role Counts"])
    lines.append(json.dumps({
        "role_counts": payload.get("role_counts"),
        "role_fractions": payload.get("role_fractions"),
    }, indent=2))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--weight-source", choices=("count", "geometric"), default="count")
    parser.add_argument("--geometric-mode", default="unbind_density")
    parser.add_argument("--geometric-neighbor-k", type=int, default=8)
    parser.add_argument("--geometric-laplace", type=float, default=1e-6)
    parser.add_argument("--geometric-temperature", type=float, default=0.05)
    args = parser.parse_args()

    payload = audit_snapshot(
        args.snapshot,
        device=args.device,
        mask_token_id=args.mask_token_id,
        weight_source=args.weight_source,
        geometric_mode=args.geometric_mode,
        geometric_neighbor_k=args.geometric_neighbor_k,
        geometric_laplace=args.geometric_laplace,
        geometric_temperature=args.geometric_temperature,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path = (
        Path(args.markdown_output)
        if args.markdown_output is not None
        else out_path.with_suffix(".md")
    )
    write_report(payload, md_path)
    print(f"{payload['status'].upper()} {args.snapshot}")
    if payload["failure_reasons"]:
        print("failure_reasons:", ", ".join(payload["failure_reasons"]))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
