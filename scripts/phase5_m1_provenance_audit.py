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
    if n_roles <= 1:
        return {
            "mean_normalized": float("nan"),
            "median_normalized": float("nan"),
            "max_normalized": float("nan"),
            "uniform_row_fraction": float("nan"),
            "uniform_rows": 0,
        }
    totals = counts.sum(dim=1, keepdim=True)
    nonempty = totals.squeeze(1) > 0
    probs = torch.where(totals > 0, counts / totals.clamp(min=1e-12), counts)
    safe = probs.clamp(min=1e-12)
    entropy = -(safe * safe.log()).sum(dim=1) / math.log(n_roles)
    entropy = torch.where(nonempty, entropy, torch.zeros_like(entropy))
    values = [float(x) for x in entropy.detach().cpu().tolist()]
    uniform_rows = sum(1 for x in values if x >= UNIFORM_ROW_ENTROPY_THRESHOLD)
    return {
        "mean_normalized": sum(values) / len(values) if values else float("nan"),
        "median_normalized": _quantile(values, 0.5),
        "max_normalized": max(values) if values else float("nan"),
        "uniform_row_fraction": uniform_rows / len(values) if values else float("nan"),
        "uniform_rows": uniform_rows,
    }


def audit_snapshot(
    snapshot_path: str | Path,
    *,
    device: str = "cpu",
    mask_token_id: Optional[int] = None,
) -> Dict[str, Any]:
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
    }
    role_counts: List[float] = []
    role_fractions: List[float] = []
    role_coverage = 0
    if not any(
        reason in failures
        for reason in (
            "pattern_encoder_terms_row_count_mismatch",
            "malformed_pattern_encoder_terms",
            "role_index_out_of_bounds",
            "invalid_role_count",
        )
    ):
        stats = RoleBindingStats.from_pattern_encoder_terms(
            normalized_terms,
            n_roles=n_roles,
            device=device,
            require_complete=False,
        )
        role_total_tensor = stats.counts.sum(dim=0)
        total_terms = float(role_total_tensor.sum().detach().cpu())
        role_counts = [float(x) for x in role_total_tensor.detach().cpu().tolist()]
        role_fractions = [
            (count / total_terms if total_terms > 0.0 else 0.0)
            for count in role_counts
        ]
        role_coverage = sum(1 for count in role_counts if count > 0.0)
        entropy = _entropy_summary(stats.counts)

        if entropy["mean_normalized"] > MEAN_ENTROPY_FAIL_THRESHOLD:
            failures.append("mean_role_entropy_degenerate")
        if entropy["uniform_row_fraction"] >= UNIFORM_ROW_FRACTION_FAIL_THRESHOLD:
            failures.append("uniform_role_rows_degenerate")
        if role_coverage < n_roles or any(
            fraction <= MIN_ROLE_FRACTION for fraction in role_fractions
        ):
            failures.append("near_zero_role_utilization")

    status = "pass" if not failures else "fail"
    return {
        "status": status,
        "snapshot": str(path),
        "failure_reasons": failures,
        "warnings": warnings,
        "n_atoms": n_atoms,
        "n_roles": n_roles,
        "metadata": metadata,
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
        "role_counts": role_counts,
        "role_fractions": role_fractions,
        "role_coverage": role_coverage,
        "entropy": entropy,
        "thresholds": {
            "mean_entropy_fail": MEAN_ENTROPY_FAIL_THRESHOLD,
            "uniform_row_entropy": UNIFORM_ROW_ENTROPY_THRESHOLD,
            "uniform_row_fraction_fail": UNIFORM_ROW_FRACTION_FAIL_THRESHOLD,
            "min_role_fraction": MIN_ROLE_FRACTION,
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
    args = parser.parse_args()

    payload = audit_snapshot(
        args.snapshot, device=args.device, mask_token_id=args.mask_token_id,
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
