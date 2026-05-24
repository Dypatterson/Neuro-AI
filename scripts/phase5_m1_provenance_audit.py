"""Audit Phase 5 M1 row-domain provenance in substrate snapshots.

This script checks whether a frozen Phase 4 snapshot carries per-pattern
encoder provenance that can be converted into M1 row-role weights over the
exact pattern rows M1 retrieves from. Old snapshots are expected to fail with
``missing_pattern_encoder_terms``.
"""

from __future__ import annotations

import argparse
import hashlib
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
from energy_memory.phase2.persistence import load_codebook  # noqa: E402
from energy_memory.phase5.m1_role_energy import RoleBindingStats  # noqa: E402
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402


MEAN_ENTROPY_FAIL_THRESHOLD = 0.95
UNIFORM_ROW_ENTROPY_THRESHOLD = 0.99
UNIFORM_ROW_FRACTION_FAIL_THRESHOLD = 0.95
MIN_ROLE_FRACTION = 1e-6
EMPTY_ROW_FRACTION_FAIL_THRESHOLD = 0.5
CODEBOOK_REQUIRED_MODES = {"codebook_prior_density", "codebook_prior"}


def _canonical_geometric_mode(mode: str) -> str:
    aliases = {
        "per_role_pool": "same_role_filler_density",
        "same_role_pool": "same_role_filler_density",
        "codebook_prior": "codebook_prior_density",
    }
    return aliases.get(mode, mode)


def _evidence_scope(metadata: Dict[str, Any]) -> Dict[str, Any]:
    seed = metadata.get("seed")
    try:
        is_seed17 = seed is not None and int(seed) == 17
    except (TypeError, ValueError):
        is_seed17 = False
    if is_seed17:
        note = (
            "Seed 17 is wiring/provenance/degen smoke only; "
            "it is not representative Phase 5 evidence."
        )
    else:
        note = (
            "Single-snapshot audit output is a wiring/provenance diagnostic, "
            "not representative Phase 5 evidence."
        )
    return {
        "representative_phase5_evidence": False,
        "note": note,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _codebook_location(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    repo_root = REPO_ROOT.resolve()
    relpath = (
        str(resolved.relative_to(repo_root))
        if _is_relative_to(resolved, repo_root)
        else None
    )
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "relpath_from_repo_root": relpath,
        "inside_repo": relpath is not None,
    }


def _codebook_identity(path: Path, sha256: str) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "sha256": sha256,
        "basename": resolved.name,
        "size_bytes": int(resolved.stat().st_size),
    }


def _looks_like_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _normalize_registry_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    sha256 = entry.get("sha256")
    if not _looks_like_sha256(sha256):
        return None
    normalized = dict(entry)
    normalized["sha256"] = str(sha256).lower()
    return normalized


def _load_codebook_registry(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_entries: List[Any]
    if isinstance(payload, list):
        raw_entries = payload
    elif isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        raw_entries = payload["entries"]
    elif isinstance(payload, dict) and isinstance(payload.get("codebooks"), list):
        raw_entries = payload["codebooks"]
    elif isinstance(payload, dict):
        raw_entries = []
        for key, value in payload.items():
            if not _looks_like_sha256(key):
                continue
            if isinstance(value, dict):
                entry = dict(value)
                entry.setdefault("sha256", key)
            else:
                entry = {"sha256": key, "value": value}
            raw_entries.append(entry)
    else:
        raise ValueError("codebook registry must be a list or object")

    entries: Dict[str, Dict[str, Any]] = {}
    for raw_entry in raw_entries:
        entry = _normalize_registry_entry(raw_entry)
        if entry is not None:
            entries[entry["sha256"]] = entry
    if not entries:
        raise ValueError("codebook registry has no valid sha256 entries")
    return entries


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
    codebook_path: Optional[str | Path] = None,
    codebook_registry_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    if weight_source not in {"count", "geometric"}:
        raise ValueError("weight_source must be 'count' or 'geometric'")
    if geometric_temperature is not None and geometric_temperature <= 0.0:
        raise ValueError("geometric_temperature must be positive")
    active_geometric_mode = _canonical_geometric_mode(geometric_mode)
    path = Path(snapshot_path)
    mem, cons, info = _load_snapshot(path, device=device)
    n_atoms = mem.stored_count
    metadata = info.get("metadata") or {}
    evidence_scope = _evidence_scope(metadata)
    positions = info.get("positions")
    raw_terms = info.get("pattern_encoder_terms")
    raw_kinds = info.get("pattern_encoder_term_kinds")
    codebook_arg_path = Path(codebook_path) if codebook_path is not None else None
    codebook_registry_arg_path = (
        Path(codebook_registry_path) if codebook_registry_path is not None else None
    )
    geometric_config: Dict[str, Any] = {
        "mode": geometric_mode,
        "active_mode": active_geometric_mode,
        "neighbor_k": geometric_neighbor_k,
        "laplace": geometric_laplace,
        "temperature": geometric_temperature,
        "codebook_path": None if codebook_arg_path is None else str(codebook_arg_path),
        "codebook_shape": None,
        "codebook_dtype": None,
        "codebook_fingerprint": None,
        "codebook_identity": None,
        "codebook_location": (
            None if codebook_arg_path is None else _codebook_location(codebook_arg_path)
        ),
        "codebook_registry": {
            "path": (
                None
                if codebook_registry_arg_path is None
                else str(codebook_registry_arg_path)
            ),
            "matched": None,
            "entry": None,
        },
    }
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
            "evidence_scope": evidence_scope,
            "weight_source": weight_source,
            "geometric_config": geometric_config,
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
            reference_codebook = None
            codebook_problem = None
            codebook_registry = None
            codebook_registry_problem = None
            if active_geometric_mode in CODEBOOK_REQUIRED_MODES:
                codebook_location = geometric_config.get("codebook_location")
                if (
                    codebook_location is not None
                    and not codebook_location.get("inside_repo", False)
                ):
                    warnings.append("codebook_path_outside_repo")
                if weight_source == "geometric" and codebook_registry_arg_path is None:
                    warnings.append("codebook_registry_not_supplied")
                elif codebook_registry_arg_path is not None:
                    try:
                        codebook_registry = _load_codebook_registry(
                            codebook_registry_arg_path
                        )
                    except (OSError, json.JSONDecodeError, ValueError) as exc:
                        codebook_registry_problem = "codebook_registry_load_failed"
                        warnings.append("codebook_registry_load_failed")
                        warnings.append(f"codebook_registry_load_error: {exc}")
                if codebook_arg_path is None:
                    codebook_problem = "codebook_required_for_mode"
                else:
                    try:
                        reference_codebook = load_codebook(codebook_arg_path, device=device)
                    except (FileNotFoundError, RuntimeError, ValueError) as exc:
                        codebook_problem = "codebook_load_failed"
                        warnings.append(f"codebook_load_error: {exc}")
                    else:
                        geometric_config["codebook_shape"] = [
                            int(dim) for dim in reference_codebook.shape
                        ]
                        geometric_config["codebook_dtype"] = str(reference_codebook.dtype)
                        codebook_sha256 = _sha256_file(codebook_arg_path)
                        geometric_config["codebook_fingerprint"] = {
                            "sha256": codebook_sha256,
                        }
                        geometric_config["codebook_identity"] = _codebook_identity(
                            codebook_arg_path,
                            codebook_sha256,
                        )
                        if codebook_registry is not None:
                            registry_entry = codebook_registry.get(codebook_sha256)
                            if registry_entry is None:
                                warnings.append("codebook_not_in_registry")
                                geometric_config["codebook_registry"]["matched"] = False
                            else:
                                geometric_config["codebook_registry"]["matched"] = True
                                geometric_config["codebook_registry"]["entry"] = (
                                    registry_entry
                                )
                        if (
                            reference_codebook.ndim != 2
                            or reference_codebook.shape[1] != patterns.shape[1]
                        ):
                            codebook_problem = "codebook_dim_mismatch"
                        elif reference_codebook.shape[0] == 0:
                            codebook_problem = "codebook_empty"

            if (
                codebook_registry_problem is not None
                and weight_source == "geometric"
            ):
                failures.append(codebook_registry_problem)
            if codebook_problem is not None:
                if weight_source == "geometric":
                    failures.append(codebook_problem)
                else:
                    warnings.append(codebook_problem)
            else:
                try:
                    geometric_scores = RoleBindingStats.geometric_row_role_scores(
                        mem.substrate,
                        patterns,
                        role_vectors,
                        mode=active_geometric_mode,
                        neighbor_k=geometric_neighbor_k,
                        reference_codebook=reference_codebook,
                    )
                except ValueError as exc:
                    if weight_source == "geometric":
                        failures.append(str(exc))
                    else:
                        warnings.append(str(exc))
                else:
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
        "evidence_scope": evidence_scope,
        "weight_source": weight_source,
        "geometric_config": geometric_config,
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
    evidence_scope = payload.get("evidence_scope") or {}
    if evidence_scope:
        lines.append(
            f"- Representative Phase 5 evidence: "
            f"{evidence_scope.get('representative_phase5_evidence')}"
        )
        lines.append(f"- Evidence scope: {evidence_scope.get('note')}")
    geometric_config = payload.get("geometric_config") or {}
    if geometric_config:
        lines.append(
            f"- Geometric mode: {geometric_config.get('active_mode')}"
        )
        if geometric_config.get("codebook_path"):
            lines.append(f"- Codebook: `{geometric_config.get('codebook_path')}`")
        codebook_identity = geometric_config.get("codebook_identity") or {}
        if codebook_identity:
            lines.append(f"- Codebook SHA-256: `{codebook_identity.get('sha256')}`")
            lines.append(
                f"- Codebook size bytes: {codebook_identity.get('size_bytes')}"
            )
        codebook_location = geometric_config.get("codebook_location") or {}
        if codebook_location:
            lines.append(
                "- Codebook repo-relative path: "
                f"`{codebook_location.get('relpath_from_repo_root')}`"
            )
        codebook_registry = geometric_config.get("codebook_registry") or {}
        if codebook_registry.get("path"):
            lines.append(f"- Codebook registry: `{codebook_registry.get('path')}`")
            lines.append(
                f"- Codebook registry match: {codebook_registry.get('matched')}"
            )
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
    parser.add_argument("--codebook", default=None)
    parser.add_argument("--codebook-registry", default=None)
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
        codebook_path=args.codebook,
        codebook_registry_path=args.codebook_registry,
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
