"""Reusable bundle-first scene-memory mechanics for Phase 5'.

This module owns fixed local scene-memory dynamics only. Report scripts keep
artifact loading, SHA/path validation, protocol selection, and payload writing.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch

from energy_memory.phase2.encoding import build_position_vectors
from energy_memory.phase5.natural_source_protocol import (
    fixedpoint_free_shuffle,
    role_derangement,
    role_permutation,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR


EXP44 = importlib.import_module("experiments.44_phase5_prime_bundle_first")

CONDITIONS = {
    "candidate",
    "random_role",
    "deranged_role",
    "shuffled_role",
    "fixedpoint_free_shuffled_role",
    "content_cleanup_positive",
    "bundle_positive",
    "perfect_cue",
}


@dataclass(frozen=True)
class BundleFirstConfig:
    D: int
    N: int
    K_roles: int
    C_codebook: int
    context_roles: int
    n_queries: int
    beta: float
    max_iter: int
    scene_token_weight: float
    cooccurrence: str
    source_name: str


@dataclass(frozen=True)
class BundleFirstResult:
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


@dataclass(frozen=True)
class BundleFirstSeedState:
    seed: int
    fhrr: TorchFHRR
    roles: torch.Tensor
    content: torch.Tensor
    rows: torch.Tensor
    query_plan: List[dict]
    query_context_tokens: torch.Tensor
    scene_matrix: torch.Tensor
    scene_token_weight: float


def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> dict:
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


def build_native_roles(fhrr: TorchFHRR, k_roles: int) -> torch.Tensor:
    return torch.stack(build_position_vectors(fhrr, k_roles), dim=0)


def build_scene_matrix(
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


def build_query_context_tokens(
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


def build_bundle_first_seed_state(
    *,
    seed: int,
    source: dict,
    config: BundleFirstConfig,
    scene_token_weight: float | None = None,
    device: str,
) -> BundleFirstSeedState:
    """Build the fixed scene-memory tensors for one source seed.

    This is construction-only: no candidate/control condition is run and no
    diagnostic result is interpreted here.
    """
    token_weight = (
        config.scene_token_weight
        if scene_token_weight is None
        else float(scene_token_weight)
    )
    if token_weight < 0.0:
        raise ValueError("scene_token_weight must be non-negative")
    rows_raw, query_plan = _source_rows_and_plan(
        source=source,
        seed=seed,
        n_rows=config.N,
        n_queries=config.n_queries,
    )
    fhrr = TorchFHRR(dim=config.D, seed=seed, device=device)
    roles = build_native_roles(fhrr, config.K_roles)
    content = fhrr.random_vectors(config.C_codebook)
    rows = torch.tensor(rows_raw, dtype=torch.long, device=fhrr.device)
    scene_matrix = build_scene_matrix(
        fhrr,
        roles,
        content,
        rows,
        scene_token_weight=token_weight,
    )
    query_context_tokens = build_query_context_tokens(
        fhrr,
        roles,
        content,
        rows,
        query_plan,
    )
    return BundleFirstSeedState(
        seed=seed,
        fhrr=fhrr,
        roles=roles,
        content=content,
        rows=rows,
        query_plan=[dict(item) for item in query_plan],
        query_context_tokens=query_context_tokens,
        scene_matrix=scene_matrix,
        scene_token_weight=token_weight,
    )


def _role_tensor(indices: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([int(role) for role in indices], dtype=torch.long, device=device)


def _role_controls(
    *,
    condition: str,
    seed: int,
    k_roles: int,
    known_role: torch.Tensor,
    query_role: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cue_role = known_role
    unbind_role = query_role
    if condition == "random_role" and k_roles > 1:
        unbind_role = (query_role + 1) % k_roles
    elif condition == "shuffled_role":
        role_shuffle = _role_tensor(role_permutation(seed, k_roles), device)
        cue_role = role_shuffle[known_role]
        unbind_role = role_shuffle[query_role]
    elif condition == "deranged_role":
        derangement = _role_tensor(role_derangement(seed, k_roles), device)
        cue_role = derangement[known_role]
        unbind_role = derangement[query_role]
    elif condition == "fixedpoint_free_shuffled_role":
        no_fixed_shuffle = _role_tensor(fixedpoint_free_shuffle(seed, k_roles), device)
        cue_role = no_fixed_shuffle[known_role]
        unbind_role = no_fixed_shuffle[query_role]
    return cue_role, unbind_role


def _source_rows_and_plan(
    *,
    source: dict,
    seed: int,
    n_rows: int,
    n_queries: int,
) -> tuple[list, list]:
    seed_key = str(seed)
    rows_raw = source["source_rows_by_seed"][seed_key]
    query_plan = source["query_plan_by_seed"][seed_key]
    if len(rows_raw) != n_rows:
        raise ValueError(f"source rows for seed {seed} have len {len(rows_raw)} != {n_rows}")
    if len(query_plan) != n_queries:
        raise ValueError(
            f"query plan for seed {seed} has len {len(query_plan)} != {n_queries}"
        )
    return rows_raw, query_plan


def run_bundle_first_seed_condition(
    *,
    condition: str,
    seed: int,
    source: dict,
    source_path: Path | str,
    source_sha: str,
    config: BundleFirstConfig,
    cue_noise: float,
    scene_token_weight: float | None = None,
    device: str,
) -> BundleFirstResult:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    token_weight = (
        config.scene_token_weight
        if scene_token_weight is None
        else float(scene_token_weight)
    )
    if token_weight < 0.0:
        raise ValueError("scene_token_weight must be non-negative")
    rows_raw, query_plan = _source_rows_and_plan(
        source=source,
        seed=seed,
        n_rows=config.N,
        n_queries=config.n_queries,
    )

    fhrr = TorchFHRR(dim=config.D, seed=seed, device=device)
    roles = build_native_roles(fhrr, config.K_roles)
    content = fhrr.random_vectors(config.C_codebook)
    rows = torch.tensor(rows_raw, dtype=torch.long, device=fhrr.device)
    scene_matrix = build_scene_matrix(
        fhrr,
        roles,
        content,
        rows,
        scene_token_weight=token_weight,
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
    zero_stats = torch.zeros(config.n_queries, device=fhrr.device)

    if condition == "content_cleanup_positive":
        scene_top_index = scene_idx
        scene_entropy = zero_stats
        scene_margin = zero_stats
        content_query = EXP44._perturb_batch(fhrr, content[target_atom], cue_noise)
    elif condition == "bundle_positive":
        scene_state = scene_matrix[scene_idx]
        scene_top_index = scene_idx
        scene_entropy = zero_stats
        scene_margin = zero_stats
        content_query = fhrr.normalize(fhrr.unbind(scene_state, roles[query_role]))
    elif condition == "perfect_cue":
        scene_state, scene_top_index, scene_entropy, scene_margin = (
            EXP44._batched_hopfield_retrieve(
                fhrr,
                scene_matrix,
                scene_matrix[scene_idx],
                beta=config.beta,
                max_iter=config.max_iter,
            )
        )
        content_query = fhrr.normalize(fhrr.unbind(scene_state, roles[query_role]))
    else:
        cue_role, unbind_role = _role_controls(
            condition=condition,
            seed=seed,
            k_roles=config.K_roles,
            known_role=known_role,
            query_role=query_role,
            device=fhrr.device,
        )
        known_atom = rows[scene_idx, known_role]
        cue = roles[cue_role] * content[known_atom]
        query_tokens = build_query_context_tokens(fhrr, roles, content, rows, query_plan)
        cue = fhrr.normalize(cue + token_weight * query_tokens)
        cue = EXP44._perturb_batch(fhrr, cue, cue_noise)
        scene_state, scene_top_index, scene_entropy, scene_margin = (
            EXP44._batched_hopfield_retrieve(
                fhrr,
                scene_matrix,
                cue,
                beta=config.beta,
                max_iter=config.max_iter,
            )
        )
        content_query = fhrr.normalize(fhrr.unbind(scene_state, roles[unbind_role]))

    scene_tix = int((scene_top_index == scene_idx).sum().detach().cpu())
    content_state, content_top_index, content_entropy, content_margin = (
        EXP44._batched_hopfield_retrieve(
            fhrr,
            content,
            content_query,
            beta=config.beta,
            max_iter=config.max_iter,
        )
    )
    content_tix = int((content_top_index == target_atom).sum().detach().cpu())
    pred = torch.argmax((content_state @ content.conj().T).real / content.shape[1], dim=1)
    n_correct = int((pred == target_atom).sum().detach().cpu())

    return BundleFirstResult(
        condition=condition,
        D=config.D,
        N=config.N,
        K_roles=config.K_roles,
        cue_noise=cue_noise,
        scene_token_weight=token_weight,
        source_name=config.source_name,
        context_roles=config.context_roles,
        cooccurrence=config.cooccurrence,
        seed=seed,
        n_queries=config.n_queries,
        n_correct=n_correct,
        scene_tix=scene_tix,
        content_tix=content_tix,
        scene_entropy=float(scene_entropy.mean().detach().cpu()),
        content_entropy=float(content_entropy.mean().detach().cpu()),
        scene_margin=float(scene_margin.mean().detach().cpu()),
        content_margin=float(content_margin.mean().detach().cpu()),
        source_rows_available=len(rows_raw),
        source_rows_used=config.N,
        source_rows_invalid=0,
        source_rows_too_short=0,
        source_artifact_path=str(source_path),
        source_artifact_sha256=source_sha,
    )


def leave_one_seed_out(results: Sequence[BundleFirstResult]) -> List[dict]:
    if len(results) <= 1:
        return []
    out = []
    for held_out in results:
        kept = [row for row in results if row.seed != held_out.seed]
        n_total = sum(row.n_queries for row in kept)
        n_correct = sum(row.n_correct for row in kept)
        out.append({
            "held_out_seed": held_out.seed,
            "top1": n_correct / n_total if n_total else 0.0,
            "n_total": n_total,
            "n_correct": n_correct,
        })
    return out


def aggregate_bundle_first_results(results: Sequence[BundleFirstResult]) -> dict:
    if not results:
        raise ValueError("cannot aggregate empty bundle-first results")
    n_total = sum(r.n_queries for r in results)
    n_correct = sum(r.n_correct for r in results)
    ci = wilson_ci(n_correct, n_total)
    scene_tix = sum(r.scene_tix for r in results)
    content_tix = sum(r.content_tix for r in results)
    loo = leave_one_seed_out(results)
    loo_vals = [float(row["top1"]) for row in loo]
    return {
        "top1_mean": ci["mean"],
        "wilson_lo": ci["lo"],
        "wilson_hi": ci["hi"],
        "n_total": n_total,
        "n_correct": n_correct,
        "per_seed_top1": [r.top1 for r in results],
        "per_seed": [
            {
                "seed": r.seed,
                "top1": r.top1,
                "n_correct": r.n_correct,
                "n_queries": r.n_queries,
                "scene_tix": r.scene_tix,
                "content_tix": r.content_tix,
            }
            for r in results
        ],
        "leave_one_seed_out_top1": loo,
        "leave_one_seed_out_top1_min": min(loo_vals) if loo_vals else None,
        "leave_one_seed_out_top1_max": max(loo_vals) if loo_vals else None,
        "scene_tix": scene_tix,
        "content_tix": content_tix,
        "scene_tix_rate": scene_tix / n_total if n_total else 0.0,
        "content_tix_rate": content_tix / n_total if n_total else 0.0,
        "mean_scene_entropy": sum(r.scene_entropy for r in results) / len(results),
        "mean_content_entropy": sum(r.content_entropy for r in results) / len(results),
        "mean_scene_margin": sum(r.scene_margin for r in results) / len(results),
        "mean_content_margin": sum(r.content_margin for r in results) / len(results),
        "source_rows_available": [r.source_rows_available for r in results],
        "source_rows_used": [r.source_rows_used for r in results],
        "source_rows_invalid": [r.source_rows_invalid for r in results],
        "source_rows_too_short": [r.source_rows_too_short for r in results],
        "source_artifact_paths": sorted({r.source_artifact_path for r in results}),
        "source_artifact_sha256": sorted({r.source_artifact_sha256 for r in results}),
    }
