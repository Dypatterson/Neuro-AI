"""Phase 5′ bundle-first structural-memory diagnostic harness.

This is not a Phase 5 graduation experiment. It tests the Phase 5′
bundle-first architecture proposed after Reports 066/067:

  scene-MHN identification -> role unbinding -> content-MHN cleanup

All mechanisms are static geometry/energy/algebra. Controls and diagnostics are
reported; no condition is selected adaptively inside the run.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass(frozen=True)
class WilsonCI:
    mean: float
    lo: float
    hi: float
    n: int


@dataclass(frozen=True)
class CellResult:
    condition: str
    D: int
    N: int
    K_roles: int
    cue_noise: float
    scene_token: bool
    scene_token_weight: float
    scene_token_pool_size: int
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

    @property
    def top1(self) -> float:
        return self.n_correct / self.n_queries if self.n_queries else 0.0


def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> WilsonCI:
    if n_total == 0:
        return WilsonCI(0.0, 0.0, 0.0, 0)
    p = n_success / n_total
    denom = 1.0 + z * z / n_total
    center = (p + z * z / (2 * n_total)) / denom
    half = (
        z
        * math.sqrt(p * (1.0 - p) / n_total + z * z / (4.0 * n_total * n_total))
        / denom
    )
    return WilsonCI(mean=p, lo=max(0.0, center - half), hi=min(1.0, center + half), n=n_total)


def _filler_indices(
    N: int,
    K_roles: int,
    C_codebook: int,
    *,
    cooccurrence: str,
    generator: torch.Generator,
) -> List[List[int]]:
    if cooccurrence == "uniform":
        return torch.randint(0, C_codebook, (N, K_roles), generator=generator).tolist()
    if cooccurrence != "skewed":
        raise ValueError("cooccurrence must be 'uniform' or 'skewed'")
    bucket = max(2, min(64, C_codebook // max(1, K_roles)))
    rows: List[List[int]] = []
    for scene_idx in range(N):
        row: List[int] = []
        for role in range(K_roles):
            base = (role * bucket + scene_idx % bucket) % C_codebook
            offset = int(torch.randint(0, bucket, (1,), generator=generator).item())
            row.append((base + offset) % C_codebook)
        rows.append(row)
    return rows


def _build_scene_bundles(
    fhrr: TorchFHRR,
    roles: torch.Tensor,
    content: torch.Tensor,
    filler_indices: Sequence[Sequence[int]],
    *,
    scene_tokens: Optional[torch.Tensor] = None,
    scene_token_weight: float = 0.5,
) -> List[torch.Tensor]:
    bundles: List[torch.Tensor] = []
    for scene_idx, row in enumerate(filler_indices):
        terms = [roles[role] * content[int(atom)] for role, atom in enumerate(row)]
        if scene_tokens is not None:
            terms.append(scene_token_weight * scene_tokens[scene_idx])
        bundles.append(fhrr.bundle(terms))
    return bundles


def _scene_tokens(
    fhrr: TorchFHRR,
    N: int,
    *,
    enabled: bool,
    pool_size: int,
) -> Optional[torch.Tensor]:
    if not enabled:
        return None
    if pool_size < 0:
        raise ValueError("scene_token_pool_size must be non-negative")
    if pool_size == 0 or pool_size >= N:
        return fhrr.random_vectors(N)
    pool = fhrr.random_vectors(pool_size)
    indices = torch.arange(N, device=fhrr.device) % pool_size
    return pool[indices]


def _query_plan(
    N: int,
    K_roles: int,
    n_queries: int,
    *,
    generator: torch.Generator,
) -> List[Tuple[int, int, int]]:
    plan: List[Tuple[int, int, int]] = []
    for _ in range(n_queries):
        scene_idx = int(torch.randint(0, N, (1,), generator=generator).item())
        known_role = int(torch.randint(0, K_roles, (1,), generator=generator).item())
        query_role = known_role
        if K_roles > 1:
            while query_role == known_role:
                query_role = int(torch.randint(0, K_roles, (1,), generator=generator).item())
        plan.append((scene_idx, known_role, query_role))
    return plan


def _role_permutation(K_roles: int, *, generator: torch.Generator) -> List[int]:
    if K_roles == 1:
        return [0]
    perm = torch.randperm(K_roles, generator=generator).tolist()
    if all(i == p for i, p in enumerate(perm)):
        perm = perm[1:] + perm[:1]
    return perm


def _perturb_batch(fhrr: TorchFHRR, vectors: torch.Tensor, noise: float) -> torch.Tensor:
    if noise <= 0.0:
        return vectors
    if noise < 0.0:
        raise ValueError("noise must be non-negative")
    phase = torch.randn(vectors.shape, generator=fhrr.generator, device="cpu") * noise
    phase = phase.to(fhrr.device)
    return vectors * torch.polar(torch.ones_like(phase, device=fhrr.device), phase)


def _batched_hopfield_retrieve(
    fhrr: TorchFHRR,
    patterns: torch.Tensor,
    queries: torch.Tensor,
    *,
    beta: float,
    max_iter: int = 10,
    tol: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched softmax Hopfield settling for the MQAR diagnostic.

    The scalar TorchHopfieldMemory path is intentionally general-purpose, but
    this experiment is query-parallel and was spending most of its wall time in
    Python loops. This helper preserves the same fixed iterative softmax update
    while running all queries for a cell as GPU matrix multiplies.
    """
    if patterns.numel() == 0:
        raise ValueError("cannot retrieve from an empty pattern matrix")
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    state = queries.to(fhrr.device)
    patterns = patterns.to(fhrr.device)
    frozen = torch.zeros(state.shape[0], dtype=torch.bool, device=fhrr.device)
    final_state = state
    prev_energy: Optional[torch.Tensor] = None

    for _ in range(max_iter):
        scores = (state @ patterns.conj().T).real / patterns.shape[1]
        energy = -torch.logsumexp(beta * scores, dim=1) / beta
        weights = torch.softmax(beta * scores, dim=1)
        next_state = fhrr.normalize(weights.to(patterns.dtype) @ patterns)
        if prev_energy is not None:
            converged_now = ((energy - prev_energy).abs() < tol) & (~frozen)
            final_state = torch.where(converged_now[:, None], next_state, final_state)
            frozen = frozen | converged_now
        prev_energy = energy
        state = next_state

    final_state = torch.where(frozen[:, None], final_state, state)
    final_scores = (final_state @ patterns.conj().T).real / patterns.shape[1]
    final_weights = torch.softmax(beta * final_scores, dim=1)
    top_index = torch.argmax(final_scores, dim=1)

    if final_weights.shape[1] > 1:
        safe = final_weights.clamp_min(1e-12)
        entropy = -(safe * safe.log()).sum(dim=1) / math.log(final_weights.shape[1])
        top2 = torch.topk(final_scores, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
    else:
        entropy = torch.zeros(final_weights.shape[0], device=fhrr.device)
        margin = torch.zeros(final_weights.shape[0], device=fhrr.device)

    return final_state, top_index, entropy, margin


def run_cell(
    *,
    condition: str,
    D: int,
    N: int,
    K_roles: int,
    cue_noise: float,
    scene_token: bool,
    scene_token_weight: float,
    scene_token_pool_size: int,
    cooccurrence: str,
    seed: int,
    n_queries: int,
    beta: float,
    C_codebook: int,
    device: str,
) -> CellResult:
    if K_roles <= 0:
        raise ValueError("K_roles must be positive")
    if scene_token_weight < 0.0:
        raise ValueError("scene_token_weight must be non-negative")
    if condition not in {
        "candidate",
        "random_role",
        "shuffled_role",
        "perfect_cue",
        "bundle_positive",
        "content_cleanup_positive",
    }:
        raise ValueError(f"unknown condition: {condition}")

    fhrr = TorchFHRR(dim=D, seed=seed, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed * 1009 + N * 37 + K_roles)
    roles = fhrr.random_vectors(K_roles)
    content = fhrr.random_vectors(C_codebook)
    scene_tokens = _scene_tokens(
        fhrr,
        N,
        enabled=scene_token,
        pool_size=scene_token_pool_size,
    )
    fillers = _filler_indices(
        N,
        K_roles,
        C_codebook,
        cooccurrence=cooccurrence,
        generator=generator,
    )
    scene_bundles = _build_scene_bundles(
        fhrr,
        roles,
        content,
        fillers,
        scene_tokens=scene_tokens,
        scene_token_weight=scene_token_weight,
    )
    scene_matrix = torch.stack(scene_bundles, dim=0)
    filler_tensor = torch.tensor(fillers, dtype=torch.long, device=fhrr.device)

    role_shuffle = _role_permutation(K_roles, generator=generator)
    role_shuffle_tensor = torch.tensor(role_shuffle, dtype=torch.long, device=fhrr.device)
    plan = _query_plan(N, K_roles, n_queries, generator=generator)
    plan_tensor = torch.tensor(plan, dtype=torch.long, device=fhrr.device)
    scene_idx = plan_tensor[:, 0]
    known_role = plan_tensor[:, 1]
    query_role = plan_tensor[:, 2]
    target_atom = filler_tensor[scene_idx, query_role]

    cue_role = known_role
    unbind_role = query_role
    if condition == "random_role" and K_roles > 1:
        unbind_role = (query_role + 1) % K_roles
    elif condition == "shuffled_role":
        cue_role = role_shuffle_tensor[known_role]
        unbind_role = role_shuffle_tensor[query_role]

    known_atom = filler_tensor[scene_idx, known_role]
    cue = roles[cue_role] * content[known_atom]
    if scene_tokens is not None:
        cue = cue + scene_token_weight * scene_tokens[scene_idx]
    cue = fhrr.normalize(cue)
    cue = _perturb_batch(fhrr, cue, cue_noise)

    scene_query = scene_matrix[scene_idx] if condition == "perfect_cue" else cue

    zero_stats = torch.zeros(n_queries, device=fhrr.device)
    if condition == "bundle_positive":
        scene_state = scene_matrix[scene_idx]
        scene_top_index = scene_idx
        scene_entropy = zero_stats
        scene_margin = zero_stats
    elif condition == "content_cleanup_positive":
        scene_state = None
        scene_top_index = scene_idx
        scene_entropy = zero_stats
        scene_margin = zero_stats
    else:
        scene_state, scene_top_index, scene_entropy, scene_margin = _batched_hopfield_retrieve(
            fhrr,
            scene_matrix,
            scene_query,
            beta=beta,
        )

    scene_tix = int((scene_top_index == scene_idx).sum().detach().cpu())

    if condition == "content_cleanup_positive":
        content_query = content[target_atom]
        content_query = _perturb_batch(fhrr, content_query, cue_noise)
    else:
        assert scene_state is not None
        content_query = fhrr.normalize(fhrr.unbind(scene_state, roles[unbind_role]))

    content_state, content_top_index, content_entropy, content_margin = _batched_hopfield_retrieve(
        fhrr,
        content,
        content_query,
        beta=beta,
    )
    content_tix = int((content_top_index == target_atom).sum().detach().cpu())

    pred = torch.argmax((content_state @ content.conj().T).real / content.shape[1], dim=1)
    n_correct = int((pred == target_atom).sum().detach().cpu())

    return CellResult(
        condition=condition,
        D=D,
        N=N,
        K_roles=K_roles,
        cue_noise=cue_noise,
        scene_token=scene_token,
        scene_token_weight=scene_token_weight,
        scene_token_pool_size=scene_token_pool_size,
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
    )


def _aggregate(cell: Sequence[CellResult]) -> dict:
    n_total = sum(r.n_queries for r in cell)
    n_correct = sum(r.n_correct for r in cell)
    ci = wilson_ci(n_correct, n_total)
    scene_tix = sum(r.scene_tix for r in cell)
    content_tix = sum(r.content_tix for r in cell)
    return {
        "top1_mean": ci.mean,
        "wilson_lo": ci.lo,
        "wilson_hi": ci.hi,
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ds", nargs="+", type=int, default=[4096])
    parser.add_argument("--Ns", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--K_roles", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--cue_noise", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.15])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 11, 23])
    parser.add_argument("--n_queries", type=int, default=512)
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--C_codebook", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "candidate",
            "random_role",
            "shuffled_role",
            "perfect_cue",
            "bundle_positive",
            "content_cleanup_positive",
        ],
    )
    parser.add_argument(
        "--scene_token",
        nargs="+",
        type=int,
        default=[0],
        help="0/1 scene-token condition flags.",
    )
    parser.add_argument(
        "--scene_token_weight",
        nargs="+",
        type=float,
        default=[0.5],
        help="Scene-token bundle/cue weights. Effective only when scene_token=1.",
    )
    parser.add_argument(
        "--scene_token_pool_size",
        nargs="+",
        type=int,
        default=[0],
        help=(
            "Number of distinct scene-token anchors to reuse. 0 means one unique "
            "anchor per scene; 1 means all scenes share one anchor."
        ),
    )
    parser.add_argument(
        "--cooccurrence",
        nargs="+",
        choices=["uniform", "skewed"],
        default=["uniform"],
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_bundle_first/results.json",
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

    print(
        f"device={device} Ds={args.Ds} Ns={args.Ns} K_roles={args.K_roles} "
        f"cue_noise={args.cue_noise} seeds={args.seeds} n_queries={args.n_queries} "
        f"conditions={args.conditions} scene_token={args.scene_token} "
        f"scene_token_weight={args.scene_token_weight} "
        f"scene_token_pool_size={args.scene_token_pool_size} "
        f"cooccurrence={args.cooccurrence}"
    )

    raw: List[CellResult] = []
    aggregates: Dict[str, dict] = {}

    for condition in args.conditions:
        for D in args.Ds:
            for K in args.K_roles:
                for N in args.Ns:
                    for noise in args.cue_noise:
                        for scene_token_flag in args.scene_token:
                            weights = args.scene_token_weight if scene_token_flag else [0.0]
                            pools = args.scene_token_pool_size if scene_token_flag else [0]
                            for token_weight in weights:
                                for token_pool_size in pools:
                                    for cooccurrence in args.cooccurrence:
                                        key = (
                                            f"{condition}|D={D}|K={K}|N={N}|noise={noise}|"
                                            f"scene_token={int(scene_token_flag)}|"
                                            f"token_weight={token_weight}|"
                                            f"token_pool={token_pool_size}|"
                                            f"cooc={cooccurrence}"
                                        )
                                        cell: List[CellResult] = []
                                        for seed in args.seeds:
                                            result = run_cell(
                                                condition=condition,
                                                D=D,
                                                N=N,
                                                K_roles=K,
                                                cue_noise=noise,
                                                scene_token=bool(scene_token_flag),
                                                scene_token_weight=token_weight,
                                                scene_token_pool_size=token_pool_size,
                                                cooccurrence=cooccurrence,
                                                seed=seed,
                                                n_queries=args.n_queries,
                                                beta=args.beta,
                                                C_codebook=args.C_codebook,
                                                device=device,
                                            )
                                            cell.append(result)
                                            raw.append(result)
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
            "phase": "5-prime diagnostic",
            "not_graduation": True,
            "anti_homunculus": (
                "static conditions and controls; no metric-triggered routing "
                "or best-of-N selection"
            ),
        },
        "config": {
            "Ds": args.Ds,
            "Ns": args.Ns,
            "K_roles": args.K_roles,
            "cue_noise": args.cue_noise,
            "seeds": args.seeds,
            "n_queries": args.n_queries,
            "beta": args.beta,
            "C_codebook": args.C_codebook,
            "conditions": args.conditions,
            "scene_token": args.scene_token,
            "scene_token_weight": args.scene_token_weight,
            "scene_token_pool_size": args.scene_token_pool_size,
            "cooccurrence": args.cooccurrence,
            "device": device,
        },
        "aggregates": aggregates,
        "raw": [
            {
                "condition": r.condition,
                "D": r.D,
                "N": r.N,
                "K_roles": r.K_roles,
                "cue_noise": r.cue_noise,
                "scene_token": r.scene_token,
                "scene_token_weight": r.scene_token_weight,
                "scene_token_pool_size": r.scene_token_pool_size,
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
            }
            for r in raw
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
