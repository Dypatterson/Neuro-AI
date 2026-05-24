"""Spike D1+D3 local smoke: pseudo-inverse storage swap and cross-K softmax
retrieval, measured on local Phase 5 substrate snapshots.

This is a SMOKE test (n=3 seeds, n=30 cues per seed), not the n=10×100
control matrix that would inform a graduation claim. The purpose is to
answer the binary question for each mechanism:

  Does it materially move `hit_role` and `rank_role` above the
  report-061 baseline (hit_role ≈ 0.003, rank_role ≈ 443/489)?

If yes → scope a Colab n=10×100 run as the next step.
If no  → document the null result and pivot.

Three retrieval conditions, same role-binding cue set, K=2 branches each
(role-target prior vs content-distractor prior):

  baseline  — standard MHN softmax settling (per-branch independent)
  D1        — linear pseudo-inverse settling: s_{t+1} = X (XᵀX + λI)⁻¹ Xᵀ s_t
              (Kymn/Stewart 2022 storage-rule swap; no softmax)
  D3        — MHN settling with ADDITIVE cross-K + per-branch update
              s_k ← Σ_p [(1-μ)·π_k(p) + μ·α_k(p)] · x_p
              where π_k = softmax_p, α_k = softmax_k (per the 2026-05-24
              Lyapunov analytical pass; multiplicative form is non-gradient
              and is NOT implemented here).

Substrate-pure metrics only. No readouts. Active phase: 5. Headline
metric per phase-5-unified-design.md:269-292.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import importlib.util  # noqa: E402

_EXP40_PATH = REPO_ROOT / "experiments" / "40_phase5_branching.py"
_spec = importlib.util.spec_from_file_location("experiments_40", str(_EXP40_PATH))
_exp40 = importlib.util.module_from_spec(_spec)
sys.modules["experiments_40"] = _exp40
_spec.loader.exec_module(_exp40)

LOCAL_SNAPSHOTS = {
    17: REPO_ROOT / "reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800_AB_A1prime.pt",
    11: REPO_ROOT / "reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800_AB_A1prime_seed11.pt",
    23: REPO_ROOT / "reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800_AB_A1prime_seed23.pt",
}

# Operating point — locked to the report-061 v2 conventions.
BETA = 10.0
N_CUES = 30        # smoke; full run is 100
MAX_ITER = 12
BINDING_NOISE_STD = 0.05
CONTENT_DISTORTION = 0.6
CUE_SEED = 117
PSEUDOINVERSE_LAMBDA = 1e-3
D3_MIX_MU = 0.5    # cross-K weight in additive update; baseline = MHN at μ=0
MAGNITUDE_FLOOR = 5.5e-3


def _mhn_settle(
    *,
    substrate,
    patterns_matrix: torch.Tensor,   # [N, D]
    init_state: torch.Tensor,        # [D]
    beta: float,
    max_iter: int,
) -> torch.Tensor:
    """Standard MHN softmax settling."""
    state = init_state.clone()
    for _ in range(max_iter):
        scores = substrate.similarity_matrix(state, patterns_matrix)
        weights = torch.softmax(beta * scores, dim=0)
        update = (patterns_matrix * weights[:, None]).sum(dim=0)
        state = substrate.normalize(update)
    return state


def _build_pseudoinverse_kernel(
    patterns_matrix: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute inv_gram = (X Xᴴ + λI)⁻¹ on CPU (MPS lacks complex LU).

    Returned tensor lives on the same device as patterns_matrix.
    """
    target_device = patterns_matrix.device
    X_cpu = patterns_matrix.detach().to("cpu")
    gram = X_cpu @ X_cpu.conj().transpose(-2, -1)        # [N, N]
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device="cpu")
    inv_gram = torch.linalg.solve(gram + lam * eye, eye)  # [N, N]
    return inv_gram.to(target_device)


def _pseudoinverse_settle(
    *,
    substrate,
    patterns_matrix: torch.Tensor,   # [N, D] complex FHRR
    init_state: torch.Tensor,        # [D]
    max_iter: int,
    inv_gram: torch.Tensor,          # [N, N] precomputed
) -> torch.Tensor:
    """D1: linear pseudo-inverse settling.

    For FHRR (complex-valued patterns), W = Xᴴ (X Xᴴ + λI)⁻¹ X — the
    Hermitian-conjugate analog of the real pseudo-inverse projection.
    This projects state onto the row space of X (the stored patterns)
    via the regularized Moore-Penrose pseudo-inverse.
    """
    X = patterns_matrix                          # [N, D] complex
    state = init_state.clone()
    for _ in range(max_iter):
        # state_new = Xᴴ (inv_gram (X state))
        z = X @ state                            # [N]
        z = inv_gram @ z                         # [N]
        update = X.conj().transpose(-2, -1) @ z  # [D]
        state = substrate.normalize(update)
    return state


def _mhn_settle_cross_k(
    *,
    substrate,
    patterns_matrix: torch.Tensor,   # [N, D]
    init_states: torch.Tensor,       # [K, D]
    beta: float,
    max_iter: int,
    mu: float,
) -> torch.Tensor:
    """D3: additive cross-K + per-branch MHN settling.

    Per the 2026-05-24 D3 Lyapunov analytical pass:
      s_k ← Σ_p [(1-μ)·π_k(p) + μ·α_k(p)] · x_p
    where π_k = softmax over patterns p (per-branch), α_k = softmax over
    branches K (per-pattern). At μ=0 this is independent K-branch MHN.
    At μ=1 it is pure cross-K competition. Lyapunov-clean for all μ ∈ [0,1]
    (sum of two log-sum-exp energies is Lyapunov; convex combination
    preserves the property).
    """
    K = init_states.shape[0]
    states = init_states.clone()                 # [K, D]
    for _ in range(max_iter):
        # logits[k, p] = β · Re⟨x_p, s_k⟩ for FHRR
        # similarity_matrix returns real scores
        all_scores = torch.stack([
            substrate.similarity_matrix(states[k], patterns_matrix)
            for k in range(K)
        ], dim=0)                                # [K, N]
        logits = beta * all_scores               # [K, N]
        # per-branch softmax over patterns
        pi = torch.softmax(logits, dim=-1)       # [K, N]
        # cross-K softmax over branches, for each pattern
        alpha = torch.softmax(logits, dim=0)     # [K, N]
        combined = (1.0 - mu) * pi + mu * alpha  # [K, N] real
        # Cast real weights to complex for matmul with complex patterns
        combined_c = combined.to(patterns_matrix.dtype)
        # update each branch: state_k = Σ_p combined[k,p] · x_p
        # combined is not row-normalized; normalize after weighting
        update = combined_c @ patterns_matrix    # [K, D]
        for k in range(K):
            states[k] = substrate.normalize(update[k])
    return states


def _final_state_score_rank(
    *,
    substrate,
    state: torch.Tensor,
    patterns_matrix: torch.Tensor,
    target_idx: int,
) -> Tuple[float, int, int]:
    """Score and rank of target_idx in the final-state similarity profile."""
    scores = substrate.similarity_matrix(state, patterns_matrix)
    target_score = float(scores[target_idx])
    # rank: 1 = highest score
    sorted_scores, _ = torch.sort(scores, descending=True)
    # rank = position of target_score in sorted_scores
    rank = int((sorted_scores >= target_score - 1e-9).sum())  # bigger or equal
    top_idx = int(scores.argmax())
    return target_score, rank, top_idx


def _unbiased_energy(
    *,
    substrate,
    state: torch.Tensor,
    patterns_matrix: torch.Tensor,
    beta: float,
) -> float:
    """E(q) = -(1/β) log Σ_p exp(β · Re⟨x_p, q⟩) per Ramsauer."""
    scores = substrate.similarity_matrix(state, patterns_matrix)
    return float(-(1.0 / beta) * torch.logsumexp(beta * scores, dim=0))


def _run_seed_condition(
    *,
    snapshot_path: Path,
    device: str,
    condition: str,    # 'baseline' | 'D1' | 'D3'
    n_cues: int,
    cue_seed: int,
) -> Dict:
    """One seed × one condition. Returns aggregated metrics."""
    mem, _cons, patterns, positions, _info = _exp40._load_substrate_from_snapshot(
        path=str(snapshot_path), device=device,
    )
    if positions is None:
        raise RuntimeError(f"snapshot {snapshot_path} has no positions; cannot build role cues")
    substrate = mem.substrate
    patterns_matrix = mem._pattern_matrix()
    pos_list = [positions[i] for i in range(positions.shape[0])]

    cues_data = _exp40._build_role_binding_cues(
        substrate=substrate,
        positions=pos_list,
        patterns=patterns,
        n_cues=n_cues,
        binding_noise_std=BINDING_NOISE_STD,
        content_distortion=CONTENT_DISTORTION,
        seed=cue_seed,
    )

    de_values: List[float] = []   # E_content_prior - E_role_prior
    hit_role_flags: List[int] = []
    rank_role_values: List[int] = []
    random_lowest_flags: List[int] = []
    role_target_score_values: List[float] = []

    # Precompute pseudo-inverse kernel once per snapshot (D1 only)
    inv_gram = None
    if condition == "D1":
        inv_gram = _build_pseudoinverse_kernel(patterns_matrix, lam=PSEUDOINVERSE_LAMBDA)

    rng = torch.Generator().manual_seed(cue_seed + 1)

    for cue_record in cues_data:
        cue = cue_record["cue"].to(device)
        role_idx = cue_record["role_target_idx"]
        content_idx = cue_record["content_distractor_idx"]
        # Random distractor: pick any pattern that's not role_idx or content_idx
        rand_idx = int(torch.randint(
            0, len(patterns), (1,), generator=rng,
        ).item())
        while rand_idx in (role_idx, content_idx):
            rand_idx = int(torch.randint(
                0, len(patterns), (1,), generator=rng,
            ).item())

        # Build the three priors as initial states (atoms themselves)
        role_init = patterns[role_idx].to(device).clone()
        content_init = patterns[content_idx].to(device).clone()
        random_init = patterns[rand_idx].to(device).clone()

        if condition == "baseline":
            s_role = _mhn_settle(
                substrate=substrate, patterns_matrix=patterns_matrix,
                init_state=role_init, beta=BETA, max_iter=MAX_ITER,
            )
            s_content = _mhn_settle(
                substrate=substrate, patterns_matrix=patterns_matrix,
                init_state=content_init, beta=BETA, max_iter=MAX_ITER,
            )
            s_random = _mhn_settle(
                substrate=substrate, patterns_matrix=patterns_matrix,
                init_state=random_init, beta=BETA, max_iter=MAX_ITER,
            )
        elif condition == "D1":
            s_role = _pseudoinverse_settle(
                substrate=substrate, patterns_matrix=patterns_matrix,
                init_state=role_init, max_iter=MAX_ITER, inv_gram=inv_gram,
            )
            s_content = _pseudoinverse_settle(
                substrate=substrate, patterns_matrix=patterns_matrix,
                init_state=content_init, max_iter=MAX_ITER, inv_gram=inv_gram,
            )
            s_random = _pseudoinverse_settle(
                substrate=substrate, patterns_matrix=patterns_matrix,
                init_state=random_init, max_iter=MAX_ITER, inv_gram=inv_gram,
            )
        elif condition == "D3":
            # Cross-K settle all three branches together
            init = torch.stack([role_init, content_init, random_init], dim=0)
            final = _mhn_settle_cross_k(
                substrate=substrate, patterns_matrix=patterns_matrix,
                init_states=init, beta=BETA, max_iter=MAX_ITER, mu=D3_MIX_MU,
            )
            s_role, s_content, s_random = final[0], final[1], final[2]
        else:
            raise ValueError(f"unknown condition {condition}")

        # Phase 5 headline: ΔE = E_content_prior - E_role_prior
        e_role = _unbiased_energy(
            substrate=substrate, state=s_role,
            patterns_matrix=patterns_matrix, beta=BETA,
        )
        e_content = _unbiased_energy(
            substrate=substrate, state=s_content,
            patterns_matrix=patterns_matrix, beta=BETA,
        )
        e_random = _unbiased_energy(
            substrate=substrate, state=s_random,
            patterns_matrix=patterns_matrix, beta=BETA,
        )
        de_values.append(e_content - e_role)

        # Role-target basin retrieval: does role-prior branch land at role_idx?
        target_score, rank, top_idx = _final_state_score_rank(
            substrate=substrate, state=s_role,
            patterns_matrix=patterns_matrix, target_idx=role_idx,
        )
        hit_role_flags.append(1 if top_idx == role_idx else 0)
        rank_role_values.append(rank)
        role_target_score_values.append(target_score)

        # random_lowest: did random-prior branch land at the lowest energy?
        # (i.e. random > content AND random > role, since we want role lowest)
        is_random_lowest = (e_random <= e_content) and (e_random <= e_role)
        random_lowest_flags.append(1 if is_random_lowest else 0)

    n = len(de_values)
    return {
        "condition": condition,
        "n_cues": n,
        "mean_delta_e": sum(de_values) / n,
        "delta_e_values": de_values,
        "mean_hit_role": sum(hit_role_flags) / n,
        "mean_rank_role": sum(rank_role_values) / n,
        "mean_role_target_score": sum(role_target_score_values) / n,
        "mean_random_lowest": sum(random_lowest_flags) / n,
        "magnitude_floor": MAGNITUDE_FLOOR,
        "magnitude_floor_ratio": (sum(de_values) / n) / MAGNITUDE_FLOOR,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 11, 23])
    parser.add_argument("--conditions", nargs="+", default=["baseline", "D1", "D3"])
    parser.add_argument("--n-cues", type=int, default=N_CUES)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "reports/spike_d1_d3_local_smoke.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Spike D1+D3 local smoke")
    print("=" * 70)
    print("Active phase: 5")
    print("Headline metric per phase-5-unified-design.md:269-292:")
    print("  ΔE = E_content_prior - E_role_prior, paired per cue, CI > 0,")
    print(f"  magnitude floor {MAGNITUDE_FLOOR}, n_seeds ≥ 10 for graduation.")
    print("Required controls per phase-5-unified-design.md:296-303:")
    print("  This is a SMOKE test (n=3 seeds × 30 cues), not graduation.")
    print(f"Last verified result: Report 061 — baseline hit_role ≈ 0.003,")
    print(f"  rank_role ≈ 443/489, mean ΔE +0.0089 (above floor) under Path C")
    print(f"  log-prior spike but with random_lowest 0.37 (structural retrieval absent).")
    print("Why now: D1 (pseudo-inverse) and D3 (cross-K softmax) are Tier-0")
    print("  diagnostic spikes from the 2026-05-24 Phase 5 rescue brainstorm.")
    print("  This smoke confirms or refutes whether either materially moves")
    print("  hit_role/rank_role before committing Colab compute.")
    print(f"Conditions: {args.conditions}")
    print(f"Seeds: {args.seeds}, n_cues per seed: {args.n_cues}")
    print(f"Device: {args.device}")
    print()

    results: Dict[str, Dict] = {
        "metadata": {
            "active_phase": 5,
            "seeds": args.seeds,
            "n_cues": args.n_cues,
            "beta": BETA,
            "max_iter": MAX_ITER,
            "binding_noise_std": BINDING_NOISE_STD,
            "content_distortion": CONTENT_DISTORTION,
            "cue_seed": CUE_SEED,
            "pseudoinverse_lambda": PSEUDOINVERSE_LAMBDA,
            "d3_mix_mu": D3_MIX_MU,
            "magnitude_floor": MAGNITUDE_FLOOR,
        },
        "per_seed": {},
        "aggregated": {},
    }

    for cond in args.conditions:
        results["per_seed"][cond] = {}

    for seed in args.seeds:
        snap = LOCAL_SNAPSHOTS.get(seed)
        if snap is None or not snap.exists():
            print(f"[skip] seed {seed}: snapshot not found at {snap}")
            continue
        print(f"--- seed {seed} ---")
        for cond in args.conditions:
            t0 = time.time()
            res = _run_seed_condition(
                snapshot_path=snap, device=args.device, condition=cond,
                n_cues=args.n_cues, cue_seed=CUE_SEED + seed,
            )
            dt = time.time() - t0
            print(
                f"  [{cond:8s}] ΔE={res['mean_delta_e']:+.5f} "
                f"({res['magnitude_floor_ratio']:+.2f}× floor)  "
                f"hit_role={res['mean_hit_role']:.3f}  "
                f"rank_role={res['mean_rank_role']:.1f}  "
                f"rand_low={res['mean_random_lowest']:.3f}  "
                f"({dt:.1f}s)"
            )
            results["per_seed"][cond][str(seed)] = res

    # Aggregate across seeds
    for cond in args.conditions:
        seed_results = list(results["per_seed"][cond].values())
        if not seed_results:
            continue
        n_seeds = len(seed_results)
        agg = {
            "n_seeds": n_seeds,
            "mean_delta_e": sum(r["mean_delta_e"] for r in seed_results) / n_seeds,
            "mean_hit_role": sum(r["mean_hit_role"] for r in seed_results) / n_seeds,
            "mean_rank_role": sum(r["mean_rank_role"] for r in seed_results) / n_seeds,
            "mean_random_lowest": sum(r["mean_random_lowest"] for r in seed_results) / n_seeds,
            "seeds_de_positive": sum(1 for r in seed_results if r["mean_delta_e"] > 0),
        }
        results["aggregated"][cond] = agg

    print()
    print("=" * 70)
    print("Aggregated across seeds")
    print("=" * 70)
    print(f"{'condition':10s}  {'mean ΔE':>10s}  {'×floor':>7s}  "
          f"{'hit_role':>9s}  {'rank_role':>10s}  {'rand_low':>9s}  "
          f"{'seeds+':>6s}")
    for cond in args.conditions:
        agg = results["aggregated"].get(cond)
        if not agg:
            continue
        ratio = agg["mean_delta_e"] / MAGNITUDE_FLOOR
        print(
            f"{cond:10s}  {agg['mean_delta_e']:+10.5f}  {ratio:+7.2f}  "
            f"{agg['mean_hit_role']:9.3f}  {agg['mean_rank_role']:10.1f}  "
            f"{agg['mean_random_lowest']:9.3f}  "
            f"{agg['seeds_de_positive']}/{agg['n_seeds']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
