"""Spike E1 — Centered log-prior field (GPT Idea 1: 'Path C done right').

A landscape-reshaping closure of Path C, run as a smoke at the same
operating point as Report 062 so results are directly comparable.

Mechanism
---------
Instead of the report-061 one-hot per-pattern log-prior spike that boosts
only the selector-chosen atom (arbitration-shape under the no-schema-store
control), this computes a **zero-mean per-atom logit field** derived from
role-vs-content similarity:

  role_score_i    = max_w cosine(unbind(cue, p_w), atom_i)
  content_score_i = cosine(cue, atom_i)
  z_role    = (role_score - mean(role_score)) / std(role_score)
  z_content = (content_score - mean(content_score)) / std(content_score)
  log_ratio = log((softplus(z_role)+ε) / (softplus(z_content)+ε))
  b_i = λ · (log_ratio - mean(log_ratio))   # CENTER → zero-mean field

  per-pattern logits = β · sim(s, atom_i) + b_i

The centering is load-bearing per anti-homunculus: the field redistributes
basin depth rather than injecting free energy into one chosen atom. b is
the same vector for all branches of a given cue — it is a static landscape
modulator, not a per-branch arbiter. Each branch still settles via its
own MHN gradient flow under the modified energy.

This is the spike-level test of GPT's Idea 1 from the 2026-05-24 brainstorm
discussion: does asymmetric role/content landscape reshaping move
hit_role / rank_role on the current substrate, or does it produce only
energy-margin movement (the Path-C failure mode)?

Anti-homunculus check
---------------------
- b_i is a deterministic function of (cue, atom_i) — pointwise local
- Centering is a fixed algebraic operation (subtract mean)
- λ is a global substrate parameter (sweep is over its values, not adaptive)
- Branch selection remains energy-only after b is fixed
- No metric is read at runtime to choose anything

Falsification
-------------
λ=0 must be bit-identical to baseline (no E1 modulation).
If E1 moves ΔE clean above floor but hit_role stays ≈0: Path C done right
is still wrong-shape on this substrate; the substrate genuinely lacks
role-target basins (consistent with Report 062 D1/D3 null).
If E1 moves hit_role materially above 0.01: substrate has weak role
information that asymmetric landscape reshaping surfaces; revisit Path D
decision with E1 in the mix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

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

BETA = 10.0
N_CUES = 30
MAX_ITER = 12
BINDING_NOISE_STD = 0.05
CONTENT_DISTORTION = 0.6
CUE_SEED = 117
MAGNITUDE_FLOOR = 5.5e-3
EPS = 1e-6


def _compute_e1_field(
    *,
    substrate,
    cue: torch.Tensor,
    patterns_matrix: torch.Tensor,   # [N, D]
    positions: List[torch.Tensor],   # W FHRR vectors
    lam: float,
) -> torch.Tensor:
    """Compute the centered E1 logit field b_i per atom.

    Returns a real-valued [N] tensor (same shape as scores).
    """
    # content_score_i = cosine similarity between cue and atom_i (real part)
    content_scores = substrate.similarity_matrix(cue, patterns_matrix)  # [N] real

    # role_score_i = sum over positions of cosine(unbind(cue, p_w), atom_i)
    # Sum (not max) avoids noise-peak amplification across W positions:
    # max would inflate atoms whose noisy unbinding at any position
    # happens to be high. Sum integrates the "atom is bound at any
    # position" signal and is the natural FHRR analog of the project's
    # existing role-binding similarity in experiments/40.
    role_scores_per_pos = []
    for p_w in positions:
        u = substrate.unbind(cue, p_w)
        sim = substrate.similarity_matrix(u, patterns_matrix)  # [N] real
        role_scores_per_pos.append(sim)
    role_scores = torch.stack(role_scores_per_pos, dim=0).sum(dim=0)  # [N]

    # Standardize each
    def _standardize(x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean()) / (x.std(unbiased=False) + EPS)

    z_role = _standardize(role_scores)
    z_content = _standardize(content_scores)

    # log((softplus(z_role)+ε) / (softplus(z_content)+ε))
    log_ratio = torch.log(
        (torch.nn.functional.softplus(z_role) + EPS)
        / (torch.nn.functional.softplus(z_content) + EPS)
    )
    # CENTER (zero-mean): subtract mean across atoms
    log_ratio_centered = log_ratio - log_ratio.mean()
    b = lam * log_ratio_centered
    return b


def _mhn_settle_with_field(
    *,
    substrate,
    patterns_matrix: torch.Tensor,
    init_state: torch.Tensor,
    field_b: torch.Tensor,         # [N] real per-atom logit bias (constant during settling)
    beta: float,
    max_iter: int,
) -> torch.Tensor:
    """MHN settling with an additive per-pattern logit field."""
    state = init_state.clone()
    for _ in range(max_iter):
        scores = substrate.similarity_matrix(state, patterns_matrix)  # [N] real
        logits = beta * scores + field_b
        weights = torch.softmax(logits, dim=0)
        # patterns_matrix is complex; weights is real → cast
        w_complex = weights.to(patterns_matrix.dtype)
        update = (patterns_matrix * w_complex[:, None]).sum(dim=0)
        state = substrate.normalize(update)
    return state


def _unbiased_energy(
    *,
    substrate,
    state: torch.Tensor,
    patterns_matrix: torch.Tensor,
    beta: float,
) -> float:
    """E(q) = -(1/β) log Σ_p exp(β · Re⟨x_p, q⟩) per Ramsauer.

    Note: E1 spike reports UNBIASED energy (without the b_i field), so
    cross-condition ΔE comparison is on the same energy function. The
    field reshapes the settling dynamics but the reported energy is the
    substrate-pure Hopfield energy on the final state.
    """
    scores = substrate.similarity_matrix(state, patterns_matrix)
    return float(-(1.0 / beta) * torch.logsumexp(beta * scores, dim=0))


def _final_state_rank(
    *,
    substrate,
    state: torch.Tensor,
    patterns_matrix: torch.Tensor,
    target_idx: int,
):
    scores = substrate.similarity_matrix(state, patterns_matrix)
    target_score = float(scores[target_idx])
    sorted_scores, _ = torch.sort(scores, descending=True)
    rank = int((sorted_scores >= target_score - 1e-9).sum())
    top_idx = int(scores.argmax())
    return target_score, rank, top_idx


def _run_seed_lambda(
    *,
    snapshot_path: Path,
    device: str,
    lam: float,            # 0 = baseline (no E1 modulation)
    n_cues: int,
    cue_seed: int,
) -> Dict:
    mem, _cons, patterns, positions, _info = _exp40._load_substrate_from_snapshot(
        path=str(snapshot_path), device=device,
    )
    substrate = mem.substrate
    patterns_matrix = mem._pattern_matrix()
    pos_list = [positions[i] for i in range(positions.shape[0])]

    cues_data = _exp40._build_role_binding_cues(
        substrate=substrate, positions=pos_list, patterns=patterns,
        n_cues=n_cues, binding_noise_std=BINDING_NOISE_STD,
        content_distortion=CONTENT_DISTORTION, seed=cue_seed,
    )

    de_values: List[float] = []
    hit_role_flags: List[int] = []
    rank_role_values: List[int] = []
    random_lowest_flags: List[int] = []
    field_magnitudes: List[float] = []

    rng = torch.Generator().manual_seed(cue_seed + 1)

    for cue_record in cues_data:
        cue = cue_record["cue"].to(device)
        role_idx = cue_record["role_target_idx"]
        content_idx = cue_record["content_distractor_idx"]
        rand_idx = int(torch.randint(0, len(patterns), (1,), generator=rng).item())
        while rand_idx in (role_idx, content_idx):
            rand_idx = int(torch.randint(0, len(patterns), (1,), generator=rng).item())

        role_init = patterns[role_idx].to(device).clone()
        content_init = patterns[content_idx].to(device).clone()
        random_init = patterns[rand_idx].to(device).clone()

        # E1 field is per-cue, same for all branches
        if lam == 0.0:
            field_b = torch.zeros(
                patterns_matrix.shape[0], dtype=torch.float32, device=device,
            )
        else:
            field_b = _compute_e1_field(
                substrate=substrate, cue=cue,
                patterns_matrix=patterns_matrix, positions=pos_list, lam=lam,
            )
        field_magnitudes.append(float(field_b.abs().mean()))

        s_role = _mhn_settle_with_field(
            substrate=substrate, patterns_matrix=patterns_matrix,
            init_state=role_init, field_b=field_b, beta=BETA, max_iter=MAX_ITER,
        )
        s_content = _mhn_settle_with_field(
            substrate=substrate, patterns_matrix=patterns_matrix,
            init_state=content_init, field_b=field_b, beta=BETA, max_iter=MAX_ITER,
        )
        s_random = _mhn_settle_with_field(
            substrate=substrate, patterns_matrix=patterns_matrix,
            init_state=random_init, field_b=field_b, beta=BETA, max_iter=MAX_ITER,
        )

        # Unbiased energies for cross-condition comparability
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

        _, rank, top_idx = _final_state_rank(
            substrate=substrate, state=s_role,
            patterns_matrix=patterns_matrix, target_idx=role_idx,
        )
        hit_role_flags.append(1 if top_idx == role_idx else 0)
        rank_role_values.append(rank)

        is_random_lowest = (e_random <= e_content) and (e_random <= e_role)
        random_lowest_flags.append(1 if is_random_lowest else 0)

    n = len(de_values)
    mean_de = sum(de_values) / n
    return {
        "lambda": lam,
        "n_cues": n,
        "mean_delta_e": mean_de,
        "magnitude_floor_ratio": mean_de / MAGNITUDE_FLOOR,
        "delta_e_values": de_values,
        "mean_hit_role": sum(hit_role_flags) / n,
        "mean_rank_role": sum(rank_role_values) / n,
        "mean_random_lowest": sum(random_lowest_flags) / n,
        "mean_field_magnitude": sum(field_magnitudes) / n,
        "n_de_positive": sum(1 for d in de_values if d > 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 11, 23])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--n-cues", type=int, default=N_CUES)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "reports/spike_e1_centered_log_prior.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Spike E1 — Centered log-prior closure of Path C")
    print("=" * 70)
    print("Active phase: 5")
    print("Headline metric per phase-5-unified-design.md:269-292:")
    print(f"  ΔE = E_content_prior - E_role_prior; floor {MAGNITUDE_FLOOR}, n_seeds ≥ 10 for graduation.")
    print("Required controls: λ=0 must be bit-identical to baseline MHN")
    print("  (same E_total at λ=0 since field is zero).")
    print("Last verified result: Report 062 — D1/D3 smoke null on hit_role")
    print("  for all 90 cues, routing the brainstorm to Tier-2 (Path D).")
    print("Why now: E1 ('Path C done right') closes the question of whether")
    print("  asymmetric role/content LANDSCAPE reshaping (zero-mean field over")
    print("  whole codebook) materially differs from Path C's one-hot spike")
    print("  (which is arbitration-shape under no-schema-store amplification).")
    print(f"Conditions: λ ∈ {args.lambdas}")
    print(f"Seeds: {args.seeds}, n_cues per seed: {args.n_cues}")
    print()

    results: Dict = {
        "metadata": {
            "active_phase": 5,
            "seeds": args.seeds,
            "n_cues": args.n_cues,
            "beta": BETA,
            "max_iter": MAX_ITER,
            "binding_noise_std": BINDING_NOISE_STD,
            "content_distortion": CONTENT_DISTORTION,
            "cue_seed": CUE_SEED,
            "magnitude_floor": MAGNITUDE_FLOOR,
            "lambdas": args.lambdas,
        },
        "per_seed": {},
        "aggregated_by_lambda": {},
    }

    for lam in args.lambdas:
        results["per_seed"][str(lam)] = {}

    for seed in args.seeds:
        snap = LOCAL_SNAPSHOTS.get(seed)
        if snap is None or not snap.exists():
            print(f"[skip] seed {seed}: snapshot not found")
            continue
        print(f"--- seed {seed} ---")
        for lam in args.lambdas:
            t0 = time.time()
            res = _run_seed_lambda(
                snapshot_path=snap, device=args.device, lam=lam,
                n_cues=args.n_cues, cue_seed=CUE_SEED + seed,
            )
            dt = time.time() - t0
            print(
                f"  [λ={lam:>4.2f}] ΔE={res['mean_delta_e']:+.5f} "
                f"({res['magnitude_floor_ratio']:+.2f}× floor)  "
                f"hit_role={res['mean_hit_role']:.3f}  "
                f"rank_role={res['mean_rank_role']:.1f}  "
                f"rand_low={res['mean_random_lowest']:.3f}  "
                f"|b|={res['mean_field_magnitude']:.3f}  "
                f"({dt:.1f}s)"
            )
            results["per_seed"][str(lam)][str(seed)] = res

    # Aggregate across seeds per λ
    for lam in args.lambdas:
        seed_results = list(results["per_seed"][str(lam)].values())
        if not seed_results:
            continue
        n_seeds = len(seed_results)
        agg = {
            "n_seeds": n_seeds,
            "mean_delta_e": sum(r["mean_delta_e"] for r in seed_results) / n_seeds,
            "mean_hit_role": sum(r["mean_hit_role"] for r in seed_results) / n_seeds,
            "mean_rank_role": sum(r["mean_rank_role"] for r in seed_results) / n_seeds,
            "mean_random_lowest": sum(r["mean_random_lowest"] for r in seed_results) / n_seeds,
            "mean_field_magnitude": sum(r["mean_field_magnitude"] for r in seed_results) / n_seeds,
            "seeds_de_positive": sum(1 for r in seed_results if r["mean_delta_e"] > 0),
            "magnitude_floor_ratio": (sum(r["mean_delta_e"] for r in seed_results) / n_seeds) / MAGNITUDE_FLOOR,
        }
        results["aggregated_by_lambda"][str(lam)] = agg

    print()
    print("=" * 70)
    print("Aggregated across seeds (λ sweep)")
    print("=" * 70)
    print(f"{'lambda':>6s}  {'mean ΔE':>10s}  {'×floor':>7s}  "
          f"{'hit_role':>9s}  {'rank_role':>10s}  {'rand_low':>9s}  "
          f"{'|b| field':>10s}  {'seeds+':>7s}")
    for lam in args.lambdas:
        agg = results["aggregated_by_lambda"].get(str(lam))
        if not agg:
            continue
        print(
            f"{lam:>6.2f}  {agg['mean_delta_e']:+10.5f}  {agg['magnitude_floor_ratio']:+7.2f}  "
            f"{agg['mean_hit_role']:9.3f}  {agg['mean_rank_role']:10.1f}  "
            f"{agg['mean_random_lowest']:9.3f}  "
            f"{agg['mean_field_magnitude']:10.4f}  "
            f"{agg['seeds_de_positive']}/{agg['n_seeds']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
