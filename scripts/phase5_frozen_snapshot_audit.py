"""Phase 5 frozen-snapshot audit: substrate measurement harness.

Reusable measurement layer for cross-seed substrate inspection and
operating-point characterization. Loads frozen Phase 4 substrate
snapshots (the .pt files produced by experiments/19_phase34_integrated.py
with --snapshot-steps), emits structured JSON/CSV with per-snapshot:

  - geometry: n_atoms, dim, coverage_lambda, epsilon (retrieval_weight_epsilon),
    tau (retrieval_weight_tau)
  - effective-strength distribution: |E_i| min, median, mean, max, std
  - step-3 retrieval-weight bias distribution: bias min, median, mean,
    max, std, n_below_epsilon, n_bias_ge_1, bias_cv = std / |mean|
  - optional β preflight: for each β in {1, 3, 5, 10, 30} on n_cue_probes
    random cues, capture max softmax weight at iteration 1 and final
    iteration, softmax entropy at iteration 1 and final, and the
    trajectory gap max_t(w_top) − w_top(final).

No training, no graduation claim, no retuning. Pure measurement.

Usage
-----
Audit a single snapshot::

    python scripts/phase5_frozen_snapshot_audit.py \\
        --snapshot reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt \\
        --output reports/phase5_audit/seed17.json

Audit a directory of snapshots (matches *.pt recursively)::

    python scripts/phase5_frozen_snapshot_audit.py \\
        --snapshot-dir reports/phase5_a1prime_pilot_seed17/snapshots \\
        --output reports/phase5_audit/a1prime_all.json

Include the β preflight (adds ~30s per snapshot at D=4096)::

    python scripts/phase5_frozen_snapshot_audit.py \\
        --snapshot ... --beta-preflight --n-cue-probes 8 \\
        --output reports/phase5_audit/seed17_beta.json

CSV companion alongside JSON::

    python scripts/phase5_frozen_snapshot_audit.py ... --csv-also

Why
---
Report 054 surfaced that on the seed-17 A1' substrate, every atom has
|E_i| ≈ 0.025 (all below ε=0.05), making the step-3 sigmoidal bias
near-uniform and therefore softmax-shift-invariant — step 3 is
empirically inert. That was one seed. This harness is the cross-seed
generalization check: does the saturation pattern hold across all
available A+B+A1' snapshots, or is seed 17 a corner case?

The β preflight is the cheaper, parallelizable arm of the upstream
diagnostics — it tests whether β=10 + D=4096 is the saturation corner,
without retraining, on whatever snapshots already exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Reuse experiments/40's snapshot loader so the substrate setup is
# identical to what the headline run sees. Loaded via importlib because
# the filename starts with a digit.
import importlib.util  # noqa: E402

_EXP40_PATH = REPO_ROOT / "experiments" / "40_phase5_branching.py"
_spec = importlib.util.spec_from_file_location("experiments_40", str(_EXP40_PATH))
_exp40 = importlib.util.module_from_spec(_spec)
sys.modules["experiments_40"] = _exp40
_spec.loader.exec_module(_exp40)


def _tensor_summary(t: torch.Tensor) -> Dict[str, float]:
    """min/median/mean/max/std as Python floats. Empty → all NaN."""
    if t.numel() == 0:
        return {
            "min": float("nan"), "median": float("nan"),
            "mean": float("nan"), "max": float("nan"), "std": float("nan"),
        }
    return {
        "min": float(t.min()),
        "median": float(t.median()),
        "mean": float(t.mean()),
        "max": float(t.max()),
        "std": float(t.std(unbiased=False)),
    }


def _settle_for_preflight(
    *,
    memory,
    cue: torch.Tensor,
    beta: float,
    max_iter: int = 12,
) -> Dict[str, float]:
    """Plain unbiased Hopfield settling that captures per-iter softmax stats.

    Returns max_w at iter 1, max_w final, entropy at iter 1, entropy final,
    trajectory_gap = max over iterations of w_top minus final w_top.

    No score_bias, no prior, no γ — this is the raw landscape probe. It
    isolates the substrate's softmax sharpness as a function of β.
    """
    patterns = memory._pattern_matrix()
    substrate = memory.substrate
    device = substrate.device
    state = cue.to(device)

    max_w_per_iter: List[float] = []
    entropy_per_iter: List[float] = []
    top_idx_history: List[int] = []
    for _ in range(max_iter):
        scores = substrate.similarity_matrix(state, patterns)
        weights = torch.softmax(beta * scores, dim=0)
        max_w_per_iter.append(float(weights.max()))
        top_idx_history.append(int(weights.argmax()))
        safe = weights.clamp(min=1e-12)
        entropy_per_iter.append(float(-(safe * safe.log()).sum()))
        update = (patterns * weights[:, None]).sum(dim=0)
        state = substrate.normalize(update)

    final_idx = top_idx_history[-1]
    # Trajectory gap (Path-3 c_i style): how much did the winner's weight
    # peak above its final value? Captures "lost-out" dynamics — when this
    # is near zero, the substrate is winner-take-all from iteration 1.
    winning_w_history = []
    state2 = cue.to(device)
    for _ in range(max_iter):
        scores = substrate.similarity_matrix(state2, patterns)
        weights = torch.softmax(beta * scores, dim=0)
        winning_w_history.append(float(weights[final_idx]))
        update = (patterns * weights[:, None]).sum(dim=0)
        state2 = substrate.normalize(update)
    traj_gap = max(winning_w_history) - winning_w_history[-1]

    return {
        "max_w_iter1": max_w_per_iter[0],
        "max_w_final": max_w_per_iter[-1],
        "entropy_iter1": entropy_per_iter[0],
        "entropy_final": entropy_per_iter[-1],
        "trajectory_gap": traj_gap,
    }


def _run_beta_preflight(
    *,
    memory,
    betas: List[float],
    n_cue_probes: int,
    seed: int,
    max_iter: int = 12,
) -> Dict[str, Dict[str, float]]:
    """Average per-β preflight stats across n_cue_probes random unit cues.

    Cues are fresh complex random vectors normalized to unit FHRR phase
    so the preflight measures the substrate's response geometry, not a
    cue-substrate alignment. Off-substrate cues are a deliberate choice:
    they probe the substrate's pulling-power across β regimes without
    being biased toward any particular stored pattern.
    """
    substrate = memory.substrate
    g = torch.Generator(device="cpu").manual_seed(seed)
    out: Dict[str, Dict[str, float]] = {}
    for beta in betas:
        accum: Dict[str, List[float]] = {
            k: [] for k in (
                "max_w_iter1", "max_w_final", "entropy_iter1",
                "entropy_final", "trajectory_gap",
            )
        }
        for k in range(n_cue_probes):
            real = torch.randn(substrate.dim, generator=g)
            imag = torch.randn(substrate.dim, generator=g)
            cue = substrate.normalize((real + 1j * imag).to(torch.complex64))
            stats = _settle_for_preflight(
                memory=memory, cue=cue, beta=beta, max_iter=max_iter,
            )
            for kname in accum:
                accum[kname].append(stats[kname])
        out[f"beta_{beta:g}"] = {
            kname + "_mean": sum(vals) / len(vals)
            for kname, vals in accum.items()
        }
        out[f"beta_{beta:g}"]["n_cue_probes"] = float(n_cue_probes)
    return out


def audit_snapshot(
    *,
    snapshot_path: Path,
    device: str = "cpu",
    beta_preflight: bool = False,
    betas: Optional[List[float]] = None,
    n_cue_probes: int = 4,
    preflight_seed: int = 17,
) -> Dict[str, Any]:
    """Audit one snapshot. Returns a flat dict suitable for JSON or one CSV row."""
    mem, cons, patterns, positions, info = _exp40._load_substrate_from_snapshot(
        path=str(snapshot_path), device=device,
    )
    e_abs = cons.effective_strength().abs()
    eps = float(cons.config.retrieval_weight_epsilon)
    tau = float(cons.config.retrieval_weight_tau)
    coverage_lambda = float(cons.config.coverage_lambda)

    record: Dict[str, Any] = {
        "snapshot": str(snapshot_path),
        "label": info.get("label"),
        "n_atoms": len(patterns),
        "dim": int(mem.substrate.dim),
        "coverage_lambda": coverage_lambda,
        "epsilon": eps,
        "tau": tau,
        "effective_strength": _tensor_summary(e_abs),
        "n_below_epsilon": int((e_abs < eps).sum()),
        "n_above_epsilon": int((e_abs >= eps).sum()),
    }

    if coverage_lambda > 0.0:
        bias = cons.retrieval_weight_bias()
        bias_summary = _tensor_summary(bias)
        # CV: std / |mean|. Near zero → near-uniform bias → softmax-shift-invariant.
        bias_cv = (
            bias_summary["std"] / abs(bias_summary["mean"])
            if abs(bias_summary["mean"]) > 1e-12 else float("nan")
        )
        record["retrieval_weight_bias"] = bias_summary
        record["bias_cv"] = bias_cv
        record["n_bias_ge_1"] = int((bias >= 1.0).sum())
        record["n_bias_ge_0p5"] = int((bias >= 0.5).sum())
        # The shift-invariance test: the step-3 mechanism can only move
        # paired ΔE if bias_cv is materially above zero. We pin the
        # threshold at 0.05 (5% relative spread) as the "materially
        # non-uniform" floor — below this, softmax(x − bias) ≈ softmax(x).
        record["step3_shift_invariant_likely"] = bool(bias_cv < 0.05)
    else:
        record["retrieval_weight_bias"] = None
        record["bias_cv"] = None
        record["n_bias_ge_1"] = None
        record["n_bias_ge_0p5"] = None
        record["step3_shift_invariant_likely"] = None

    if beta_preflight:
        record["beta_preflight"] = _run_beta_preflight(
            memory=mem,
            betas=betas or [1.0, 3.0, 5.0, 10.0, 30.0],
            n_cue_probes=n_cue_probes,
            seed=preflight_seed,
        )

    return record


def _run_headline_beta_sweep(
    *,
    snapshot_path: Path,
    device: str,
    betas: List[float],
    n_cues: int,
    k_main: int,
    gamma: float,
    binding_noise_std: float,
    content_distortion: float,
    formulation: str,
    cue_seed: int,
    delta_energy: float,
    delta_state: float,
    delta_redundant: float,
    temperature: float,
    max_settling_iter: int,
) -> Dict[str, Any]:
    """Headline-cue β sweep: for each β, run the role/content/random
    triplet from experiments/40's headline mode and report paired ΔE
    (raw + step3) plus drill-downs.

    Uses experiments/40's own cue builder, schema selector, and
    `run_branched_retrieval`, so the sweep is on the same code path as
    the production headline run — only β is varied.

    Distinct from the random-cue β preflight: that one probes substrate
    response geometry from random off-substrate cues; this one probes
    the cued retrieval landscape using the exact cue distribution the
    headline reports against.

    No retuning, no graduation claim, no winner-cell selection.
    """
    mem, cons, patterns, positions, info = _exp40._load_substrate_from_snapshot(
        path=str(snapshot_path), device=device,
    )
    if positions is None:
        raise SystemExit(
            f"snapshot {snapshot_path} has no positions; "
            "headline β sweep requires position vectors (re-save with "
            "--snapshot-steps positions support)."
        )

    # Build cues once. The cue_seed defaults to args.seed + 100 in
    # experiments/40 main() — we mirror that convention for parity.
    cue_specs = _exp40._build_role_binding_cues(
        substrate=mem.substrate,
        positions=positions,
        patterns=patterns,
        n_cues=n_cues,
        binding_noise_std=binding_noise_std,
        content_distortion=content_distortion,
        seed=cue_seed,
    )

    # Schema store: top-k by effective_strength (same selector as headline).
    schema_store, atom_idx = _exp40.get_schema_store(
        consolidation=cons, patterns=mem._pattern_matrix(),
        selection_rule="top_k_by_effective_strength",
        k=min(8, len(patterns)),
    )
    schema_bindings = _exp40.compute_schema_bindings(
        substrate=mem.substrate, schemas=schema_store, positions=positions,
    )

    step3_bias = (
        cons.retrieval_weight_bias()
        if cons.config.coverage_lambda > 0.0 else None
    )

    # The three conditions per β (headline core triplet). γ=0 controls
    # and fid_* conditions are skipped for this sweep — adds noise to
    # the β-axis signal we're trying to read. The fixed K_main is used
    # for all three; this matches the production K4 cell.
    condition_specs = [
        ("role", "role"),
        ("content", "content"),
        ("random", "random"),
    ]

    # Per-β results.
    results: Dict[str, Any] = {}
    # RNGs identical across β so each cue's random_K branch uses the
    # same sample at every β — keeps "random" interpretable as a per-β
    # control rather than a per-β fresh sample.
    boltzmann_rng = torch.Generator().manual_seed(13)
    random_prior_rng = torch.Generator().manual_seed(29)

    for beta in betas:
        # Reset RNGs each β so the random-prior samples reproduce.
        boltzmann_rng.manual_seed(13)
        random_prior_rng.manual_seed(29)

        per_condition: Dict[str, Dict[str, List[float]]] = {
            name: {
                "e_min_raw": [], "e_min_step3": [],
                "softmax_entropy": [],
                "max_w_proxy": [],   # max sim on substrate of q_settled — close to max_w under sharp dynamics
                "state_divergence": [],
            } for name, _ in condition_specs
        }

        for spec in cue_specs:
            for name, prior_type in condition_specs:
                res = _exp40.run_branched_retrieval(
                    cue=spec["cue"], cue_id=0, target_id=spec["role_target_idx"],
                    memory=mem, codebook=mem._pattern_matrix(), positions=positions,
                    decode_ids=[], decode_k=5, masked_pos=0,
                    schema_store=schema_store, schema_atom_idx=atom_idx,
                    consolidation=cons, prior_type=prior_type, k_main=k_main,
                    gamma=gamma, beta=beta, temperature=temperature,
                    delta_energy=delta_energy, delta_state=delta_state,
                    delta_redundant=delta_redundant,
                    formulation=formulation,
                    cue_bindings=spec["cue_bindings"],
                    schema_bindings=schema_bindings,
                    include_surprise_branch=False,
                    max_settling_iter=max_settling_iter,
                    boltzmann_rng=boltzmann_rng,
                    random_prior_rng=random_prior_rng,
                    score_bias=step3_bias,
                )
                if not res.branches:
                    continue
                bs = res.branches
                e_raw = min(b.energy_unbiased for b in bs)
                e_step3 = min(b.energy_unbiased_step3 for b in bs)
                nb = len(bs)
                per_condition[name]["e_min_raw"].append(e_raw)
                per_condition[name]["e_min_step3"].append(e_step3)
                per_condition[name]["softmax_entropy"].append(
                    float(res.softmax_entropy)
                )
                per_condition[name]["max_w_proxy"].append(
                    max(
                        float(mem.substrate.similarity(b.q_settled, p))
                        for b in bs for p in patterns
                    )
                )
                per_condition[name]["state_divergence"].append(
                    sum(b.final_state_divergence for b in bs) / nb
                )

        def _mean(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")

        # Paired ΔE = E_content - E_role (positive = role found lower energy).
        c_raw = per_condition["content"]["e_min_raw"]
        r_raw = per_condition["role"]["e_min_raw"]
        c_s3 = per_condition["content"]["e_min_step3"]
        r_s3 = per_condition["role"]["e_min_step3"]
        if c_raw and r_raw and len(c_raw) == len(r_raw):
            per_cue_dE_raw = [c - r for c, r in zip(c_raw, r_raw)]
            per_cue_dE_step3 = [c - r for c, r in zip(c_s3, r_s3)]
        else:
            per_cue_dE_raw = []
            per_cue_dE_step3 = []

        def _frac_pos(xs: List[float]) -> float:
            return sum(1 for x in xs if x > 0) / len(xs) if xs else float("nan")

        # Ordering: rank role / content / random by mean min-energy.
        ordering_raw = sorted(
            ["role", "content", "random"],
            key=lambda k: _mean(per_condition[k]["e_min_raw"]),
        )

        results[f"beta_{beta:g}"] = {
            "n_cues": len(per_cue_dE_raw),
            "delta_e_raw_mean": _mean(per_cue_dE_raw),
            "delta_e_raw_frac_positive": _frac_pos(per_cue_dE_raw),
            "delta_e_step3_mean": _mean(per_cue_dE_step3),
            "delta_e_step3_frac_positive": _frac_pos(per_cue_dE_step3),
            "per_condition_mean_e_min_raw": {
                k: _mean(v["e_min_raw"]) for k, v in per_condition.items()
            },
            "per_condition_mean_e_min_step3": {
                k: _mean(v["e_min_step3"]) for k, v in per_condition.items()
            },
            "per_condition_mean_softmax_entropy": {
                k: _mean(v["softmax_entropy"]) for k, v in per_condition.items()
            },
            "per_condition_mean_max_w_proxy": {
                k: _mean(v["max_w_proxy"]) for k, v in per_condition.items()
            },
            "per_condition_mean_state_divergence": {
                k: _mean(v["state_divergence"]) for k, v in per_condition.items()
            },
            # Ordering low-to-high energy. The headline-intuitive ordering
            # is role < content < random (role-prior finds the lowest
            # energy state; random-prior finds the highest). Deviations
            # are themselves a diagnostic.
            "energy_ordering_low_to_high_raw": ordering_raw,
        }

    return {
        "snapshot": str(snapshot_path),
        "label": info.get("label"),
        "n_atoms": len(patterns),
        "dim": int(mem.substrate.dim),
        "coverage_lambda": float(cons.config.coverage_lambda),
        "step3_bias_active": step3_bias is not None,
        "config": {
            "n_cues": n_cues,
            "k_main": k_main,
            "gamma": gamma,
            "binding_noise_std": binding_noise_std,
            "content_distortion": content_distortion,
            "formulation": formulation,
            "cue_seed": cue_seed,
            "betas": betas,
        },
        "by_beta": results,
    }


def _basin_diagnostics_from_sims(
    *, sims: torch.Tensor, role_target_idx: int,
) -> Dict[str, float]:
    """Per-cue basin-membership readout from precomputed similarities.

    Derives:
      - role_target_basin_hit: 1.0 if argmax similarity == role_target_idx, else 0.0
      - role_target_rank: 1-indexed rank of role_target_idx in
        descending-similarity ordering (1 = top, N = worst)
      - top_similarity: max similarity to any pattern

    This split lets the cue-regime sweep reuse one similarity_matrix call
    for both basin metrics and K=1 max-similarity diagnostics.
    """
    argmax_idx = int(sims.argmax())
    hit = 1.0 if argmax_idx == role_target_idx else 0.0
    sim_role_target = float(sims[role_target_idx])
    rank = 1 + int((sims > sim_role_target).sum())
    top_sim = float(sims.max())
    return {
        "role_target_basin_hit": hit,
        "role_target_rank": float(rank),
        "top_similarity": top_sim,
    }


def _basin_diagnostics(
    *, q_settled: torch.Tensor, role_target_idx: int, patterns_matrix: torch.Tensor,
    substrate,
) -> Dict[str, float]:
    """Per-cue basin-membership readout.

    Computes the similarity of q_settled to every stored pattern, then
    reads out whether the K=1 settled state landed in the role target's
    basin — independent of the paired ΔE energy comparison.
    """
    sims = substrate.similarity_matrix(q_settled, patterns_matrix)
    return _basin_diagnostics_from_sims(
        sims=sims, role_target_idx=role_target_idx,
    )


def _run_headline_cue_regime_sweep(
    *,
    snapshot_path: Path,
    device: str,
    beta: float,
    k_main: int,
    gamma: float,
    binding_noise_grid: List[float],
    content_distortion_grid: List[float],
    n_cues: int,
    formulation: str,
    cue_seed: int,
    delta_energy: float,
    delta_state: float,
    delta_redundant: float,
    temperature: float,
    max_settling_iter: int,
) -> Dict[str, Any]:
    """Cue-regime sweep at fixed β, K, γ over the (binding_noise_std,
    content_distortion) grid. Per cell: paired ΔE + basin membership
    diagnostics + ordering counts.

    Per the working agreement (GPT's recommendation 2026-05-21): paired
    ΔE alone reads basin depth, not structural correctness — the
    cross-seed β sweep showed role < content < random ordering happens
    only 1/10 seeds even when ΔE > 0 9/10. Basin-membership diagnostics
    (role-target hit rate, role-target rank) measure whether q_settled
    landed in the role-target basin; ordering counts measure how often
    random-prior produces the lowest energy (the pathology to watch).

    Single-snapshot. Fixed β=10 + K=1 are GPT-confirmed best operating
    points from reports 056 + 057; the goal here is to find whether
    any cue cell moves ΔE toward the magnitude floor and/or improves
    basin hit rate. No retuning of β/K/γ.
    """
    mem, cons, patterns, positions, info = _exp40._load_substrate_from_snapshot(
        path=str(snapshot_path), device=device,
    )
    if positions is None:
        raise SystemExit(
            f"snapshot {snapshot_path} has no positions; cue-regime sweep "
            "requires position vectors."
        )

    patterns_matrix = mem._pattern_matrix()
    schema_store, atom_idx = _exp40.get_schema_store(
        consolidation=cons, patterns=patterns_matrix,
        selection_rule="top_k_by_effective_strength",
        k=min(8, len(patterns)),
    )
    schema_bindings = _exp40.compute_schema_bindings(
        substrate=mem.substrate, schemas=schema_store, positions=positions,
    )
    step3_bias = (
        cons.retrieval_weight_bias()
        if cons.config.coverage_lambda > 0.0 else None
    )

    condition_specs = [("role", "role"), ("content", "content"), ("random", "random")]
    cells: List[Dict[str, Any]] = []

    boltzmann_rng = torch.Generator().manual_seed(13)
    random_prior_rng = torch.Generator().manual_seed(29)

    for bns in binding_noise_grid:
        for cd in content_distortion_grid:
            # Rebuild cues per cell with the cell's (bns, cd) params.
            cue_specs = _exp40._build_role_binding_cues(
                substrate=mem.substrate,
                positions=positions,
                patterns=patterns,
                n_cues=n_cues,
                binding_noise_std=bns,
                content_distortion=cd,
                seed=cue_seed,
            )

            # Reset prior-side RNGs so the random-prior samples are
            # comparable across cells (controls cell-to-cell variance
            # in the "random" condition).
            boltzmann_rng.manual_seed(13)
            random_prior_rng.manual_seed(29)

            per_condition: Dict[str, Dict[str, List[float]]] = {
                name: {
                    "e_min_raw": [], "e_min_step3": [],
                    "softmax_entropy": [], "max_w_proxy": [],
                    "role_target_basin_hit": [], "role_target_rank": [],
                    "top_similarity": [],
                } for name, _ in condition_specs
            }

            for spec in cue_specs:
                # Per-cue energy of all three conditions (for ordering
                # counts at the cue level).
                cue_energies_raw: Dict[str, float] = {}
                for name, prior_type in condition_specs:
                    res = _exp40.run_branched_retrieval(
                        cue=spec["cue"], cue_id=0, target_id=spec["role_target_idx"],
                        memory=mem, codebook=patterns_matrix, positions=positions,
                        decode_ids=[], decode_k=5, masked_pos=0,
                        schema_store=schema_store, schema_atom_idx=atom_idx,
                        consolidation=cons, prior_type=prior_type, k_main=k_main,
                        gamma=gamma, beta=beta, temperature=temperature,
                        delta_energy=delta_energy, delta_state=delta_state,
                        delta_redundant=delta_redundant,
                        formulation=formulation,
                        cue_bindings=spec["cue_bindings"],
                        schema_bindings=schema_bindings,
                        include_surprise_branch=False,
                        max_settling_iter=max_settling_iter,
                        boltzmann_rng=boltzmann_rng,
                        random_prior_rng=random_prior_rng,
                        score_bias=step3_bias,
                        run_combiners=(k_main > 1),
                    )
                    if not res.branches:
                        cue_energies_raw[name] = float("nan")
                        continue
                    bs = res.branches
                    nb = len(bs)
                    e_raw = min(b.energy_unbiased for b in bs)
                    e_step3 = min(b.energy_unbiased_step3 for b in bs)
                    cue_energies_raw[name] = e_raw
                    per_condition[name]["e_min_raw"].append(e_raw)
                    per_condition[name]["e_min_step3"].append(e_step3)
                    per_condition[name]["softmax_entropy"].append(
                        float(res.softmax_entropy)
                    )
                    # Basin diagnostic: take the K=1 branch's q_settled
                    # (or the bundle re-settle for K>1) and read off
                    # role-target hit/rank.
                    use_bundle = k_main > 1 and res.q_bundle is not None
                    q_star = (
                        res.q_bundle if use_bundle
                        else bs[0].q_settled
                    )
                    q_star_sims = mem.substrate.similarity_matrix(
                        q_star.to(mem.substrate.device), patterns_matrix,
                    )
                    bd = _basin_diagnostics_from_sims(
                        sims=q_star_sims,
                        role_target_idx=int(spec["role_target_idx"]),
                    )
                    max_w_proxy = (
                        bd["top_similarity"] if len(bs) == 1 and not use_bundle
                        else max(
                            float(mem.substrate.similarity(b.q_settled, p))
                            for b in bs for p in patterns
                        )
                    )
                    per_condition[name]["max_w_proxy"].append(max_w_proxy)
                    per_condition[name]["role_target_basin_hit"].append(
                        bd["role_target_basin_hit"]
                    )
                    per_condition[name]["role_target_rank"].append(
                        bd["role_target_rank"]
                    )
                    per_condition[name]["top_similarity"].append(
                        bd["top_similarity"]
                    )

            def _mean(xs: List[float]) -> float:
                return sum(xs) / len(xs) if xs else float("nan")

            # Paired ΔE = E_content - E_role.
            c = per_condition["content"]["e_min_raw"]
            r = per_condition["role"]["e_min_raw"]
            n_pairs = min(len(c), len(r))
            per_cue_dE_raw = [c[i] - r[i] for i in range(n_pairs)]
            cs3 = per_condition["content"]["e_min_step3"]
            rs3 = per_condition["role"]["e_min_step3"]
            per_cue_dE_step3 = [cs3[i] - rs3[i] for i in range(n_pairs)]

            n_pos = sum(1 for x in per_cue_dE_raw if x > 0)
            mean_dE = _mean(per_cue_dE_raw)
            mean_dE_step3 = _mean(per_cue_dE_step3)

            # Ordering counts at the cue level. For each cue, look at
            # the three condition energies and rank them. Track:
            #   - "role_lt_content_lt_random" (the headline-predicted ordering)
            #   - "role_lt_content" (the weaker form: role beats content
            #     regardless of where random sits)
            #   - "random_lowest" (the pathology to watch: random-prior
            #     produces the lowest energy state)
            n_role_lt_content_lt_random = 0
            n_role_lt_content = 0
            n_random_lowest = 0
            for i in range(n_pairs):
                er = per_condition["role"]["e_min_raw"][i]
                ec = per_condition["content"]["e_min_raw"][i]
                ed = per_condition["random"]["e_min_raw"][i] if i < len(per_condition["random"]["e_min_raw"]) else float("inf")
                if er < ec < ed:
                    n_role_lt_content_lt_random += 1
                if er < ec:
                    n_role_lt_content += 1
                if ed < er and ed < ec:
                    n_random_lowest += 1

            cells.append({
                "binding_noise_std": bns,
                "content_distortion": cd,
                "n_pairs": n_pairs,
                "mean_delta_e_raw": mean_dE,
                "mean_delta_e_step3": mean_dE_step3,
                "frac_positive_raw": n_pos / n_pairs if n_pairs else float("nan"),
                "delta_e_over_floor": mean_dE / 5.5e-3 if n_pairs else float("nan"),
                "frac_role_lt_content_lt_random": (
                    n_role_lt_content_lt_random / n_pairs if n_pairs else float("nan")
                ),
                "frac_role_lt_content": (
                    n_role_lt_content / n_pairs if n_pairs else float("nan")
                ),
                "frac_random_lowest": (
                    n_random_lowest / n_pairs if n_pairs else float("nan")
                ),
                "per_condition_basin_hit_rate": {
                    name: _mean(per_condition[name]["role_target_basin_hit"])
                    for name, _ in condition_specs
                },
                "per_condition_mean_role_target_rank": {
                    name: _mean(per_condition[name]["role_target_rank"])
                    for name, _ in condition_specs
                },
                "per_condition_mean_softmax_entropy": {
                    name: _mean(per_condition[name]["softmax_entropy"])
                    for name, _ in condition_specs
                },
                "per_condition_mean_max_w_proxy": {
                    name: _mean(per_condition[name]["max_w_proxy"])
                    for name, _ in condition_specs
                },
                "per_condition_mean_top_similarity": {
                    name: _mean(per_condition[name]["top_similarity"])
                    for name, _ in condition_specs
                },
                "per_condition_mean_e_min_raw": {
                    name: _mean(per_condition[name]["e_min_raw"])
                    for name, _ in condition_specs
                },
            })

    return {
        "snapshot": str(snapshot_path),
        "label": info.get("label"),
        "n_atoms": len(patterns),
        "dim": int(mem.substrate.dim),
        "coverage_lambda": float(cons.config.coverage_lambda),
        "step3_bias_active": step3_bias is not None,
        "config": {
            "beta": beta,
            "k_main": k_main,
            "gamma": gamma,
            "n_cues_per_cell": n_cues,
            "binding_noise_grid": binding_noise_grid,
            "content_distortion_grid": content_distortion_grid,
            "formulation": formulation,
            "cue_seed": cue_seed,
            "magnitude_floor": 5.5e-3,
        },
        "cells": cells,
    }


def _run_headline_log_prior_sweep(
    *,
    snapshot_path: Path,
    device: str,
    beta: float,
    k_main: int,
    gamma: float,
    gains: List[float],
    binding_noise_std: float,
    content_distortion: float,
    n_cues: int,
    formulation: str,
    cue_seed: int,
    delta_energy: float,
    delta_state: float,
    delta_redundant: float,
    temperature: float,
    max_settling_iter: int,
    schema_source: str = "slow_store",
) -> Dict[str, Any]:
    """Varner-style log-prior spike over a predeclared gain grid.

    Default operating point: beta=10, gamma=0.5, K=1, per-pattern
    formulation, content_distortion=0.6, binding_noise_std=0.05. Required
    controls may intentionally override gamma (for example gamma=0). The
    sweep varies only the additive branch-local log-multiplicity boost
    passed through experiments/40's production branch path unless the caller
    also changes the explicit control flags.
    """
    if formulation != "per_pattern":
        raise SystemExit("--log-prior-sweep requires --headline-formulation per_pattern")

    mem, cons, patterns, positions, info = _exp40._load_substrate_from_snapshot(
        path=str(snapshot_path), device=device,
    )
    if positions is None:
        raise SystemExit(
            f"snapshot {snapshot_path} has no positions; log-prior sweep "
            "requires position vectors."
        )

    patterns_matrix = mem._pattern_matrix()
    if schema_source == "slow_store":
        schema_store, atom_idx = _exp40.get_schema_store(
            consolidation=cons, patterns=patterns_matrix,
            selection_rule="top_k_by_effective_strength",
            k=min(8, len(patterns)),
        )
    elif schema_source == "full_codebook":
        # No-schema-store control: draw priors directly from the stored
        # pattern codebook instead of the filtered slow-store top-k.
        schema_store = patterns_matrix
        atom_idx = None
    else:
        raise SystemExit(
            f"unknown log-prior schema_source {schema_source!r}; expected "
            "'slow_store' or 'full_codebook'"
        )
    schema_bindings = _exp40.compute_schema_bindings(
        substrate=mem.substrate, schemas=schema_store, positions=positions,
    )
    step3_bias = (
        cons.retrieval_weight_bias()
        if cons.config.coverage_lambda > 0.0 else None
    )
    cue_specs = _exp40._build_role_binding_cues(
        substrate=mem.substrate,
        positions=positions,
        patterns=patterns,
        n_cues=n_cues,
        binding_noise_std=binding_noise_std,
        content_distortion=content_distortion,
        seed=cue_seed,
    )

    condition_specs = [("role", "role"), ("content", "content"), ("random", "random")]
    cells: List[Dict[str, Any]] = []
    boltzmann_rng = torch.Generator().manual_seed(13)
    random_prior_rng = torch.Generator().manual_seed(29)

    for gain in gains:
        boltzmann_rng.manual_seed(13)
        random_prior_rng.manual_seed(29)
        per_condition: Dict[str, Dict[str, List[float]]] = {
            name: {
                "e_min_raw": [], "e_min_step3": [],
                "role_target_basin_hit": [], "role_target_rank": [],
                "top_similarity": [],
                "selected_atom_index": [],
            } for name, _ in condition_specs
        }

        for spec in cue_specs:
            for name, prior_type in condition_specs:
                res = _exp40.run_branched_retrieval(
                    cue=spec["cue"], cue_id=0, target_id=spec["role_target_idx"],
                    memory=mem, codebook=patterns_matrix, positions=positions,
                    decode_ids=[], decode_k=5, masked_pos=0,
                    schema_store=schema_store, schema_atom_idx=atom_idx,
                    consolidation=cons, prior_type=prior_type, k_main=k_main,
                    gamma=gamma, beta=beta, temperature=temperature,
                    delta_energy=delta_energy, delta_state=delta_state,
                    delta_redundant=delta_redundant,
                    formulation=formulation,
                    cue_bindings=spec["cue_bindings"],
                    schema_bindings=schema_bindings,
                    include_surprise_branch=False,
                    max_settling_iter=max_settling_iter,
                    boltzmann_rng=boltzmann_rng,
                    random_prior_rng=random_prior_rng,
                    score_bias=step3_bias,
                    log_prior_gain=gain,
                    run_combiners=(k_main > 1),
                )
                if not res.branches:
                    continue
                bs = res.branches
                e_raw = min(b.energy_unbiased for b in bs)
                e_step3 = min(b.energy_unbiased_step3 for b in bs)
                per_condition[name]["e_min_raw"].append(e_raw)
                per_condition[name]["e_min_step3"].append(e_step3)
                for b in bs:
                    if b.schema_atom_index is not None:
                        per_condition[name]["selected_atom_index"].append(
                            float(b.schema_atom_index)
                        )

                use_bundle = k_main > 1 and res.q_bundle is not None
                q_star = res.q_bundle if use_bundle else bs[0].q_settled
                q_star_sims = mem.substrate.similarity_matrix(
                    q_star.to(mem.substrate.device), patterns_matrix,
                )
                bd = _basin_diagnostics_from_sims(
                    sims=q_star_sims,
                    role_target_idx=int(spec["role_target_idx"]),
                )
                per_condition[name]["role_target_basin_hit"].append(
                    bd["role_target_basin_hit"]
                )
                per_condition[name]["role_target_rank"].append(
                    bd["role_target_rank"]
                )
                per_condition[name]["top_similarity"].append(
                    bd["top_similarity"]
                )

        def _mean(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")

        c = per_condition["content"]["e_min_raw"]
        r = per_condition["role"]["e_min_raw"]
        n_pairs = min(len(c), len(r))
        per_cue_dE_raw = [c[i] - r[i] for i in range(n_pairs)]
        cs3 = per_condition["content"]["e_min_step3"]
        rs3 = per_condition["role"]["e_min_step3"]
        per_cue_dE_step3 = [cs3[i] - rs3[i] for i in range(n_pairs)]

        n_role_lt_content_lt_random = 0
        n_role_lt_content = 0
        n_random_lowest = 0
        for i in range(n_pairs):
            er = per_condition["role"]["e_min_raw"][i]
            ec = per_condition["content"]["e_min_raw"][i]
            ed = (
                per_condition["random"]["e_min_raw"][i]
                if i < len(per_condition["random"]["e_min_raw"])
                else float("inf")
            )
            if er < ec < ed:
                n_role_lt_content_lt_random += 1
            if er < ec:
                n_role_lt_content += 1
            if ed < er and ed < ec:
                n_random_lowest += 1

        mean_dE = _mean(per_cue_dE_raw)
        gain_cell = {
            "log_prior_gain": gain,
            "n_pairs": n_pairs,
            "mean_delta_e_raw": mean_dE,
            "mean_delta_e_step3": _mean(per_cue_dE_step3),
            "frac_positive_raw": (
                sum(1 for x in per_cue_dE_raw if x > 0) / n_pairs
                if n_pairs else float("nan")
            ),
            "delta_e_over_floor": mean_dE / 5.5e-3 if n_pairs else float("nan"),
            "frac_role_lt_content_lt_random": (
                n_role_lt_content_lt_random / n_pairs if n_pairs else float("nan")
            ),
            "frac_role_lt_content": (
                n_role_lt_content / n_pairs if n_pairs else float("nan")
            ),
            "frac_random_lowest": (
                n_random_lowest / n_pairs if n_pairs else float("nan")
            ),
            "per_condition_basin_hit_rate": {
                name: _mean(per_condition[name]["role_target_basin_hit"])
                for name, _ in condition_specs
            },
            "per_condition_mean_role_target_rank": {
                name: _mean(per_condition[name]["role_target_rank"])
                for name, _ in condition_specs
            },
            "per_condition_mean_e_min_raw": {
                name: _mean(per_condition[name]["e_min_raw"])
                for name, _ in condition_specs
            },
            "per_condition_mean_top_similarity": {
                name: _mean(per_condition[name]["top_similarity"])
                for name, _ in condition_specs
            },
        }
        cells.append(gain_cell)

    return {
        "snapshot": str(snapshot_path),
        "label": info.get("label"),
        "n_atoms": len(patterns),
        "dim": int(mem.substrate.dim),
        "coverage_lambda": float(cons.config.coverage_lambda),
        "step3_bias_active": step3_bias is not None,
        "config": {
            "beta": beta,
            "k_main": k_main,
            "gamma": gamma,
            "gains": gains,
            "n_cues": n_cues,
            "binding_noise_std": binding_noise_std,
            "content_distortion": content_distortion,
            "formulation": formulation,
            "cue_seed": cue_seed,
            "schema_source": schema_source,
            "schema_store_size": int(schema_store.shape[0]),
            "magnitude_floor": 5.5e-3,
        },
        "cells": cells,
    }


def _flatten_for_csv(record: Dict[str, Any], parent: str = "") -> Dict[str, Any]:
    """One-level flatten of nested dicts: {a: {b: 1}} → {a.b: 1}."""
    out: Dict[str, Any] = {}
    for k, v in record.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten_for_csv(v, key))
        elif isinstance(v, list):
            # Skip lists in CSV (only beta_preflight is nested anyway).
            continue
        else:
            out[key] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--snapshot", type=Path,
        help="Path to one snapshot .pt file.",
    )
    src.add_argument(
        "--snapshot-dir", type=Path,
        help="Directory; recursively audits all *.pt files inside.",
    )
    src.add_argument(
        "--snapshot-list", type=Path,
        help="Text file with one snapshot path per line.",
    )
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path. CSV companion at "
                        "<output>.csv if --csv-also.")
    parser.add_argument("--csv-also", action="store_true",
                        help="Also write a flattened CSV next to the JSON.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--beta-preflight", action="store_true",
                        help="Run the random-cue β-sweep softmax sharpness "
                        "probe (~30s per snapshot at D=4096). Tests "
                        "substrate response geometry from off-substrate cues.")
    parser.add_argument(
        "--headline-beta-sweep", action="store_true",
        help="Run the headline-cue β sweep. Builds role-binding cues "
        "with experiments/40's _build_role_binding_cues and runs the "
        "role/content/random triplet through run_branched_retrieval at "
        "each β. Single-snapshot only (use with --snapshot). "
        "Distinct from --beta-preflight: cued retrievals vs random probes.",
    )
    parser.add_argument("--headline-n-cues", type=int, default=50,
                        help="Cue count for --headline-beta-sweep.")
    parser.add_argument("--headline-k-main", type=int, default=4,
                        help="K_main for headline β sweep.")
    parser.add_argument("--headline-gamma", type=float, default=0.5)
    parser.add_argument("--binding-noise-std", type=float, default=0.05,
                        help="Matches experiments/40 default.")
    parser.add_argument("--content-distortion", type=float, default=0.6,
                        help="Matches experiments/40 default.")
    parser.add_argument("--headline-formulation", type=str,
                        default="per_pattern",
                        choices=["per_pattern", "global_pull"])
    parser.add_argument("--headline-cue-seed", type=int, default=117,
                        help="Cue-builder seed. experiments/40 uses "
                        "args.seed + 100 (= 117 for seed 17) — match it "
                        "for cue distribution parity.")
    parser.add_argument("--headline-temperature", type=float, default=1.0)
    parser.add_argument("--headline-delta-energy", type=float, default=0.1)
    parser.add_argument("--headline-delta-state", type=float, default=0.3)
    parser.add_argument("--headline-delta-redundant", type=float, default=0.95)
    parser.add_argument("--headline-max-settling-iter", type=int, default=12)
    # Cue-regime sweep mode (separate from --headline-beta-sweep):
    # iterates (binding_noise_std × content_distortion) grid at FIXED
    # β / K / γ (the GPT-recommended best operating point post-report
    # 057: β=10, K=1, γ=0.5).
    parser.add_argument(
        "--cue-regime-sweep", action="store_true",
        help="Run the (binding_noise_std × content_distortion) cue grid "
        "at fixed β/K/γ. Single-snapshot only (use with --snapshot). "
        "Per-cell stats include paired ΔE, basin-membership "
        "(role-target hit rate, role-target rank), ordering counts "
        "(role<content<random, role<content, random_lowest), entropy, "
        "and max_w. Distinct from --headline-beta-sweep (which iterates "
        "β at fixed cue regime).",
    )
    parser.add_argument(
        "--log-prior-sweep", action="store_true",
        help="Run the C-first Varner-style log-prior gain sweep at the "
        "locked Phase 5 operating point. Single-snapshot only (use with "
        "--snapshot). Varies only --log-prior-gains.",
    )
    parser.add_argument(
        "--log-prior-gains", type=str, default="0,1,2,4,6",
        help="Comma-separated additive logit gains for --log-prior-sweep.",
    )
    parser.add_argument(
        "--log-prior-beta", type=float, default=10.0,
        help="Fixed beta for --log-prior-sweep. Default 10.0 is locked by "
        "reports 057/058.",
    )
    parser.add_argument(
        "--log-prior-schema-source", type=str, default="slow_store",
        choices=["slow_store", "full_codebook"],
        help="Schema source for --log-prior-sweep. slow_store preserves "
        "the Phase 5 design default; full_codebook is the no-schema-store "
        "control that draws priors directly from the codebook.",
    )
    parser.add_argument(
        "--cue-regime-binding-noise", type=str, default="0.01,0.05,0.10,0.20",
        help="Comma-separated binding_noise_std grid.",
    )
    parser.add_argument(
        "--cue-regime-content-distortion", type=str,
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated content_distortion grid.",
    )
    parser.add_argument(
        "--cue-regime-beta", type=float, default=10.0,
        help="Fixed β for the cue-regime sweep. β=10 confirmed best by "
        "[report 057]'s cross-seed n=10 audit.",
    )
    parser.add_argument(
        "--cue-regime-n-cues", type=int, default=200,
        help="Cues per cell. 200 matches reports 056/057 (K=1 single-snapshot).",
    )
    parser.add_argument("--betas", type=str, default="1,3,5,10,30",
                        help="Comma-separated β values for preflight.")
    parser.add_argument("--n-cue-probes", type=int, default=4,
                        help="Random cues to average preflight stats over.")
    parser.add_argument("--preflight-seed", type=int, default=17,
                        help="RNG seed for the preflight's random cues.")
    args = parser.parse_args()

    if args.headline_beta_sweep and args.snapshot is None:
        raise SystemExit(
            "--headline-beta-sweep requires --snapshot (single-snapshot mode)"
        )
    if args.cue_regime_sweep and args.snapshot is None:
        raise SystemExit(
            "--cue-regime-sweep requires --snapshot (single-snapshot mode)"
        )
    if args.log_prior_sweep and args.snapshot is None:
        raise SystemExit(
            "--log-prior-sweep requires --snapshot (single-snapshot mode)"
        )
    active_sweeps = [
        args.headline_beta_sweep,
        args.cue_regime_sweep,
        args.log_prior_sweep,
    ]
    if sum(bool(x) for x in active_sweeps) > 1:
        raise SystemExit(
            "--headline-beta-sweep, --cue-regime-sweep, and "
            "--log-prior-sweep are mutually exclusive (each iterates a "
            "different axis at fixed others)"
        )

    if args.snapshot is not None:
        snapshots = [args.snapshot]
    elif args.snapshot_dir is not None:
        snapshots = sorted(args.snapshot_dir.rglob("*.pt"))
        if not snapshots:
            raise SystemExit(f"no .pt files under {args.snapshot_dir}")
    else:
        with open(args.snapshot_list) as f:
            snapshots = [Path(line.strip()) for line in f if line.strip()]

    betas = [float(b) for b in args.betas.split(",") if b.strip()]

    if args.log_prior_sweep:
        gains = [float(g) for g in args.log_prior_gains.split(",") if g.strip()]
        print(
            f"[log-prior sweep] {snapshots[0]}  "
            f"(gains={gains}, beta={args.log_prior_beta}, "
            f"K={args.headline_k_main}, gamma={args.headline_gamma}, "
            f"n_cues={args.headline_n_cues})",
            flush=True,
        )
        sweep = _run_headline_log_prior_sweep(
            snapshot_path=snapshots[0],
            device=args.device,
            beta=args.log_prior_beta,
            k_main=args.headline_k_main,
            gamma=args.headline_gamma,
            gains=gains,
            binding_noise_std=args.binding_noise_std,
            content_distortion=args.content_distortion,
            n_cues=args.headline_n_cues,
            formulation=args.headline_formulation,
            cue_seed=args.headline_cue_seed,
            delta_energy=args.headline_delta_energy,
            delta_state=args.headline_delta_state,
            delta_redundant=args.headline_delta_redundant,
            temperature=args.headline_temperature,
            max_settling_iter=args.headline_max_settling_iter,
            schema_source=args.log_prior_schema_source,
        )
        geometry = audit_snapshot(
            snapshot_path=snapshots[0],
            device=args.device,
            beta_preflight=False,
            betas=betas,
            n_cue_probes=args.n_cue_probes,
            preflight_seed=args.preflight_seed,
        )
        out_doc = {"geometry": geometry, "log_prior_sweep": sweep}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out_doc, f, indent=2)
        print(f"[done] wrote {args.output}")
        return

    if args.cue_regime_sweep:
        bns_grid = [float(b) for b in args.cue_regime_binding_noise.split(",") if b.strip()]
        cd_grid = [float(c) for c in args.cue_regime_content_distortion.split(",") if c.strip()]
        print(
            f"[cue-regime sweep] {snapshots[0]}  "
            f"({len(bns_grid)} × {len(cd_grid)} = {len(bns_grid) * len(cd_grid)} cells, "
            f"β={args.cue_regime_beta}, K={args.headline_k_main}, "
            f"γ={args.headline_gamma}, n_cues={args.cue_regime_n_cues})",
            flush=True,
        )
        sweep = _run_headline_cue_regime_sweep(
            snapshot_path=snapshots[0],
            device=args.device,
            beta=args.cue_regime_beta,
            k_main=args.headline_k_main,
            gamma=args.headline_gamma,
            binding_noise_grid=bns_grid,
            content_distortion_grid=cd_grid,
            n_cues=args.cue_regime_n_cues,
            formulation=args.headline_formulation,
            cue_seed=args.headline_cue_seed,
            delta_energy=args.headline_delta_energy,
            delta_state=args.headline_delta_state,
            delta_redundant=args.headline_delta_redundant,
            temperature=args.headline_temperature,
            max_settling_iter=args.headline_max_settling_iter,
        )
        geometry = audit_snapshot(
            snapshot_path=snapshots[0],
            device=args.device,
            beta_preflight=False,
            betas=betas,
            n_cue_probes=args.n_cue_probes,
            preflight_seed=args.preflight_seed,
        )
        out_doc = {"geometry": geometry, "cue_regime_sweep": sweep}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out_doc, f, indent=2)
        print(f"[done] wrote {args.output}")
        return

    # Headline β sweep is a separate output shape (per-β aggregates) so
    # we branch the writer here. The standard audit_snapshot loop still
    # runs underneath when --headline-beta-sweep is set, so the geometry
    # measurements travel with the sweep results.
    if args.headline_beta_sweep:
        print(f"[headline β sweep] {snapshots[0]}", flush=True)
        sweep = _run_headline_beta_sweep(
            snapshot_path=snapshots[0],
            device=args.device,
            betas=betas,
            n_cues=args.headline_n_cues,
            k_main=args.headline_k_main,
            gamma=args.headline_gamma,
            binding_noise_std=args.binding_noise_std,
            content_distortion=args.content_distortion,
            formulation=args.headline_formulation,
            cue_seed=args.headline_cue_seed,
            delta_energy=args.headline_delta_energy,
            delta_state=args.headline_delta_state,
            delta_redundant=args.headline_delta_redundant,
            temperature=args.headline_temperature,
            max_settling_iter=args.headline_max_settling_iter,
        )
        # Also include the standard geometry audit for context.
        geometry = audit_snapshot(
            snapshot_path=snapshots[0],
            device=args.device,
            beta_preflight=args.beta_preflight,
            betas=betas,
            n_cue_probes=args.n_cue_probes,
            preflight_seed=args.preflight_seed,
        )
        out_doc = {"geometry": geometry, "headline_beta_sweep": sweep}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out_doc, f, indent=2)
        print(f"[done] wrote {args.output}")
        return

    records: List[Dict[str, Any]] = []
    for path in snapshots:
        print(f"[audit] {path}", flush=True)
        records.append(audit_snapshot(
            snapshot_path=path,
            device=args.device,
            beta_preflight=args.beta_preflight,
            betas=betas,
            n_cue_probes=args.n_cue_probes,
            preflight_seed=args.preflight_seed,
        ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"snapshots": records, "n": len(records)}, f, indent=2)
    print(f"[done] wrote {args.output}")

    if args.csv_also:
        csv_path = args.output.with_suffix(args.output.suffix + ".csv")
        flat_rows = [_flatten_for_csv(r) for r in records]
        # Union of all keys across rows so the CSV header is stable
        # even when some snapshots have coverage_lambda=0 (missing bias fields).
        all_keys: List[str] = []
        seen: set = set()
        for row in flat_rows:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for row in flat_rows:
                w.writerow(row)
        print(f"[done] wrote {csv_path}")


if __name__ == "__main__":
    main()
