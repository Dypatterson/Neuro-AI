"""C.3 — Phase 3 exit-criterion re-run on the instrumented codebook.

This is the **final Path C deliverable** per the precommit at
``notes/notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md``.

Headline metric (per ``notes/emergent-codebook/phase-3-deep-dive.md:180-189``):
**Regime-stratified Recall@K on masked-token contextual completion, vs.
genuine shuffled-token control, on n ≥ 10 seeds.**

This driver is the C.3 graduation gate when run with ``--seeds``
containing 10+ entries. With fewer seeds the run is *diagnostic-only* and
the report header records that explicitly.

The driver:
  1. For each seed, generates a fresh **random codebook** as the main
     condition's token-to-hypervector assignment.
  2. Memorizes a small Hopfield landscape of masked-token windows over
     a synthetic source corpus (Phase 2 operating envelope).
  3. Computes regime diagnostics via
     ``compute_codebook_regime_diagnostics`` from C.1.3 to label each
     atom 'tight' / 'spread' / 'borderline'.
  4. Evaluates Recall@K on held-out windows, **stratified by the masked
     token's atom regime**.
  5. Repeats steps 1-4 with a **fresh shuffled-token control codebook**
     drawn from a disjoint seed prefix (``seed + 10000``). This is the
     genuine shuffled-token control the Path C precommit demands —
     **NOT** artifact reuse from Report 017.
  6. Runs both ``theta_prime_fn`` variants (default ``1/β`` and the
     calibrated lookup) per the Path C precommit binding.
  7. Aggregates per (theta_prime_mode, stratum) across seeds with
     Wilson CIs.

This driver does **not** modify any existing ``src/`` file. It only
calls existing primitives.

Note: at the Phase 2 operating point β=10, the calibration table at
``notes/emergent-codebook/theta_prime_calibration.json`` only covers
β ∈ {0.01, 0.1, 1.0}. The calibrated loader will warn and fall back to
``1/β`` outside the calibrated range. The two ``theta_prime_mode``
runs therefore agree numerically at β=10 unless the user supplies a
β inside the calibrated range. This is itself a binding finding to
report — see the smoke output.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.encoding import (
    build_position_vectors,
    encode_window,
    masked_window,
)
from energy_memory.phase2.metrics import wilson_interval
from energy_memory.phase3.regime_diagnostic import (
    CodebookRegimeDiagnostics,
    compute_codebook_regime_diagnostics,
)
from energy_memory.phase3.theta_prime_calibration import load_theta_prime_calibration
from energy_memory.substrate.torch_fhrr import TorchFHRR


# Sentinel constants -----------------------------------------------------------

STRATA = ("tight", "spread", "borderline")
THETA_PRIME_MODES = ("default", "calibrated")


# Synthetic corpus -------------------------------------------------------------

def _make_synthetic_windows(
    *,
    n_windows: int,
    window_size: int,
    vocab_size: int,
    rng: random.Random,
) -> List[Tuple[int, ...]]:
    """Generate ``n_windows`` random windows of length ``window_size``.

    The "synthetic source distribution" here is uniform-random token ids
    in [0, vocab_size). This intentionally creates a corpus with no
    learned structure — the test of whether the random codebook supports
    contextual completion is whether *positional* binding alone can
    drive masked-token retrieval on memorized windows.

    For Phase 3's exit criterion, the *shuffled-token control* is the
    fresh-codebook permutation; the corpus itself does not need to carry
    structure for the headline to be interpretable, because what is
    being measured is whether regime-stratified Recall@K of the
    "standard" codebook beats the shuffled-codebook control.
    """
    windows: List[Tuple[int, ...]] = []
    for _ in range(n_windows):
        windows.append(tuple(rng.randrange(0, vocab_size) for _ in range(window_size)))
    return windows


# Codebook generation ----------------------------------------------------------

def _generate_codebook(*, substrate: TorchFHRR, vocab_size: int) -> torch.Tensor:
    """Generate a fresh random FHRR codebook of ``vocab_size`` atoms.

    Uses the substrate's own RNG (set via the constructor seed). The
    `random_vectors` call draws from `substrate.generator`, so two
    substrates constructed with different seeds produce disjoint
    codebooks. This is the operationalization of the "shuffled-token
    control" — the control gets a fresh substrate seed and therefore a
    fresh assignment of token ids to hypervectors.
    """
    return substrate.random_vectors(vocab_size)


# Recall@K evaluation ----------------------------------------------------------

@dataclass
class _CellOutcome:
    """Per-test-window evaluation outcome for one (mode, seed) cell."""

    masked_token: int
    masked_regime: str
    correct_at_k: bool


def _evaluate_recall_at_k(
    *,
    substrate: TorchFHRR,
    memory: TorchHopfieldMemory[str],
    codebook: torch.Tensor,
    positions: Sequence,
    test_windows: Sequence[Tuple[int, ...]],
    masked_idx: int,
    beta: float,
    k: int,
    regime_labels: Dict[int, str],
) -> List[_CellOutcome]:
    """Compute Recall@K for each held-out test window.

    For each test window:
      1. Mask the token at ``masked_idx`` (the last position by default).
      2. Encode the cue (window with mask placeholder).
      3. Hopfield-retrieve the settled state at temperature ``beta``.
      4. Unbind the masked position to get the cleaned-up query.
      5. Rank all codebook atoms by cosine similarity with the query.
      6. Mark "correct" if the true masked token is in the top-K.
      7. Tag with the regime label of the masked token's atom.

    No `if regime then route` logic anywhere — the regime is only used
    as a passive stratification label at aggregation time.
    """
    outcomes: List[_CellOutcome] = []
    vocab_size = codebook.shape[0]
    mask_id = vocab_size  # sentinel — out of vocab; encoded as zero-vector

    # Build a mask-id codebook entry by extending with a single neutral
    # vector. We use a random vector for the mask to keep the encoding
    # well-formed; the cue's masked position contributes a structural
    # binding but no information about the true token.
    mask_vector = substrate.random_vector()

    extended_codebook_list = [codebook[i] for i in range(vocab_size)] + [mask_vector]

    for window in test_windows:
        true_token = window[masked_idx]
        regime = regime_labels.get(true_token, "borderline")

        # Build the masked cue.
        cue_window = list(window)
        cue_window[masked_idx] = mask_id

        # Encode using the extended codebook (mask at end).
        terms = []
        for pos_idx, tok_id in enumerate(cue_window):
            terms.append(substrate.bind(positions[pos_idx], extended_codebook_list[tok_id]))
        cue = substrate.bundle(terms)

        # Hopfield-retrieve the settled state.
        result = memory.retrieve(cue, beta=beta, max_iter=12)
        settled_state = result.state

        # Unbind the masked position to get the cleaned query for that slot.
        slot_query = substrate.unbind(settled_state, positions[masked_idx])

        # Rank codebook atoms by similarity to the slot query.
        scores = substrate.similarity_matrix(slot_query, codebook)
        topk = torch.topk(scores, k=min(k, vocab_size)).indices.detach().cpu().tolist()

        outcomes.append(
            _CellOutcome(
                masked_token=true_token,
                masked_regime=regime,
                correct_at_k=(true_token in topk),
            )
        )
    return outcomes


def _build_theta_prime_fn(
    mode: str,
    repo_root: Path,
) -> Tuple[Optional[Callable[[float], float]], Dict[str, object]]:
    """Return the ``theta_prime_fn`` for the given mode plus metadata.

    Returns ``(fn, info_dict)``. When ``mode == 'default'`` returns
    ``(None, {...})`` — the regime-diagnostic API treats ``None`` as
    the spec's ``1/β`` default. When ``mode == 'calibrated'`` returns
    the loaded callable from ``load_theta_prime_calibration()``.
    """
    if mode == "default":
        return None, {"mode": "default", "fn": "1/beta", "source": "spec_default"}
    if mode == "calibrated":
        # Calibration JSON lives at notes/emergent-codebook/theta_prime_calibration.json
        # at the repo root. The loader resolves the default path itself
        # when called with no argument.
        fn = load_theta_prime_calibration()
        if fn is None:
            return None, {
                "mode": "calibrated",
                "fn": "1/beta (fallback — calibration file missing)",
                "source": "calibration_missing",
            }
        return fn, {
            "mode": "calibrated",
            "fn": "JSON lookup with log-beta interpolation; out-of-range falls back to 1/beta",
            "source": "calibration_loaded",
        }
    raise ValueError(f"unknown theta_prime_mode: {mode!r}")


def _aggregate_by_stratum(
    outcomes: Sequence[_CellOutcome],
) -> Dict[str, Tuple[int, int]]:
    """Return per-stratum (successes, trials) counts."""
    succ: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    for o in outcomes:
        total[o.masked_regime] += 1
        if o.correct_at_k:
            succ[o.masked_regime] += 1
    out: Dict[str, Tuple[int, int]] = {}
    for stratum in STRATA:
        out[stratum] = (succ.get(stratum, 0), total.get(stratum, 0))
    return out


def _wilson(successes: int, trials: int) -> Tuple[float, float, float]:
    """Return (mean, lower, upper). Mean is 0.0 if trials==0."""
    if trials == 0:
        return (0.0, 0.0, 0.0)
    lo, hi = wilson_interval(successes, trials)
    return (successes / trials, lo, hi)


# Run loop ---------------------------------------------------------------------

def _run_single_seed_condition(
    *,
    seed: int,
    is_control: bool,
    theta_prime_mode: str,
    D: int,
    landscape_size: int,
    window_size: int,
    n_test_windows: int,
    vocab_size: int,
    n_train_windows: int,
    beta: float,
    k: int,
    device: str,
    repo_root: Path,
) -> Dict[str, object]:
    """Run one (seed, mode, condition) cell.

    The shuffled-token control is operationalized by giving the control
    a **disjoint substrate seed** (``seed + 10000``). That fresh seed
    drives both:
      - the substrate's RNG (so the codebook is freshly drawn), and
      - the corpus RNG used here for window sampling.

    The window distribution used for training and test is held fixed
    across the (main, control) pair *within a seed* — same training
    windows, same test windows — so the only difference between main
    and control is the token→hypervector assignment. This is the
    canonical "shuffled-token" intervention.
    """
    # Substrate / corpus seeds. Main and control share the corpus draws
    # (so identical windows are evaluated) but disagree on the codebook.
    corpus_rng = random.Random(seed)  # shared by main + control
    if is_control:
        substrate_seed = seed + 10000
    else:
        substrate_seed = seed

    substrate = TorchFHRR(dim=D, seed=substrate_seed, device=device)
    codebook = _generate_codebook(substrate=substrate, vocab_size=vocab_size)
    positions = build_position_vectors(substrate, window_size)

    # Generate corpus (training + test windows). Both main and control
    # see the same training-window indices and the same test windows.
    # ``train_windows`` is sampled down to ``landscape_size`` for the
    # Hopfield landscape.
    train_windows = _make_synthetic_windows(
        n_windows=n_train_windows,
        window_size=window_size,
        vocab_size=vocab_size,
        rng=corpus_rng,
    )
    if landscape_size > len(train_windows):
        landscape_size = len(train_windows)
    landscape_windows = train_windows[:landscape_size]

    test_windows = _make_synthetic_windows(
        n_windows=n_test_windows,
        window_size=window_size,
        vocab_size=vocab_size,
        rng=corpus_rng,
    )

    # Memorize the landscape into Hopfield memory.
    memory = TorchHopfieldMemory[str](substrate)
    for index, window in enumerate(landscape_windows):
        memory.store(
            encode_window(substrate, positions, codebook, list(window)),
            label=f"window_{index}",
        )

    # Compute regime diagnostics on this codebook.
    theta_fn, theta_info = _build_theta_prime_fn(theta_prime_mode, repo_root)
    regime_diag: CodebookRegimeDiagnostics = compute_codebook_regime_diagnostics(
        codebook, k_nn=min(8, vocab_size - 1), beta=beta, theta_prime_fn=theta_fn,
    )
    regime_labels: Dict[int, str] = {
        atom_id: ar.regime for atom_id, ar in regime_diag.per_atom.items()
    }

    # Recall@K with the masked position at the last index of each window.
    masked_idx = window_size - 1
    outcomes = _evaluate_recall_at_k(
        substrate=substrate,
        memory=memory,
        codebook=codebook,
        positions=positions,
        test_windows=test_windows,
        masked_idx=masked_idx,
        beta=beta,
        k=k,
        regime_labels=regime_labels,
    )

    per_stratum = _aggregate_by_stratum(outcomes)

    return {
        "seed": seed,
        "substrate_seed": substrate_seed,
        "is_control": is_control,
        "theta_prime_mode": theta_prime_mode,
        "theta_prime_info": theta_info,
        "regime_counts": dict(regime_diag.regime_counts),
        "regime_summary": regime_diag.summary,
        "per_stratum": {
            stratum: {
                "successes": int(s),
                "trials": int(t),
                "recall_at_k": (s / t) if t > 0 else 0.0,
            }
            for stratum, (s, t) in per_stratum.items()
        },
        "n_outcomes": len(outcomes),
    }


def run(
    *,
    seeds: Sequence[int],
    D: int,
    landscape_size: int,
    window_size: int,
    n_test_windows: int,
    n_train_windows: int,
    vocab_size: int,
    k: int,
    beta: float,
    theta_prime_mode: str,
    device: str,
    output_dir: Path,
    repo_root: Path,
) -> dict:
    """Run the full C.3 experiment.

    Iterates over ``theta_prime_mode`` ∈ {default, calibrated} when
    ``theta_prime_mode == 'both'``. For each (mode, seed) pair runs the
    main and shuffled-token-control condition. Returns the assembled
    summary dict written to ``c3_summary.json``.
    """
    if theta_prime_mode == "both":
        modes = list(THETA_PRIME_MODES)
    else:
        modes = [theta_prime_mode]

    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_cell_rows: List[Dict[str, object]] = []

    for mode in modes:
        for seed in seeds:
            for is_control in (False, True):
                row = _run_single_seed_condition(
                    seed=seed,
                    is_control=is_control,
                    theta_prime_mode=mode,
                    D=D,
                    landscape_size=landscape_size,
                    window_size=window_size,
                    n_test_windows=n_test_windows,
                    vocab_size=vocab_size,
                    n_train_windows=n_train_windows,
                    beta=beta,
                    k=k,
                    device=device,
                    repo_root=repo_root,
                )
                per_cell_rows.append(row)

    # Aggregate per (mode, condition, stratum) across seeds. Pool all
    # trial counts across seeds before computing the Wilson interval;
    # this is the per-stratum n.
    aggregated: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for mode in modes:
        aggregated[mode] = {}
        for is_control in (False, True):
            condition_key = "shuffled_control" if is_control else "standard"
            aggregated[mode][condition_key] = {}
            for stratum in STRATA:
                tot_s = 0
                tot_t = 0
                for row in per_cell_rows:
                    if row["theta_prime_mode"] != mode:
                        continue
                    if row["is_control"] != is_control:
                        continue
                    cell = row["per_stratum"][stratum]
                    tot_s += int(cell["successes"])
                    tot_t += int(cell["trials"])
                mean_v, lo, hi = _wilson(tot_s, tot_t)
                aggregated[mode][condition_key][stratum] = {
                    "successes": tot_s,
                    "trials": tot_t,
                    "recall_at_k": mean_v,
                    "wilson_lower": lo,
                    "wilson_upper": hi,
                }

        # Per-stratum delta and CI: delta is standard - shuffled_control.
        # CI for the delta is the standard's CI shifted by the control's
        # point estimate — *not* a paired test (different test windows
        # under each codebook draw share the same indices but evaluate
        # under different codebooks, which makes a clean paired analysis
        # subtle). For the smoke we report both CIs honestly and the
        # point-estimate delta.
        aggregated[mode]["delta_standard_minus_control"] = {}
        for stratum in STRATA:
            std_cell = aggregated[mode]["standard"][stratum]
            ctrl_cell = aggregated[mode]["shuffled_control"][stratum]
            delta = std_cell["recall_at_k"] - ctrl_cell["recall_at_k"]
            aggregated[mode]["delta_standard_minus_control"][stratum] = {
                "delta_recall_at_k": delta,
                "standard_trials": std_cell["trials"],
                "control_trials": ctrl_cell["trials"],
                # Disjoint-CI gate: standard's lower > control's upper.
                "ci_disjoint_standard_beats_control": (
                    std_cell["wilson_lower"] > ctrl_cell["wilson_upper"]
                ),
            }

    summary = {
        "header": {
            "diagnostic_not_graduation": len(seeds) < 10,
            "n_seeds": len(seeds),
            "seeds": list(seeds),
            "phase_3_exit_criterion_source": (
                "notes/emergent-codebook/phase-3-deep-dive.md:180-189"
            ),
            "path_c_precommit_source": (
                "notes/notes/2026-05-26-path-c-phase3-diagnostic-backfill-precommit.md"
            ),
            "headline_metric": (
                "Regime-stratified Recall@K on masked-token contextual "
                "completion vs. genuine shuffled-token control"
            ),
            "graduation_gate_n_seeds": 10,
            "operating_point": {
                "D": D,
                "landscape_size": landscape_size,
                "window_size": window_size,
                "vocab_size": vocab_size,
                "n_train_windows": n_train_windows,
                "n_test_windows": n_test_windows,
                "beta": beta,
                "K": k,
            },
            "theta_prime_modes_run": modes,
            "device": device,
            "wall_clock_seconds": None,  # filled in below
        },
        "per_cell_rows": per_cell_rows,
        "aggregated": aggregated,
    }

    elapsed = time.time() - start
    summary["header"]["wall_clock_seconds"] = elapsed

    return summary


# Reporting --------------------------------------------------------------------

def _format_markdown(summary: dict) -> str:
    header = summary["header"]
    aggregated = summary["aggregated"]
    lines: List[str] = []
    lines.append("# C.3 — Phase 3 Exit Criterion Re-run")
    lines.append("")
    if header["diagnostic_not_graduation"]:
        lines.append(
            "**DIAGNOSTIC, NOT GRADUATION EVIDENCE.** "
            f"This run used n_seeds={header['n_seeds']} (< 10). The "
            "Phase 3 exit criterion requires n ≥ 10 seeds; the smoke "
            "output below is diagnostic-only and CIs are wide."
        )
    else:
        lines.append(
            f"Run with n_seeds={header['n_seeds']} (≥ 10). "
            "Graduation gate eligible if at least one stratum's standard "
            "Wilson lower > control Wilson upper."
        )
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    op = header["operating_point"]
    lines.append(f"- D = `{op['D']}` (Phase 2 baseline envelope)")
    lines.append(f"- landscape_size = `{op['landscape_size']}`")
    lines.append(f"- window_size = `{op['window_size']}`")
    lines.append(f"- vocab_size = `{op['vocab_size']}`")
    lines.append(f"- n_train_windows = `{op['n_train_windows']}`")
    lines.append(f"- n_test_windows = `{op['n_test_windows']}`")
    lines.append(f"- β = `{op['beta']}`")
    lines.append(f"- K = `{op['K']}`")
    lines.append(f"- seeds = `{header['seeds']}`")
    lines.append(f"- theta_prime_modes = `{header['theta_prime_modes_run']}`")
    lines.append(f"- device = `{header['device']}`")
    lines.append(f"- wall_clock = `{header['wall_clock_seconds']:.1f}s`")
    lines.append("")
    lines.append("## Headline table — per-mode, per-stratum")
    lines.append("")
    lines.append(
        "| Mode | Stratum | Standard Recall@K [Wilson CI] | "
        "Shuffled-control Recall@K [Wilson CI] | Δ (std − ctrl) | "
        "CI-disjoint (std lower > ctrl upper)? | n_std | n_ctrl |"
    )
    lines.append(
        "|---|---|---|---|---:|:--:|---:|---:|"
    )
    for mode in header["theta_prime_modes_run"]:
        for stratum in STRATA:
            std = aggregated[mode]["standard"][stratum]
            ctrl = aggregated[mode]["shuffled_control"][stratum]
            dlt = aggregated[mode]["delta_standard_minus_control"][stratum]
            std_cell = (
                f"{std['recall_at_k']:.3f} "
                f"[{std['wilson_lower']:.3f}, {std['wilson_upper']:.3f}]"
            )
            ctrl_cell = (
                f"{ctrl['recall_at_k']:.3f} "
                f"[{ctrl['wilson_lower']:.3f}, {ctrl['wilson_upper']:.3f}]"
            )
            lines.append(
                f"| {mode} | {stratum} | {std_cell} | {ctrl_cell} | "
                f"{dlt['delta_recall_at_k']:+.3f} | "
                f"{'YES' if dlt['ci_disjoint_standard_beats_control'] else 'no'} | "
                f"{std['trials']} | {ctrl['trials']} |"
            )
    lines.append("")
    lines.append("## Regime classifier agreement diagnostic")
    lines.append("")
    lines.append(
        "Per the C.1.4 calibration finding (`1/β` off by 2-3 orders at "
        "low β), the regime classifier *can* disagree between the "
        "`default` (1/β) and `calibrated` modes. At β=10 the "
        "calibration JSON does not cover the operating point (its grid "
        "is β ∈ {0.01, 0.1, 1.0}), so the calibrated loader falls "
        "back to 1/β with a stderr warning; the two modes therefore "
        "agree numerically at this β."
    )
    lines.append("")
    lines.append("## Per-cell rows (per-seed)")
    lines.append("")
    lines.append(
        "| Seed | Substrate seed | Control? | Mode | tight/spread/borderline counts | tight Recall@K | spread Recall@K | borderline Recall@K |"
    )
    lines.append("|---:|---:|:--:|---|---|---:|---:|---:|")
    for row in summary["per_cell_rows"]:
        rc = row["regime_counts"]
        ps = row["per_stratum"]
        lines.append(
            f"| {row['seed']} | {row['substrate_seed']} | "
            f"{'CTRL' if row['is_control'] else 'STD'} | "
            f"{row['theta_prime_mode']} | "
            f"{rc.get('tight',0)}/{rc.get('spread',0)}/{rc.get('borderline',0)} | "
            f"{ps['tight']['recall_at_k']:.3f} ({ps['tight']['trials']}) | "
            f"{ps['spread']['recall_at_k']:.3f} ({ps['spread']['trials']}) | "
            f"{ps['borderline']['recall_at_k']:.3f} ({ps['borderline']['trials']}) |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_stdout_table(summary: dict) -> str:
    """A short ASCII table of the headline for stdout."""
    header = summary["header"]
    aggregated = summary["aggregated"]
    lines: List[str] = []
    lines.append("=== C.3 HEADLINE ===")
    if header["diagnostic_not_graduation"]:
        lines.append(
            f"[diagnostic, n_seeds={header['n_seeds']} < 10 — not graduation]"
        )
    lines.append(
        f"D={header['operating_point']['D']} "
        f"L={header['operating_point']['landscape_size']} "
        f"W={header['operating_point']['window_size']} "
        f"vocab={header['operating_point']['vocab_size']} "
        f"test_windows={header['operating_point']['n_test_windows']} "
        f"K={header['operating_point']['K']} β={header['operating_point']['beta']}"
    )
    for mode in header["theta_prime_modes_run"]:
        lines.append(f"\n  theta_prime_mode = {mode}")
        lines.append(
            f"    {'stratum':<12} {'std R@K':>9}  {'std CI':>16}  "
            f"{'ctrl R@K':>9}  {'ctrl CI':>16}  {'Δ':>7}  disjoint?"
        )
        for stratum in STRATA:
            std = aggregated[mode]["standard"][stratum]
            ctrl = aggregated[mode]["shuffled_control"][stratum]
            dlt = aggregated[mode]["delta_standard_minus_control"][stratum]
            std_ci = f"[{std['wilson_lower']:.3f},{std['wilson_upper']:.3f}]"
            ctrl_ci = f"[{ctrl['wilson_lower']:.3f},{ctrl['wilson_upper']:.3f}]"
            lines.append(
                f"    {stratum:<12} {std['recall_at_k']:>9.3f}  {std_ci:>16}  "
                f"{ctrl['recall_at_k']:>9.3f}  {ctrl_ci:>16}  "
                f"{dlt['delta_recall_at_k']:>+7.3f}  "
                f"{'YES' if dlt['ci_disjoint_standard_beats_control'] else 'no'}"
            )
    return "\n".join(lines)


def write_outputs(summary: dict, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "c3_summary.json"
    md_path = output_dir / "c3_summary.md"
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    md_path.write_text(_format_markdown(summary), encoding="utf-8")
    return json_path, md_path


def _json_default(o):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


# CLI --------------------------------------------------------------------------

def _parse_seeds(raw: str) -> List[int]:
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    if not parts:
        raise ValueError("expected at least one seed")
    return [int(p) for p in parts]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="C.3 — Phase 3 exit-criterion re-run.",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--D", type=int, default=4096)
    parser.add_argument("--landscape-size", type=int, default=64)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--n-test-windows", type=int, default=200)
    parser.add_argument("--n-train-windows", type=int, default=1000)
    parser.add_argument("--vocab-size", type=int, default=200)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument(
        "--theta-prime-mode",
        choices=["default", "calibrated", "both"],
        default="both",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write c3_summary.{json,md} into. Default: "
            "reports/c3_smoke_<timestamp>/."
        ),
    )

    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    seeds = _parse_seeds(args.seeds)
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = repo_root / "reports" / f"c3_smoke_{ts}"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir

    summary = run(
        seeds=seeds,
        D=args.D,
        landscape_size=args.landscape_size,
        window_size=args.window,
        n_test_windows=args.n_test_windows,
        n_train_windows=args.n_train_windows,
        vocab_size=args.vocab_size,
        k=args.K,
        beta=args.beta,
        theta_prime_mode=args.theta_prime_mode,
        device=args.device,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    json_path, md_path = write_outputs(summary, output_dir)
    print(_format_stdout_table(summary))
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
