"""Gate 0 (Frame A) — matched-world difference-in-differences.

Phase 3 reframe (2026-05-28): does the *existing* consolidation stack
(pull/push + C.2.1–C.2.5 at Path C values) extract corpus co-occurrence
structure, measured against a control that can actually detect it?

The prior C.3 "shuffled-token" control was found gauge-vacuous — a row
permutation of an i.i.d. codebook, so ``E[Δ]=0`` by exchangeability
(``notes/notes/2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md``).
Gate 0 replaces it with a **global token-stream shuffle** (preserves unigram
marginals, destroys co-occurrence) and differences out the Phase-2 landscape
effect via no-consolidation baselines (difference-in-differences), so each
world is self-consistent. Precommit:
``notes/notes/2026-05-28-gate0-frame-a-valid-control-precommit.md``.

GATE / DIAGNOSTIC — does NOT graduate Phase 3 under any outcome. Its job is
to (a) confirm the gauge finding empirically (arm E) and (b) select the
Frame B build path via the precommit branch table.

Arms (atom seeds 0..N-1, paired; E = atom seed 0 × permutation seeds):
  A  real world,     Path C consolidation stack
  B  shuffled world, Path C consolidation stack
  C  real world,     frozen codebook (Phase 2 baseline)
  D  shuffled world, frozen codebook (Phase 2 baseline)
  E  real world,     Path C stack, codebook row-permuted (gauge confirmation)

Primary metric — per-seed DiD (atom seed = independent unit):
  DiD_s = [Recall_A(s) − Recall_C(s)] − [Recall_B(s) − Recall_D(s)]
on stratum-pooled Recall@K. Pass = mean-DiD 95% CI strictly above 0 AND
≥ 70% of seeds with DiD_s > 0. (Uses the per-seed inference in
``c3_phase3_exit_criterion`` — NOT the pseudo-replicated pooled-Wilson gate.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Import the C.3 driver primitives (private helpers reused intentionally).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import c3_phase3_exit_criterion as c3  # noqa: E402

ARMS = ("A", "B", "C", "D", "E")

# (world, is_control, standard_mode) per non-gauge arm. E is handled
# separately (permutation seeds at fixed atom seed 0).
_ARM_SPEC = {
    "A": dict(world="real", is_control=False, standard_mode="consolidated"),
    "B": dict(world="shuffled", is_control=False, standard_mode="consolidated"),
    "C": dict(world="real", is_control=False, standard_mode="frozen"),
    "D": dict(world="shuffled", is_control=False, standard_mode="frozen"),
}


def _overall_recall(row: Dict[str, object]) -> Optional[float]:
    """Stratum-pooled Recall@K for one cell (None if no trials)."""
    s = sum(int(row["per_stratum"][st]["successes"]) for st in c3.STRATA)
    t = sum(int(row["per_stratum"][st]["trials"]) for st in c3.STRATA)
    return (s / t) if t > 0 else None


def _paired_deltas(
    a: Dict[int, Dict[str, object]],
    b: Dict[int, Dict[str, object]],
    seeds: Sequence[int],
) -> List[float]:
    """Per-seed (recall_a − recall_b); skips seeds with an undefined arm."""
    out: List[float] = []
    for s in seeds:
        ra, rb = _overall_recall(a[s]), _overall_recall(b[s])
        if ra is not None and rb is not None:
            out.append(ra - rb)
    return out


def _did_deltas(
    arms: Dict[str, Dict[int, Dict[str, object]]], seeds: Sequence[int]
) -> List[float]:
    """Per-seed DiD = [A−C] − [B−D]; skips seeds with any undefined arm."""
    out: List[float] = []
    for s in seeds:
        rs = {k: _overall_recall(arms[k][s]) for k in ("A", "B", "C", "D")}
        if any(v is None for v in rs.values()):
            continue
        out.append((rs["A"] - rs["C"]) - (rs["B"] - rs["D"]))
    return out


def _classify_verdict(
    did: Dict[str, object],
    a_minus_b: Dict[str, object],
    gauge: Dict[str, object],
    n_seeds: int,
    meaningful_effect: float = 0.02,
) -> str:
    """Map the stats onto the precommit's pre-committed branch table.

    Order: confound (E fails) → pass → the DiD ladder.

    The precommit's "weak" vs "null-cons"/"dead" boundary turns on whether
    the data can CONFIDENTLY rule out a meaningful effect, which requires an
    effect-size floor the precommit left implicit. We make it explicit:
    ``meaningful_effect`` (default 0.02 — the ~Δ magnitude the Report 112
    walk-back chased, i.e. what Frame B would care about). A verdict of
    null-cons/dead is only reached when the relevant 95% CI sits *below* that
    floor (we can rule out a real effect). Otherwise a non-passing,
    not-confidently-zero DiD is **weak** — "real-but-underpowered, NOT a
    redesign trigger → escalate n" — which is the correct call whenever the
    per-seed variance leaves the CI wide (the usual case at n=10).
    """
    if not gauge["passes_4a_and_4b"]:
        return "G0->confound"
    if did["ci95_above_zero"] and did["per_seed_robust_ge_threshold"]:
        return "G0->pass"
    # Can we CONFIDENTLY rule out a meaningful positive DiD? Only if the 95%
    # CI upper bound is below the meaningful-effect floor.
    did_hi = did["ci95_upper"]
    did_confidently_below_floor = (did_hi is not None) and (
        did_hi < meaningful_effect
    )
    if not did_confidently_below_floor:
        return "G0->weak"  # underpowered / positive-but-not-conclusive
    # DiD confidently below the meaningful floor → consolidation carries no
    # corpus-specific signal worth chasing. Landscape, or nothing?
    if a_minus_b["ci95_above_zero"]:
        return "G0->null-cons"  # landscape carries structure, cons does not
    ab_hi = a_minus_b["ci95_upper"]
    if (ab_hi is not None) and (ab_hi < meaningful_effect):
        return "G0->dead"  # whole pipeline confidently captures no structure
    return "G0->weak"  # (A)−(B) also underpowered → escalate before concluding


def run_gate0(
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
    theta_prime_mode: str = "both",
    n_consolidation_events: int = 1000,
    alpha_anti: float = 0.01,
    repulsion_step_size: float = 0.05,
    lr_pull: float = 0.1,
    lr_push: float = 0.05,
    device: str,
    output_dir: Path,
    repo_root: Path,
    corpus_source: str = "wikitext",
    wikitext_name: str = "wikitext-2-raw-v1",
    vocab_cap: int = 1000,
    wikitext_corpus: Optional["c3._WikiTextCorpus"] = None,
    meaningful_effect_floor: float = 0.02,
) -> dict:
    """Run all five Gate 0 arms and assemble the DiD summary."""
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(seeds)

    # The Gate 0 DiD uses stratum-POOLED (overall) Recall@K, which is
    # INVARIANT to theta_prime_mode: the mode only changes how outcomes are
    # partitioned into regime strata (drill-down only — precommit §"Primary
    # metric") and never the correct/total count. The per-cell driver helper
    # takes a SINGLE mode ("both" is expanded at the c3 run() layer, which
    # Gate 0 bypasses), so we run cells at one mode. "both" collapses to
    # "default" — at β=10 the calibrated loader falls back to 1/β anyway, so
    # the two stratifications are numerically identical here.
    cell_theta_mode = "default" if theta_prime_mode == "both" else theta_prime_mode

    # Corpus: Gate 0 is designed for wikitext (real co-occurrence). The
    # synthetic path is supported for plumbing tests (DiD ≈ 0 expected,
    # since synthetic windows have no co-occurrence structure).
    effective_vocab = vocab_size
    corpus_info: Dict[str, object] = {"corpus_source": corpus_source}
    if corpus_source == "wikitext":
        if wikitext_corpus is None:
            wikitext_corpus = c3._load_wikitext_corpus(
                repo_root=repo_root,
                wikitext_name=wikitext_name,
                vocab_cap=vocab_cap,
            )
        effective_vocab = wikitext_corpus.vocab_size
        corpus_info.update(
            {
                "wikitext_name": wikitext_name,
                "vocab_cap": vocab_cap,
                "effective_vocab_size": effective_vocab,
                "n_train_tokens": len(wikitext_corpus.train_ids),
                "n_test_tokens": len(wikitext_corpus.val_ids)
                + len(wikitext_corpus.test_ids),
            }
        )

    def _cell(
        *, seed, world, is_control, standard_mode,
        perm_seed_override=None, identity_permutation=False,
    ):
        return c3._run_single_seed_condition(
            seed=seed,
            is_control=is_control,
            theta_prime_mode=cell_theta_mode,
            standard_mode=standard_mode,
            control_mode="shuffled-token",
            n_consolidation_events=n_consolidation_events,
            D=D,
            landscape_size=landscape_size,
            window_size=window_size,
            n_test_windows=n_test_windows,
            vocab_size=effective_vocab,
            n_train_windows=n_train_windows,
            beta=beta,
            k=k,
            alpha_anti=alpha_anti,
            repulsion_step_size=repulsion_step_size,
            lr_pull=lr_pull,
            lr_push=lr_push,
            device=device,
            repo_root=repo_root,
            wikitext_corpus=wikitext_corpus,
            use_context_residual=False,
            lr_cr=0.1,
            use_pull_push=True,
            world=world,
            perm_seed_override=perm_seed_override,
            identity_permutation=identity_permutation,
        )

    arms: Dict[str, Dict[int, Dict[str, object]]] = {a: {} for a in ARMS}
    for seed in seeds:
        for name in ("A", "B", "C", "D"):
            arms[name][seed] = _cell(seed=seed, **_ARM_SPEC[name])
        # Arm E: gauge confirmation — the old shuffled-token control run
        # PER SEED (fresh atoms X_s + fresh permutation π_s = s+70000), i.e.
        # "run the old gauge control as one of its conditions". Paired with
        # A[s]. This is the comparison whose per-seed Δ the exchangeability
        # proof guarantees has E[Δ]=0 (joint over X_s and π_s).
        arms["E"][seed] = _cell(
            seed=seed, world="real", is_control=True,
            standard_mode="consolidated",
        )

    # --- Primary: per-seed DiD ------------------------------------------
    did = c3._delta_ci_stats(_did_deltas(arms, seeds))
    # --- Secondaries ----------------------------------------------------
    a_minus_b = c3._delta_ci_stats(_paired_deltas(arms["A"], arms["B"], seeds))
    a_minus_c = c3._delta_ci_stats(_paired_deltas(arms["A"], arms["C"], seeds))

    # --- Gauge confirmation (E) -----------------------------------------
    # 4a: identity-permutation gauge control must reproduce A[0] exactly.
    a0 = arms["A"][0]
    e_identity = _cell(
        seed=0, world="real", is_control=True,
        standard_mode="consolidated", identity_permutation=True,
    )
    byte_identical_4a = (
        a0["per_stratum"] == e_identity["per_stratum"]
        and a0["regime_counts"] == e_identity["regime_counts"]
    )
    # 4b (corrected): per-seed Δ = Recall(gauge control) − Recall(A) over
    # the atom seeds. Predicts mean ≈ 0 (the exchangeability claim).
    #
    # NOTE — precommit deviation (surfaced 2026-05-28). The precommit
    # specified arm E as *fixed atom seed 0 × permutation seeds* with the
    # prediction "mean Δ ≈ 0". That is mis-specified: exchangeability gives
    # E[Δ]=0 only over the JOINT (atoms, π) draw. At a fixed atom set X_0,
    # E_π[Recall(P_π X_0)] is the mean recall over all *relabelings* of X_0,
    # which is generally ≠ the identity labeling's recall (= A[0]) — so a
    # nonzero fixed-atom Δ is EXPECTED, not a confound. (Empirically: a
    # fixed-atom×perm probe gave a systematic −0.125, CI excluding 0, while
    # this per-seed version is consistent with 0.) We therefore confirm the
    # gauge per-seed, which is also literally "the old gauge control run as
    # a condition". Pass = mean-Δ 95% CI CONTAINS 0 (fail to reject Δ=0).
    gauge_deltas = _paired_deltas(arms["E"], arms["A"], seeds)
    gauge_4b = c3._delta_ci_stats(gauge_deltas)
    if gauge_4b["ci95_lower"] is not None:
        ci_contains_zero_4b = (
            gauge_4b["ci95_lower"] <= 0.0 <= gauge_4b["ci95_upper"]
        )
    else:  # n<2 seeds: cannot reject; treat near-zero mean as consistent
        ci_contains_zero_4b = abs(gauge_4b["mean_delta"]) < 1e-9
    gauge = {
        "method": "per-seed Recall(shuffled-token control) − Recall(A)",
        "byte_identical_4a": bool(byte_identical_4a),
        "mean_delta_4b": gauge_4b["mean_delta"],
        "sem_delta_4b": gauge_4b["sem_delta"],
        "ci95_4b": [gauge_4b["ci95_lower"], gauge_4b["ci95_upper"]],
        "ci_contains_zero_4b": bool(ci_contains_zero_4b),
        "n_gauge_seeds": gauge_4b["n_seeds_used"],
        "precommit_deviation": (
            "fixed-atom×perm 4b replaced by per-seed gauge — E_π at fixed X "
            "≠ identity recall, so the precommit's ≈0 prediction was "
            "mis-specified; see anchor note / Gate 0 precommit 4b amendment."
        ),
        "passes_4a_and_4b": bool(byte_identical_4a and ci_contains_zero_4b),
        "per_seed": gauge_4b,
    }

    verdict = _classify_verdict(
        did, a_minus_b, gauge, len(seeds),
        meaningful_effect=meaningful_effect_floor,
    )

    summary = {
        "header": {
            "gate": "Gate 0 (Frame A) — matched-world DiD",
            "diagnostic_not_graduation": True,  # Gate 0 NEVER graduates
            "precommit_source": (
                "notes/notes/2026-05-28-gate0-frame-a-valid-control-precommit.md"
            ),
            "anchor_source": (
                "notes/notes/2026-05-28-phase3-frame-b-continual-learning-"
                "and-gauge-control-finding.md"
            ),
            "n_seeds": len(seeds),
            "seeds": list(seeds),
            "operating_point": {
                "D": D,
                "landscape_size": landscape_size,
                "window_size": window_size,
                "vocab_size": effective_vocab,
                "n_train_windows": n_train_windows,
                "n_test_windows": n_test_windows,
                "beta": beta,
                "K": k,
                "n_consolidation_events": n_consolidation_events,
                "theta_prime_mode_requested": theta_prime_mode,
                "theta_prime_mode_used_per_cell": cell_theta_mode,
                "did_is_theta_mode_invariant": True,
            },
            "path_c_stack": {
                "lr_pull": lr_pull,
                "lr_push": lr_push,
                "alpha_anti": alpha_anti,
                "repulsion_step_size": repulsion_step_size,
                "use_pull_push": True,
                "use_context_residual": False,
            },
            "corpus": corpus_info,
            "meaningful_effect_floor": meaningful_effect_floor,
            "device": device,
            "wall_clock_seconds": None,
        },
        "primary_did": {
            "definition": "[Recall_A − Recall_C] − [Recall_B − Recall_D], per seed",
            "stats": did,
            "pass_clause1_ci_above_zero": did["ci95_above_zero"],
            "pass_clause2_robust_ge_70pct": did["per_seed_robust_ge_threshold"],
            "passes": bool(
                did["ci95_above_zero"] and did["per_seed_robust_ge_threshold"]
            ),
        },
        "secondary": {
            "a_minus_b_whole_pipeline": a_minus_b,
            "a_minus_c_consolidation_on_real": a_minus_c,
        },
        "gauge_confirmation_E": gauge,
        "verdict": verdict,
        "per_seed_recall": {
            name: {str(s): _overall_recall(arms[name][s]) for s in seeds}
            for name in ARMS
        },
        "per_cell_rows": {
            name: [arms[name][s] for s in seeds] for name in ARMS
        },
    }
    summary["header"]["wall_clock_seconds"] = time.time() - start
    return summary


# Reporting --------------------------------------------------------------------

def _fmt_ci(st: dict) -> str:
    if st["ci95_lower"] is None:
        return f"{st['mean_delta']:+.4f} [n<2]"
    return (
        f"{st['mean_delta']:+.4f} "
        f"[{st['ci95_lower']:+.4f}, {st['ci95_upper']:+.4f}]"
    )


def format_gate0_markdown(summary: dict) -> str:
    h = summary["header"]
    op = h["operating_point"]
    did = summary["primary_did"]
    lines: List[str] = []
    lines.append("# Gate 0 (Frame A) — matched-world difference-in-differences")
    lines.append("")
    lines.append(
        "> **GATE / DIAGNOSTIC — does NOT graduate Phase 3 under any "
        "outcome.** Confirms the gauge finding (arm E) and selects the "
        "Frame B build path via the precommit branch table."
    )
    lines.append("")
    lines.append(f"- precommit: `{h['precommit_source']}`")
    lines.append(
        f"- operating point: D={op['D']} β={op['beta']} K={op['K']} "
        f"window={op['window_size']} vocab={op['vocab_size']} "
        f"landscape={op['landscape_size']} "
        f"cons_events={op['n_consolidation_events']} "
        f"θ′={op['theta_prime_mode_used_per_cell']} "
        f"(requested {op['theta_prime_mode_requested']}; DiD is θ′-invariant)"
    )
    pc = h["path_c_stack"]
    lines.append(
        f"- Path C stack: lr_pull={pc['lr_pull']} lr_push={pc['lr_push']} "
        f"α_anti={pc['alpha_anti']} repulsion={pc['repulsion_step_size']}"
    )
    cs = h["corpus"]
    lines.append(
        f"- corpus: `{cs.get('corpus_source')}`"
        + (
            f" ({cs.get('wikitext_name')}, vocab_cap={cs.get('vocab_cap')}, "
            f"eff_vocab={cs.get('effective_vocab_size')})"
            if cs.get("corpus_source") == "wikitext"
            else ""
        )
    )
    lines.append(f"- n_seeds = {h['n_seeds']} (graduation-scale ≥ 10)")
    lines.append("")
    lines.append(f"## VERDICT: `{summary['verdict']}`")
    lines.append("")
    lines.append("## Primary — consolidation corpus-specificity (DiD)")
    lines.append("")
    lines.append("`DiD_s = [Recall_A − Recall_C] − [Recall_B − Recall_D]`")
    lines.append("")
    st = did["stats"]
    lines.append(
        f"- mean DiD = **{_fmt_ci(st)}** (n={st['n_seeds_used']})"
    )
    lines.append(
        f"- clause 1 — CI > 0: **{st['ci95_above_zero']}**"
    )
    lines.append(
        f"- clause 2 — per-seed robustness ≥ 70%: "
        f"**{st['per_seed_robust_ge_threshold']}** "
        f"({st['n_seeds_positive']}/{st['n_seeds_used']} positive)"
    )
    lines.append(f"- **DiD passes (both clauses): {did['passes']}**")
    lines.append(f"- per-seed DiD: {[round(d, 4) for d in st['per_seed_deltas']]}")
    lines.append("")
    lines.append("## Secondary reads")
    lines.append("")
    ab = summary["secondary"]["a_minus_b_whole_pipeline"]
    ac = summary["secondary"]["a_minus_c_consolidation_on_real"]
    lines.append(
        f"- (A)−(B) whole-pipeline corpus-sensitivity: {_fmt_ci(ab)} "
        f"(CI>0: {ab['ci95_above_zero']})"
    )
    lines.append(
        f"- (A)−(C) consolidation benefit on real: {_fmt_ci(ac)} "
        f"(CI>0: {ac['ci95_above_zero']})"
    )
    lines.append("")
    g = summary["gauge_confirmation_E"]
    lines.append("## Gauge confirmation (arm E) — predicts Δ ≈ 0")
    lines.append("")
    lines.append(
        f"- **4a** identity-permutation byte-identical to A[0]: "
        f"**{g['byte_identical_4a']}**"
    )
    ci4b = g.get("ci95_4b", [None, None])
    ci4b_str = (
        f"[{ci4b[0]:+.4f}, {ci4b[1]:+.4f}]" if ci4b[0] is not None else "[n<2]"
    )
    lines.append(
        f"- **4b** per-seed Δ = Recall(gauge control) − Recall(A) over "
        f"{g['n_gauge_seeds']} seeds: {g['mean_delta_4b']:+.4f} {ci4b_str}; "
        f"CI contains 0 (consistent with Δ=0): **{g['ci_contains_zero_4b']}**"
    )
    lines.append(
        f"  - *precommit deviation:* {g['precommit_deviation']}"
    )
    lines.append(
        f"- gauge arm passes (4a ∧ 4b): **{g['passes_4a_and_4b']}** "
        f"— if False → `G0->confound`, STOP and re-derive"
    )
    lines.append("")
    lines.append("## Pre-committed branch (all route to Frame B)")
    lines.append("")
    lines.append(
        "| verdict | meaning | next |\n|---|---|---|\n"
        "| G0->pass | consolidation is corpus-specific | build Frame B on "
        "the Path C stack; Γ1/Γ2/Γ3 stay closed |\n"
        "| G0->weak | real but underpowered | escalate n before deciding |\n"
        "| G0->null-cons | landscape carries structure, cons does not | "
        "redesign targeting consolidation, in Frame B |\n"
        "| G0->dead | pipeline captures no structure | deeper redesign |\n"
        "| G0->confound | gauge arm E failed | STOP, re-derive Part 1 |"
    )
    lines.append("")
    return "\n".join(lines)


def reclassify_summary(summary: dict, meaningful_effect: float = 0.02) -> dict:
    """Re-derive the verdict from an already-computed summary (no re-run).

    The verdict is a pure function of the stored DiD / (A)−(B) / gauge
    stats, so a run whose verdict was produced by an older classifier can
    be corrected in place without re-running the (expensive) arms.
    """
    did = summary["primary_did"]["stats"]
    a_minus_b = summary["secondary"]["a_minus_b_whole_pipeline"]
    gauge = summary["gauge_confirmation_E"]
    old = summary.get("verdict")
    new = _classify_verdict(
        did, a_minus_b, gauge, summary["header"]["n_seeds"],
        meaningful_effect=meaningful_effect,
    )
    summary["verdict"] = new
    summary["verdict_reclassified_from"] = old
    summary["header"]["meaningful_effect_floor"] = meaningful_effect
    return summary


# Variance investigation -------------------------------------------------------
#
# Gate 0 returned G0->weak because the per-seed DiD σ (~0.19 on the n=10
# wikitext run) dwarfs the ~+0.02–0.03 effect. These two tools attack that
# bottleneck: (1) decompose the variance of an existing run with NO new
# compute, and (2) a nested atom×window run that separates the codebook-draw
# component from the corpus-window-draw component (which fix applies depends
# on which dominates).


def _mean(xs: Sequence[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def _variance(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _std(xs: Sequence[float]) -> float:
    return _variance(xs) ** 0.5


def _corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = _mean(xs), _mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def variance_report_from_summary(
    summary: dict, effect: float = 0.02
) -> dict:
    """Decompose the per-seed DiD variance of an EXISTING run (no re-run).

    Splits σ(DiD) into its contrasts and compares each to the single-arm
    binomial floor, so we can tell binomial sampling noise from real
    seed-level (codebook×corpus) variance, and which contrast carries it.
    """
    psr = summary["per_seed_recall"]
    op = summary["header"]["operating_point"]
    n_test = op.get("n_test_windows")
    seeds = list(psr["A"].keys())

    def _get(arm, s):
        v = psr[arm].get(s)
        return None if v is None else float(v)

    ac, bd, did, ab = [], [], [], []
    a_vals, b_vals, c_vals, d_vals = [], [], [], []
    for s in seeds:
        A, B, C, D = (_get(x, s) for x in ("A", "B", "C", "D"))
        if None in (A, B, C, D):
            continue
        a_vals.append(A); b_vals.append(B); c_vals.append(C); d_vals.append(D)
        ac.append(A - C); bd.append(B - D); ab.append(A - B)
        did.append((A - C) - (B - D))

    p_bar = _mean(a_vals + b_vals + c_vals + d_vals)
    binom_floor = ((p_bar * (1 - p_bar) / n_test) ** 0.5) if n_test else None

    n = len(did)
    sd_did = _std(did)
    sem_did = sd_did / (n ** 0.5) if n else 0.0
    # Honest n for ~80% power (z≈2.8 combining 1.96+0.84) at the effect size.
    n_for_power = (
        int(((2.8 * sd_did / effect) ** 2) + 0.999) if sd_did > 0 else None
    )

    report = {
        "n_seeds_used": n,
        "did_mean": _mean(did),
        "did_sd": sd_did,
        "did_sem": sem_did,
        "contrast_sd": {
            "A_minus_C_cons_on_real": _std(ac),
            "B_minus_D_cons_on_shuffled": _std(bd),
            "A_minus_B_whole_pipeline": _std(ab),
            "DiD": sd_did,
        },
        "arm_recall_sd": {
            "A": _std(a_vals), "B": _std(b_vals),
            "C": _std(c_vals), "D": _std(d_vals),
        },
        "corr_AC_BD": _corr(ac, bd),
        "mean_recall": p_bar,
        "single_arm_binomial_floor": binom_floor,
        "did_sd_over_binomial_floor": (
            (sd_did / binom_floor) if binom_floor else None
        ),
        "effect_size_floor": effect,
        "n_for_80pct_power_at_effect": n_for_power,
        "diagnosis": _variance_diagnosis(ac, bd, did, binom_floor, effect),
    }
    return report


def _variance_diagnosis(ac, bd, did, binom_floor, effect) -> str:
    sd_did = _std(did)
    if binom_floor and sd_did <= 2.5 * binom_floor:
        return (
            "DiD σ is near the binomial floor — more TEST WINDOWS reduce it. "
            "Effect may simply be ~0."
        )
    parts = []
    if _std(ac) > 0 and _std(bd) > 0:
        parts.append(
            f"consolidation-lift varies strongly by seed in BOTH worlds "
            f"(σ(A−C)={_std(ac):.3f}, σ(B−D)={_std(bd):.3f})"
        )
    c = _corr(ac, bd)
    if c is not None:
        parts.append(
            f"corr(A−C, B−D)={c:+.2f} — "
            + ("the worlds' lifts move together, so the DiD CANCELS much of "
               "the seed variance (good); residual is the real-vs-shuffled "
               "interaction." if c > 0.3 else
               "the worlds' lifts are weakly/anti correlated, so the DiD "
               "does NOT cancel seed variance — pairing on the codebook is "
               "not buying much.")
        )
    parts.append(
        "σ(DiD) ≫ binomial floor → the variance is structural (codebook×corpus "
        "draw), NOT sampling noise; more test windows won't help. The nested "
        "atom×window decomposition (run_variance_decomposition) tells you "
        "whether it's the ATOM draw (→ control-variate on C, or change the "
        "estimand to a slope) or the WINDOW draw (→ average K window draws "
        "per seed)."
    )
    return " ".join(parts)


def run_variance_decomposition(
    *,
    atom_seeds: Sequence[int],
    window_seeds: Sequence[int],
    D: int,
    landscape_size: int,
    window_size: int,
    n_test_windows: int,
    n_train_windows: int,
    vocab_size: int,
    k: int,
    beta: float,
    n_consolidation_events: int,
    alpha_anti: float = 0.01,
    repulsion_step_size: float = 0.05,
    lr_pull: float = 0.1,
    lr_push: float = 0.05,
    device: str = "cpu",
    repo_root: Path = _HERE.parent,
    corpus_source: str = "wikitext",
    wikitext_name: str = "wikitext-2-raw-v1",
    vocab_cap: int = 1000,
    wikitext_corpus: Optional["c3._WikiTextCorpus"] = None,
) -> dict:
    """Nested decomposition of the consolidation-lift L = Recall(A) − Recall(C)
    (real world) over atom_seeds × window_seeds.

    Separates σ(L) into a between-ATOM-draw component and a within-atom
    (corpus-WINDOW-draw + binomial) component via a random-effects one-way
    ANOVA. The dominant component dictates the variance-reduction lever.
    """
    effective_vocab = vocab_size
    if corpus_source == "wikitext" and wikitext_corpus is None:
        wikitext_corpus = c3._load_wikitext_corpus(
            repo_root=repo_root, wikitext_name=wikitext_name, vocab_cap=vocab_cap,
        )
    if wikitext_corpus is not None:
        effective_vocab = wikitext_corpus.vocab_size

    recall_levels: List[float] = []

    def _lift(atom_seed, window_seed):
        common = dict(
            theta_prime_mode="default", control_mode="shuffled-token",
            n_consolidation_events=n_consolidation_events, D=D,
            landscape_size=landscape_size, window_size=window_size,
            n_test_windows=n_test_windows, vocab_size=effective_vocab,
            n_train_windows=n_train_windows, beta=beta, k=k,
            alpha_anti=alpha_anti, repulsion_step_size=repulsion_step_size,
            lr_pull=lr_pull, lr_push=lr_push, device=device,
            repo_root=repo_root, wikitext_corpus=wikitext_corpus,
            use_context_residual=False, lr_cr=0.1, use_pull_push=True,
            world="real", is_control=False,
            window_seed_override=window_seed,
        )
        A = c3._run_single_seed_condition(
            seed=atom_seed, standard_mode="consolidated", **common)
        C = c3._run_single_seed_condition(
            seed=atom_seed, standard_mode="frozen", **common)
        ra, rc = _overall_recall(A), _overall_recall(C)
        if ra is None or rc is None:
            return None
        recall_levels.extend([ra, rc])
        return ra - rc

    # L[a][w]
    L = {a: {w: _lift(a, w) for w in window_seeds} for a in atom_seeds}
    cells = [L[a][w] for a in atom_seeds for w in window_seeds
             if L[a][w] is not None]
    A_count = len(atom_seeds)
    W_count = len(window_seeds)

    grand = _mean(cells)
    atom_means = {a: _mean([v for v in L[a].values() if v is not None])
                  for a in atom_seeds}
    # Random-effects one-way ANOVA (balanced approx; uses W_count per atom).
    ss_between = W_count * sum((atom_means[a] - grand) ** 2 for a in atom_seeds)
    ss_within = sum(
        (L[a][w] - atom_means[a]) ** 2
        for a in atom_seeds for w in window_seeds if L[a][w] is not None
    )
    ms_between = ss_between / (A_count - 1) if A_count > 1 else 0.0
    ms_within = ss_within / (A_count * (W_count - 1)) if W_count > 1 else 0.0
    var_atom = max(0.0, (ms_between - ms_within) / W_count) if W_count else 0.0
    var_within = ms_within  # corpus-window draw + binomial (conflated)
    var_total = _variance(cells)

    # Separate the binomial component out of var_within. L = A−C over the
    # same windows; an upper bound on its binomial variance is the
    # independent-arms bound 2·p(1−p)/n_test at the mean recall level. The
    # residual is the genuine corpus-WINDOW-draw component.
    p_bar = _mean(recall_levels) if recall_levels else 0.0
    var_binom_floor = (
        2.0 * p_bar * (1 - p_bar) / n_test_windows if n_test_windows else 0.0
    )
    var_window_draw = max(0.0, var_within - var_binom_floor)

    # Compare the three structural components: atom-draw, corpus-window-draw,
    # binomial. Binomial is killable by more test windows; window-draw by
    # averaging window draws per seed; atom-draw only by control-variates /
    # a slope estimand / large n.
    comps = {
        "atom-draw": var_atom,
        "window-draw": var_window_draw,
        "binomial": min(var_binom_floor, var_within),
    }
    dominant = max(comps, key=comps.get)
    recs = {
        "atom-draw": (
            "Between-CODEBOOK-DRAW variance dominates. Averaging window draws "
            "or adding test windows will NOT help. Levers: (a) control-variate "
            "/ regression adjustment using the frozen-codebook recall C as a "
            "covariate (CUPED-style, a re-analysis of existing data); (b) "
            "change the estimand to an exposure–recall SLOPE (Frame B) which "
            "pools within-seed and is higher-SNR; (c) accept honest power needs "
            "large n."
        ),
        "window-draw": (
            "Corpus-WINDOW-draw variance dominates. Averaging K independent "
            "window draws per atom seed reduces σ ~√K — a cheap win before "
            "scaling atom-seed n."
        ),
        "binomial": (
            "Binomial sampling noise dominates at this n_test — increase "
            "n_test_windows to shrink it, then re-decompose to see the real "
            "structural split. (Likely an artifact of a small test set or a "
            "too-simple corpus; the real wikitext op point at n_test≥512 has "
            "structural σ ≫ binomial.)"
        ),
    }

    return {
        "kind": "variance_decomposition_consolidation_lift_A_minus_C_real",
        "atom_seeds": list(atom_seeds),
        "window_seeds": list(window_seeds),
        "operating_point": {
            "D": D, "beta": beta, "K": k, "window_size": window_size,
            "vocab_size": effective_vocab, "landscape_size": landscape_size,
            "n_consolidation_events": n_consolidation_events,
            "n_test_windows": n_test_windows, "corpus_source": corpus_source,
        },
        "L_matrix": {str(a): {str(w): L[a][w] for w in window_seeds}
                     for a in atom_seeds},
        "grand_mean_lift": grand,
        "mean_recall_level": p_bar,
        "sd_total": var_total ** 0.5,
        "sd_between_atom": var_atom ** 0.5,
        "sd_within_atom_window_binom": var_within ** 0.5,
        "sd_window_draw_binom_subtracted": var_window_draw ** 0.5,
        "sd_binomial_floor_estimate": min(var_binom_floor, var_within) ** 0.5,
        "variance_components": {
            "atom_draw": var_atom,
            "window_draw": var_window_draw,
            "binomial": min(var_binom_floor, var_within),
        },
        "var_fraction_atom": (
            var_atom / (var_atom + var_within)
            if (var_atom + var_within) > 0 else None
        ),
        "dominant_source": dominant,
        "recommendation": recs[dominant],
    }


def write_gate0_outputs(summary: dict, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gate0_summary.json"
    md_path = output_dir / "gate0_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    md_path.write_text(format_gate0_markdown(summary))
    return json_path, md_path


def _parse_seeds(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(",", " ").split()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Gate 0 (Frame A) matched-world DiD.")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--D", type=int, default=4096)
    p.add_argument("--landscape-size", type=int, default=64)
    p.add_argument("--window", type=int, default=8, dest="window_size")
    p.add_argument("--n-test-windows", type=int, default=512)
    p.add_argument("--n-train-windows", type=int, default=2048)
    p.add_argument("--vocab-size", type=int, default=200,
                   help="synthetic-corpus vocab; ignored under wikitext.")
    p.add_argument("--K", type=int, default=5, dest="k")
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--theta-prime-mode", default="both",
                   choices=("default", "calibrated", "both"))
    p.add_argument("--n-consolidation-events", type=int, default=1000)
    p.add_argument("--alpha-anti", type=float, default=0.01)
    p.add_argument("--repulsion-step-size", type=float, default=0.05)
    p.add_argument("--lr-pull", type=float, default=0.1)
    p.add_argument("--lr-push", type=float, default=0.05)
    p.add_argument("--device", default="cpu")
    p.add_argument("--corpus-source", default="wikitext",
                   choices=("synthetic", "wikitext"))
    p.add_argument("--wikitext-name", default="wikitext-2-raw-v1")
    p.add_argument("--vocab-cap", type=int, default=1000)
    p.add_argument("--meaningful-effect-floor", type=float, default=0.02,
                   help="DiD effect-size floor for null-cons/dead vs weak.")
    p.add_argument("--reclassify", default=None,
                   help="Path to an existing gate0_summary.json: re-derive "
                        "the verdict (no re-run) and rewrite outputs.")
    p.add_argument("--variance-report", default=None,
                   help="Path to an existing gate0_summary.json: decompose "
                        "the per-seed DiD variance (no re-run).")
    p.add_argument("--variance-decomp", action="store_true",
                   help="Run the nested atom×window variance decomposition "
                        "of the consolidation lift (A−C).")
    p.add_argument("--atom-seeds", default="0,1,2,3")
    p.add_argument("--window-seeds", default="0,1,2,3")
    p.add_argument("--output-dir", default="reports/gate0")
    args = p.parse_args(argv)

    # Analyze an existing run's DiD variance — no re-run.
    if args.variance_report:
        summary = json.loads(Path(args.variance_report).read_text())
        rep = variance_report_from_summary(summary, args.meaningful_effect_floor)
        print(json.dumps(rep, indent=2, default=str))
        return 0

    # Nested atom×window decomposition of the consolidation lift.
    if args.variance_decomp:
        rep = run_variance_decomposition(
            atom_seeds=_parse_seeds(args.atom_seeds),
            window_seeds=_parse_seeds(args.window_seeds),
            D=args.D, landscape_size=args.landscape_size,
            window_size=args.window_size, n_test_windows=args.n_test_windows,
            n_train_windows=args.n_train_windows, vocab_size=args.vocab_size,
            k=args.k, beta=args.beta,
            n_consolidation_events=args.n_consolidation_events,
            alpha_anti=args.alpha_anti,
            repulsion_step_size=args.repulsion_step_size,
            lr_pull=args.lr_pull, lr_push=args.lr_push, device=args.device,
            repo_root=_HERE.parent, corpus_source=args.corpus_source,
            wikitext_name=args.wikitext_name, vocab_cap=args.vocab_cap,
        )
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "variance_decomp.json").write_text(
            json.dumps(rep, indent=2, default=str))
        print(json.dumps(rep, indent=2, default=str))
        print(f"\nwrote {out / 'variance_decomp.json'}")
        return 0

    # Re-label an existing run's verdict without re-running the arms.
    if args.reclassify:
        in_path = Path(args.reclassify)
        summary = json.loads(in_path.read_text())
        summary = reclassify_summary(summary, args.meaningful_effect_floor)
        out_dir = Path(args.output_dir) if args.output_dir != "reports/gate0" \
            else in_path.parent
        json_path, md_path = write_gate0_outputs(summary, out_dir)
        print(f"reclassified: {summary.get('verdict_reclassified_from')} "
              f"-> {summary['verdict']} "
              f"(meaningful_effect_floor={args.meaningful_effect_floor})")
        print(f"wrote {json_path}\nwrote {md_path}")
        return 0

    repo_root = _HERE.parent
    summary = run_gate0(
        seeds=_parse_seeds(args.seeds),
        D=args.D,
        landscape_size=args.landscape_size,
        window_size=args.window_size,
        n_test_windows=args.n_test_windows,
        n_train_windows=args.n_train_windows,
        vocab_size=args.vocab_size,
        k=args.k,
        beta=args.beta,
        theta_prime_mode=args.theta_prime_mode,
        n_consolidation_events=args.n_consolidation_events,
        alpha_anti=args.alpha_anti,
        repulsion_step_size=args.repulsion_step_size,
        lr_pull=args.lr_pull,
        lr_push=args.lr_push,
        device=args.device,
        output_dir=Path(args.output_dir),
        repo_root=repo_root,
        corpus_source=args.corpus_source,
        wikitext_name=args.wikitext_name,
        vocab_cap=args.vocab_cap,
        meaningful_effect_floor=args.meaningful_effect_floor,
    )
    json_path, md_path = write_gate0_outputs(summary, Path(args.output_dir))
    print(format_gate0_markdown(summary))
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
