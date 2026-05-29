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
) -> str:
    """Map the stats onto the precommit's pre-committed branch table.

    Order matters: confound (E fails) dominates; then the DiD ladder.
    Thresholds are deliberately simple and the raw stats are reported
    alongside so a human can confirm the routing.
    """
    if not gauge["passes_4a_and_4b"]:
        return "G0->confound"
    did_mean = did["mean_delta"]
    did_pass = did["ci95_above_zero"] and did["per_seed_robust_ge_threshold"]
    if did_pass:
        return "G0->pass"
    # "≈ 0" band for the DiD mean: within one per-seed SEM of zero.
    did_near_zero = (did["sem_delta"] == 0.0) or (
        abs(did_mean) <= did["sem_delta"]
    )
    if did_mean > 0 and not did_pass and not did_near_zero:
        return "G0->weak"
    # DiD ≈ 0 → consolidation carries no corpus-specific structure.
    if a_minus_b["ci95_above_zero"]:
        return "G0->null-cons"  # landscape carries structure, cons does not
    if (a_minus_b["sem_delta"] == 0.0) or (
        abs(a_minus_b["mean_delta"]) <= a_minus_b["sem_delta"]
    ):
        return "G0->dead"  # whole pipeline captures no corpus structure
    return "G0->weak"  # mean>0 but underpowered, or A-B positive-but-not-disjoint


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
) -> dict:
    """Run all five Gate 0 arms and assemble the DiD summary."""
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(seeds)

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
            theta_prime_mode=theta_prime_mode,
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

    verdict = _classify_verdict(did, a_minus_b, gauge, len(seeds))

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
                "theta_prime_mode": theta_prime_mode,
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
        f"θ′={op['theta_prime_mode']}"
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
    p.add_argument("--output-dir", default="reports/gate0")
    args = p.parse_args(argv)

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
    )
    json_path, md_path = write_gate0_outputs(summary, Path(args.output_dir))
    print(format_gate0_markdown(summary))
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
