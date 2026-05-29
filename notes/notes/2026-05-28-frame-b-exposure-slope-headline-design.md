---
name: frame-b-exposure-slope-headline-design
date: 2026-05-28
project: personal-ai
phase: Phase 3 — Growing Codebook (Frame B)
status: design / PROPOSED — pending user sign-off on the thresholds before it becomes binding
parent: 2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md
supersedes: the Part 3 "exposure–recall slope (and asymptote)" sketch in the anchor note
tags:
  - notes
  - subject/cognitive-architecture
  - subject/phase-3
  - subject/continual-learning
---

# Frame B headline — within-seed exposure–recall **slope-DiD**

**This is the proposed Phase 3 (Frame B) graduation headline.** It resolves the
anchor note's open question #1 (Part 8): the primary number is the
**slope-DiD**, not time-to-encode-novelty. All thresholds below are
**PROPOSED pending user sign-off** (see §Sign-off). Designed via a 4-way
design panel (MVP / variance-optimal / robustness / faithfulness) + synthesis;
winner = the slope-DiD spine (29/30).

## Why a slope (the binding constraint)

The Gate 0 endpoint difference-in-differences failed as **underpowered, not
falsified**: per-seed `σ(DiD) = 0.196`, ~5× the binomial floor, so the
variance is **structural** (the random codebook draw × corpus geometry),
needing ~590 seeds for power at effect 0.02. The per-seed codebook "luck" is
largely an **additive intercept** on the exposure–recall curve; a **within-seed
slope differences it out** (just as the DiD differenced out the landscape).
The slope is therefore a strictly-higher-SNR estimand for the *same* claim,
and — unlike the endpoint — its `weak` outcomes are fixable by affordable
n-escalation. (See the variance investigation + the CUPED/Monte-Carlo research
brief, 2026-05-28.)

## TL;DR

> **Headline.** Phase 3 (Frame B) graduates on the **within-seed slope-DiD**
> `Δβ(s) = β_real(s) − β_shuffle(s)`, where `β_w(s)` is the OLS slope of
> stratum-pooled Recall@K on **log cumulative exposure**, measured at 7
> precommitted log-spaced consolidation-event checkpoints in **one run per
> (seed, world)**. The inner slope removes the per-seed codebook intercept;
> the outer real−shuffle difference is the corpus-specificity test.

## Headline metric — the estimand

Per **atom seed** `s` (one substrate/codebook draw + one corpus-window
subsample = the independent unit), run the **existing Path C consolidation
stack** (`use_pull_push=True`, `use_context_residual=False`, C.2.1–C.2.5 at
Path C values) **once in the real world** and **once in the global-stream-
shuffle world** (`world="shuffled"`, seeded `s+80000` — same atom set, only
corpus *order* changes, so the two worlds are **paired on the codebook**).

Inside the existing consolidation loop
([c3_phase3_exit_criterion.py:617](../../experiments/c3_phase3_exit_criterion.py)),
at `M=7` cumulative-exposure checkpoints `e_m`, take a **read-only**
`codebook.clone()` snapshot and call the existing `_evaluate_recall_at_k`
([c3:248](../../experiments/c3_phase3_exit_criterion.py)) +
`_aggregate_by_stratum` ([c3:353](../../experiments/c3_phase3_exit_criterion.py))
against the **frozen landscape** memory on a **fixed held-out test set**,
yielding `r_w(s, e_m) = _overall_recall(per_stratum)` (identical pooling to
Gate 0; strata stay drill-down only).

- **Per-world slope** (closed-form OLS, invariant to any additive per-seed
  constant): `β_w(s) = Σ_m (x_m−x̄)(r_w(s,e_m)−r̄_w) / Σ_m (x_m−x̄)²`,
  with `x_m = log(e_m)`.
- **Headline per-seed contrast:** `Δβ(s) = β_real(s) − β_shuffle(s)`.
  - *Inner* difference (the slope) removes the per-seed codebook intercept
    within each world.
  - *Outer* difference (real − shuffle, paired on `s`) removes any
    exposure-correlated artifact common to both worlds (generic
    basin-tightening, repulsion). **Real-vs-shuffle is the load-bearing
    content of the headline, not a removable add-on.**
- **Aggregate** the `n` per-seed `Δβ(s)` via the existing
  `c3._delta_ci_stats` → mean, Student-t 95% CI (`ci95_above_zero`), per-seed
  positive fraction (`per_seed_robust_ge_threshold` at 0.70). **Atom seed is
  the unit — never per-trial pooling** (the pseudo-replication bug fixed
  2026-05-28).

`e=0` (the pre-consolidation codebook) is evaluated once before the loop and
**is** the frozen Phase-2 baseline = the curve intercept — this folds Gate 0's
frozen arms C/D into a free checkpoint (no separate frozen arms).

### Checkpoint schedule (PROPOSED)

`M=7` **log-spaced** checkpoints on the cumulative consolidation-**event** axis
(`n_consolidation_events=1000`): **e ∈ {16, 32, 64, 125, 250, 500, 1000}**,
plus the free `e=0` intercept anchor (not a regression point — `log(0)`
undefined). Log spacing because consolidation lift **saturates** (co-occurrence
learned early, then plateaus): `x=log(e)` linearizes the expected concave
curve, makes the slope a dimensionless "lift per e-fold of exposure," and
concentrates resolution early where real and shuffle most diverge. Lower anchor
`e=16` sits just after the first `consolidation_k`-buffer flush, so the first
fitted point already reflects ≥1 real update. **All `M` come from ONE run per
(seed, world)** via in-loop snapshots — `M` extra held-out evals, not `M` extra
runs. Schedule **fixed in the precommit before any run** (un-gameable).

## Control

The **global token-stream shuffle** (`world="shuffled"`, seed `s+80000`;
[c3:834-844](../../experiments/c3_phase3_exit_criterion.py)) — preserves
unigram marginals, destroys co-occurrence. **NOT** the retired gauge control
(codebook row-permutation, provably `E[Δ]=0`). The within-window shuffle is
held as a stricter **post-graduation** drill-down. The **gauge arm E** is
retained *solely* as the `FB→confound` STOP check (4a byte-identity + gauge
slope-diff ≈ 0).

## Variance handling (the whole point)

Three nested variance kills: (1) the **within-seed slope** removes the additive
per-seed codebook intercept (`σ²_intercept ≈ 0.196² − binom²`); (2) the
**real−shuffle** outer difference removes corpus-agnostic exposure tightening;
(3) **CUPED** (regress `Δβ` on the `e=0` frozen-baseline `C(s)`; reduction
`= ρ²`) is **held as a zero-cost re-analysis on the WEAK branch**, NOT in the
headline. The run **must report realized `σ(Δβ)` + implied honest-n** (via the
existing `variance_report_from_summary` / `n_for_80pct_power`); the variance
claim is thereby **itself falsifiable** (see the realized-σ gate).

## Graduation criterion (PROPOSED) — exactly TWO clauses + floor + σ-gate

At **n ≥ 10** atom seeds (seeds 0..9), BOTH must hold:

1. **CI-disjoint above control on the slope, with a meaningful floor:**
   `_delta_ci_stats(Δβ).ci95_above_zero == True` (Student-t 95% lower bound
   strictly > 0) **AND** mean `Δβ ≥ 0.01` recall-units **per e-fold** of
   exposure (≈ +0.06–0.07 Recall@K accumulated across the ~7 e-folds from
   e=16→1000 — the 0.02 Gate-0 endpoint floor re-expressed per-e-fold, so a
   CI-above-0-but-trivial slope does not graduate).
2. **Per-seed paired robustness ≥ 70%:**
   `per_seed_robust_ge_threshold(0.70) == True` (≥ 7/10 seeds with
   `Δβ(s) > 0`) — the clause that kills the Report 112 / Γ1 tail-draw mode.

**PLUS a realized-σ falsifiability gate:** if the clauses "pass" on a lucky
draw but realized `σ(Δβ)` has NOT beaten the intercept-dominated baseline
(honest-n still large), the verdict is **`FB→weak`, not pass** (proposed
`σ(Δβ) ≤ ~0.05` / honest-n ≤ ~30). We **decline a third monotonicity gate**
(it breaks the project's both-clauses precedent); monotonicity is a
branch-router/drill-down instead.

## Pre-committed branches (all decided before the run)

Verdict is a pure function of stored per-seed stats (reuses
`gate0._classify_verdict` / `reclassify_summary`) — no post-hoc tuning.

| verdict | signature | next |
|---|---|---|
| **FB→pass** | clause1 (CI>0 ∧ Δβ≥0.01/e-fold) ∧ clause2 (≥70%) ∧ realized-σ beat baseline | **Phase 3 Frame B GRADUATES** — consolidation gains accrue with corpus-specific exposure. Build the continual/Sense-B surface on the Path C stack; Γ1/Γ2/Γ3 stay CLOSED. (Phase 5′ reopen is a *separate* gate, not authorized here.) |
| **FB→weak** | mean Δβ>0 but CI overlaps, OR <70%, OR clauses pass but σ not beaten | real-but-underpowered, NOT a redesign trigger → (a) invoke held CUPED re-analysis (0 extra runs); (b) escalate n=10→20→30 (affordable *because* slope σ is small). |
| **FB→null-slope** | Δβ CI ≈/below floor BUT β_real ∧ β_shuffle both clearly >0 | consolidation improves with exposure but NOT corpus-specifically (shuffle learns as fast → tracking unigram marginals) → redesign targeting co-occurrence sensitivity, in the Frame B frame. |
| **FB→dead-slope** | β_real ≈ β_shuffle ≈ 0 | exposure moves nothing in either world; the prior endpoint signal was a static-snapshot artifact → deeper redesign (landscape/substrate/retrieval richness). |
| **FB→confound** | gauge arm E slope-diff CI excludes 0, OR 4a not byte-identical | exchangeability proof / harness wrong → STOP and re-derive. |
| **FB→non-linear** | curvature check fails (|quad/lin|>1) OR <70% seeds monotone | OLS log-slope is a poor summary → switch the per-seed estimand to **AUC-above-intercept** (trapezoid of `r(e)−r(e=0)` on the log axis; still intercept-free, still real−shuffle) and re-read against the **same** criterion (thresholds unchanged — cannot fish for a pass). Re-analysis, no rerun. |

## Anti-homunculus check

- **Which atoms move / when consolidation fires:** unchanged — a **local
  buffer-fill** event (`observe()` → `consolidate_if_ready()` at
  `consolidation_k`). No clock, no scheduler, no "now consolidate" controller.
- **When to measure:** the checkpoints are **pure read-only measurement** at
  fixed, data-independent event boundaries (decided before the run, never read
  from any metric). The eval (a) does not write back into codebook/buffer/C.2
  state; (b) does not gate/pause/branch/modulate the consolidation loop;
  (c) has **no `if recall<θ then act` branch**. The exposure axis counts events
  that *already happened* — it observes the local dynamics, it does not schedule
  them. The slope is a **measurement of a local-geometry trajectory**.
- **No-clock-controller guard (binding):** the implementation MUST include a
  regression test asserting a checkpointed run produces a **bit-identical**
  final codebook + final recall vs a non-checkpointed run. Byte-identity is the
  proof that checkpointing is a thermometer, not a thermostat.
- **Where the "decision" lives:** nowhere new at runtime — the slope, contrast,
  CI, robustness, σ-gate, and branch routing are all **offline statistics** that
  decide whether the *phase* graduates (a human research decision), never what
  the substrate does. The real-vs-shuffle control is a data manipulation, not an
  arbiter. **No new substrate mechanism → no new anti-homunculus reviewer pass
  for this metric** (per Gate 0 precommit / anchor Part 6). The future **Sense-B
  novelty-allocation mechanism** (drill-down only here) still needs its own pass
  (allocation = local residual-energy event; any threshold substrate-derived /
  1/β-equivalent, never an external clock).

## Implementation sketch

Builds entirely on the existing Gate-0/C.3 machinery; ~one in-loop hook + one
orchestrator + tests; **no new substrate mechanism**.

1. **Checkpoint-eval hook** in `_consolidate_codebook` (loop at
   [c3:617](../../experiments/c3_phase3_exit_criterion.py)): optional
   `checkpoint_events: Sequence[int] = ()`; after `consolidate_if_ready()`,
   when `consolidation_events` crosses a boundary, eval `_overall_recall` on
   `codebook.clone()` vs the frozen landscape + fixed test windows; append
   `(e, recall)`. **Read-only** — must not touch updater RNG or C.2 state. GPU:
   one `.clone()`/sync per checkpoint = 7 total, outside the hot per-event path.
2. Thread `checkpoint_events` through `_run_single_seed_condition`; reuse
   `world="real"/"shuffled"`; **drop** frozen arms C/D (e=0 is the baseline);
   **retain** gauge arm E verbatim.
3. New orchestrator `experiments/frame_b_exposure_slope.py` (clone
   `gate0_frame_a.py`): per seed run real + shuffled with the checkpoint set;
   `_ols_slope(log_xs, ys)` (~8 lines); `Δβ(s)`; aggregate via
   `c3._delta_ci_stats`; apply the 0.01/e-fold floor; emit realized σ + honest-n
   via `variance_report_from_summary` / `n_for_80pct_power`; classify via a
   `_classify_frame_b_verdict` mirroring `_classify_verdict`. Hold CUPED + AUC as
   weak/non-linear-branch re-analyses.
4. **Tests** (mirror `test_gate0_frame_a.py`): (a) **no-clock byte-identity**
   (checkpointed final codebook bit-identical to non-checkpointed — the binding
   guard); (b) `_ols_slope` on a known line; (c) intercept-invariance (adding a
   constant to every checkpoint leaves the slope unchanged); (d) identity-gauge
   4a + gauge slope-diff ≈ 0; (e) synthetic-world plumbing → Δβ ≈ 0.
5. Colab notebook (clone `colab_gate0_frame_a.ipynb`): 2 worlds × 10 seeds +
   gauge ≈ **30 procs**, `--device cuda`, parent must not touch CUDA, pre-warm
   wikitext cache.
6. **Pin into `phase-3-deep-dive.md` §Headline + anchor Part 3 ONLY after
   sign-off.**

## Drill-downs (explain the headline; never gates)

(1) per-world slopes `β_real`, `β_shuffle` (the null-slope vs dead-slope
discriminator); (2) the raw `r_real(e)`, `r_shuffle(e)` curves (shape →
non-linear router); (3) **endpoint level-DiD** (the prior Gate-0 metric — kept
for continuity + as the variance baseline beaten; positive level-DiD + flat
slope ⇒ "landscape carries it, not consolidation"); (4) monotonicity/
consistency read (router → AUC); (5) AUC-above-intercept re-read (sign-disagree
with slope ⇒ weak); (6) realized `σ(Δβ)` + honest-n vs 0.196; (7) CUPED `ρ²`
on the e=0 covariate (weak branch); (8) regime-stratum breakdown (labels only);
(9) **Sense-B novelty probe** (held-out token-ids introduced mid-stream;
exposures-until-retrievable in real vs shuffled — a *measurement*; the
allocation mechanism needs its own reviewer pass before any code lands).

## Cost

`n=10 × 2 worlds = 20` consolidation procs + 10 gauge (arm E, endpoint-only)
≈ **30 procs** — checkpoint overhead ~+10–20% wall-clock per run (7 read-only
evals ≪ the 1000-event loop), so **cheaper than Gate 0's 5-arm/50-proc design**.
≈ 1 hr on a single A100/L4 parallelized. CUPED/AUC/σ-report = 0 extra compute.
`FB→weak` n=30 escalation ≈ 90 procs / ~2–3 hr — affordable *because* the
slope's σ is small (the whole point). Single operating point only (the Γ1/Path C
wikitext point, so arm A reproduces prior work).

## What this does NOT permit

- Does NOT graduate any phase on its own (needs an actual n≥10 PASS run).
- Does NOT unpause Phase 5′ / authorize any ΔE / bridge / M2 / matrix / Phase-5
  work (FB→pass graduates **Phase 3 only**; Phase 5 reopen is a separate gate).
- Does NOT authorize mechanism redesign (Γ1–Γ5) — FB→pass **closes** the Γ
  track; redesign re-enters only via FB→null-slope / FB→dead-slope, after the
  run, never preemptively; never on FB→weak without n-escalation first.
- Does NOT authorize an operating-point sweep in the graduation run.
- Does NOT add CUPED / AUC / monotonicity / a denser-or-linear schedule **to the
  headline** (they are precommitted weak/non-linear-branch re-analyses).
- Does NOT build the Sense-B allocation mechanism (measurement-only here).
- Does NOT replace the stream-shuffle control with the retired gauge control.
- Does NOT turn checkpoints into any behavior-gating clock (byte-identity
  invariant binding).
- Does NOT change the floor or schedule post hoc to manufacture a verdict.

## Sign-off (these PROPOSED numbers are the user's to approve before binding)

1. **Checkpoint schedule:** M=7 log `{16,32,64,125,250,500,1000}` + free e=0.
2. **n:** 10 atom seeds (weak → 20 → 30).
3. **Per-seed robustness:** 0.70 (≥7/10 with Δβ>0).
4. **Slope floor:** mean Δβ ≥ 0.01 recall-units / e-fold (≈ +0.06–0.07 R@K
   over e=16→1000; span-relative — re-derive if the span/event count changes).
5. **Realized-σ gate:** clauses-pass-but-σ-not-beaten → `FB→weak` (proposed),
   with σ(Δβ) ≤ ~0.05 / honest-n ≤ ~30 as "beaten."
6. **CUPED / AUC / monotonicity:** held as precommitted weak/non-linear
   re-analyses + drill-downs, NOT in the headline.
7. **Exactly TWO graduation clauses** (no third monotonicity gate).
8. **Control:** global stream-shuffle primary; within-window held as a stricter
   post-graduation drill-down; gauge arm E = confound check only.
9. **Secondary:** endpoint level-DiD reported as a continuity drill-down.
10. **Pin + primary number:** slope-DiD (not time-to-encode-novelty) is THE
    primary number → resolves anchor Part 8 #1; written into
    `phase-3-deep-dive.md` §Headline + anchor Part 3 on sign-off.
