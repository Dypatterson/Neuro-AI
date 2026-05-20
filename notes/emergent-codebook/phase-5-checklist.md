# Phase 5 exit checklist

Companion to [phase-5-unified-design.md](phase-5-unified-design.md). This
file lists the **graduation criteria** for Phase 5 — what must be true
before we say Phase 5 worked. It is not an implementation to-do list;
implementation milestones (decision #2 γ sweep, decision #3 K_main sweep,
decision #4 bundle re-settle convergence, test unskipping) belong in the
design doc or a separate working note, not here.

**Phase 5 graduates when every "must-have" item is verified. Each
"verified" status requires a multi-seed report with CI evidence at
n_seeds ≥ 10**, matching the bar set by [report 038](../../reports/038_phase4_d1_graduation.md).

Single-seed results are not "verified." Mechanism-validated runs are not
"verified." Effects that depend on one schema source, one seed, or one
readout metric are not "verified."

---

## Status legend

- ✅ **verified** — multi-seed (n ≥ 10) + CI evidence in a report
- 🟨 **partial** — implemented and tested, but not multi-seed or with caveats
- ❌ **open** — required but no evidence yet
- ⚠️  **failed / reframed** — mechanism exists but doesn't pass as designed, *or* item has been reframed per a binding discipline note
- ⏳ **in flight** — run is actively executing or aggregating; evidence pending
- 🟦 **deferred** — explicitly noted in design as deferred (not blocking)

---

## A. Headline metric

The single graduation criterion. Per [phase-5-unified-design.md:258-277](phase-5-unified-design.md)
and the [2026-05-16 substrate-vs-readout discipline note](../notes/2026-05-16-substrate-vs-readout-metric-discipline.md):
energy, not readout.

| # | Headline | Status | Evidence |
| -- | --- | :---: | --- |
| A1 | **Δ final-state energy E_A − E_B (content-prior minus role-prior) > 0 with 95% CI disjoint from zero**, on a held-out cue set designed for structural retrieval, n_seeds ≥ 10 | 🟨 | partial, n=5: ΔE = −1.5e-4, CI [−5.2e-4, +2.1e-4] (includes 0); 3/5 seeds positive; LOSO never excludes zero ([report 041](../../reports/041_phase5_de_n5_partial.md)) |

Positive ΔE means role-prior branches settle into lower-energy joint
states than content-prior branches — the test of structural retrieval.

---

## B. Required controls (per design §"Required controls")

Each control runs on the same cue set and seeds as the headline. Each
must show the headline effect **disappear or shrink** when the control
removes the mechanism under test.

| # | Control | What it falsifies if it fails | Status | Evidence |
| -- | --- | --- | :---: | --- |
| B1 | **Random-schema branches** | If random schemas as priors produce the same ΔE as role-binding schemas, no structural retrieval is happening | 🟨 | random_K4 logged in exp 40 output, not paired against role_K4 in [report 041](../../reports/041_phase5_de_n5_partial.md) |
| B2 | **K=1 (single-branch, no branching)** | If single-branch with role-prior matches K-branch with role-prior, branching is gratuitous | ⚠️ | **fires at n=5**: K1 ΔE = +1.3e-4 (5/5 seeds positive, pooled bootstrap CI [+3.5e-5, +2.8e-4] excludes 0) ≥ K4 ΔE = −1.5e-4. Branching is gratuitous (in fact destructive) on 6–12-atom post-death substrates. [Report 041](../../reports/041_phase5_de_n5_partial.md) |
| B3 | **No-prior (γ=0)** | If ΔE survives with no prior weight, the prior is not doing the work | ✅ | ΔE = 0 exactly across 5 seeds × 20 cues; γ=0 makes content/role priors identical by construction. [Report 041](../../reports/041_phase5_de_n5_partial.md) |
| B4 | **No-schema-store** (priors drawn directly from codebook, not filtered slow store) | If ΔE survives without the schema store, the store is not the right source | ❌ | not yet run |

---

## C. Schema-source robustness (Phase 3/4 carryover safeguard)

**Motivation.** Phase 5 inherits the post-death Phase 4 substrate as its
schema source ([report 040](../../reports/040_freq_weighted_alpha_sweep.md)).
But [report 037](../../reports/037_seed3_collapse_diagnostic.md) showed
that pattern death collapses the W=2 population down to 5–30 survivors per
seed, and those survivors reflect **corpus-order retrieval reinforcement
breadth**, not necessarily stable structural usefulness. If role-prior
branching only works on one hand-selected schema source, that is not a
clean Phase 5 graduation — it is an artifact of the surviving atoms.

**Anti-homunculus note.** This entire section is **diagnostic-only**. The
system MUST NOT read a robustness result and adaptively switch schema
sources. The legitimate use of these comparisons is to establish whether
the Phase 5 effect survives schema-source variation; the legitimate
response to a fragile result is "Phase 5 does not graduate," not "switch
to the schema source that worked."

Five schema-source conditions, all matched to the post-death atom count
to control for "smaller schema set is easier to branch over":

| # | Schema source | What it tests | Status | Evidence |
| -- | --- | --- | :---: | --- |
| C1 | **Post-death** (the design default; surviving atoms ranked by `effective_strength`) | The Phase 5 design as specified | 🟨 | partial, n=5; A1 does not yet pass at this source ([report 041](../../reports/041_phase5_de_n5_partial.md)) |
| C2 | **Pre-death top-k** (top-k atoms by `effective_strength` from the pre-death population, k = post-death count) | Is the filter *direction* (high-strength atoms) load-bearing, independent of death? | ❌ | not yet run |
| C3 | **Pre-death random-k** (random subset of pre-death atoms, k = post-death count) | Is *having fewer schemas* load-bearing, independent of filter direction? | ❌ | not yet run |
| C4 | **Step-1500 top-k** (top-k by `effective_strength` at step 1500, before late-run death) | Does the effect survive at an earlier substrate snapshot? | ❌ | not yet run |
| C5 | **Step-1500 random-k** (random-k at step 1500) | Earlier substrate + neutral filter; control for C4 | ❌ | not yet run |

**Graduation rule for section C.** The headline ΔE must be CI-disjoint
from zero in **at least the post-death condition (C1) AND at least one
of {C2, C4}** — i.e., the effect must survive at least one schema source
that does not depend on late-run death survival. If only C1 produces the
effect, Phase 5 does not graduate cleanly; report it as substrate-fragile
and re-scope.

🟦 **Optional**: death-disabled schema source — only if cheap to produce
(requires a separate Phase 4 training run with the death mechanism off).
Not gating; reported if available.

---

## D. Seed robustness (Phase 4 carryover safeguard)

**Motivation.** Three independent runs (reports 026, 028, 029) identified
seed 23 as the cap_t05 / R@10 outlier — idiosyncratic geometry, not noise
(STATUS blocker #5). Phase 5's headline CI must not be silently carried
or destroyed by one seed.

| # | Item | Status | Evidence |
| -- | --- | :---: | --- |
| D1 | **n_seeds ≥ 10** on the headline run | 🟨 | at n=5 ([report 041](../../reports/041_phase5_de_n5_partial.md)); 5 more W=4 post-death snapshots needed |
| D2 | **Leave-one-seed-out (LOSO) CI sensitivity** reported on the headline | 🟨 | LOSO reported at n=5; for K4 no leave-one-out subset produces CI excluding zero ([report 041](../../reports/041_phase5_de_n5_partial.md)) |
| D3 | **Seed-23-specific readout** in the headline report — ΔE for seed 23 alone, side-by-side with the n=10 mean | 🟨 | seed 23 K4 ΔE = +1.55e-7 (weak positive, not outlier at n=5) ([report 041](../../reports/041_phase5_de_n5_partial.md)) |

**Graduation rule for section D.** The headline LOSO sweep must show the
n=9 CI excludes zero for **every** leave-one-out subset. If removing any
single seed makes the CI include zero, the headline is one-seed-dependent
and Phase 5 does not graduate cleanly.

---

## E. Substrate drill-downs (per design §"Drill-downs")

Per the discipline note: read these to **explain** the headline, not to
replace it. None of these are graduation-gating individually.

| # | Drill-down | Aggregated multi-seed? | Status | Evidence |
| -- | --- | :---: | :---: | --- |
| E1 | Branch-energy dispersion (entropy of softmax weights `w_k`) | 🟨 n=5 | ⚠️ | **uniform: exactly ln(K) across 5 seeds × K∈{2..8}** — branches equi-energetic to FP precision ([report 042](../../reports/042_phase5_branching_collapse_diagnostic.md)) |
| E2 | Split-eligibility rate (joint criterion fires) | ❌ | ❌ | predicted: rises with corpus complexity |
| E3 | Bundle convergence rate (fraction of re-settlings that converge below energy threshold) | 🟨 n=5 | 🟨 | uniform `energy_drop` ≈ 0.04–0.07 across all branches/seeds suggests yes (not directly measured) ([report 042](../../reports/042_phase5_branching_collapse_diagnostic.md)) |
| E4 | Schema-store utilization (fraction of schemas ever picked as top-K) | ❌ | ❌ | <10% utilization → store needs pruning |
| E5 | Branch-state diversity (mean pairwise FHRR distance among `{q_k*}`) | 🟨 n=5 | ⚠️ | post-death collapses (6.7e-6 to 2.7e-3 across 5 seeds × K=4) ([report 042](../../reports/042_phase5_branching_collapse_diagnostic.md)); **pre-death restores 3×–4978× across all 5 seeds × both prior types** ([report 043](../../reports/043_phase5_substrate_scale_diagnostic.md)) — death implicated for branch collapse; K-branch mechanism not structurally degenerate |
| E6 | Prior-domination rate at δ ∈ {0.5, 0.75, 0.9} vs γ | 🟨 sweep n=1 | 🟨 | γ sweep on seed 17: ΔE_K1 ≥ ΔE_K4 at every γ; γ≥1.0 flips both negative (prior overpowering substrate) ([report 042](../../reports/042_phase5_branching_collapse_diagnostic.md)) |

---

## F. Readout audit (Phase 4 carryover safeguard)

**Motivation.** The Phase 5 headline is energy because energy is
substrate-pure. But "energy improved while readout quietly degraded" is a
real failure mode (STATUS blocker #6′: top1 regression is a Phase 3
property and persists under Phase 4). Every successful ΔE result must
report readouts alongside, so the tradeoff is visible.

Reported in every Phase 5 headline report, **not** used to gate
graduation (per the 2026-05-16 discipline note: substrate-pure metrics
gate; readouts are drill-downs):

| # | Readout | Reported? | Status |
| -- | --- | :---: | :---: |
| F1 | Δtop1 (role-prior vs content-prior) | ❌ | ❌ |
| F2 | ΔR@10 | ❌ | ❌ |
| F3 | Δcap-coverage at τ=0.5 | ❌ | ❌ |
| F4 | Convergence behavior (per E3) | ❌ | ❌ |
| F5 | Meta-stability at W=3 — see section G | ❌ | ❌ |

---

## G. Substrate-pure tiebreaker

**Motivation.** If ΔE is favorable but readout regresses (Δtop1 / ΔR@K
CI-disjoint adverse), is that an acceptable structural-basin vs
token-identity tradeoff, or a failure? Per the 2026-05-16 discipline
note, substrate-pure metrics win the tie. Concretely:

| # | Tiebreaker | Status | Evidence |
| -- | --- | :---: | --- |
| G1 | **Δ meta-stable rate at W=3 (D1) under role-prior branching must NOT regress** vs content-prior branching | ❌ | same D1 metric that graduated Phase 4 in [report 038](../../reports/038_phase4_d1_graduation.md) |

**Graduation rule for section G.** If A1 passes (ΔE > 0 CI-disjoint) but
G1 shows W=3 meta-stable rate regresses under role-prior branching, the
energy improvement is being purchased by destabilizing the substrate.
That is not a clean graduation — report as a structural-basin tradeoff
to characterize, not as Phase 5 verified.

---

## H. Anti-homunculus discipline (explicit non-actions)

Per [CLAUDE.md §anti-homunculus filter](../../CLAUDE.md) and the
[2026-05-09 paper synthesis](../notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md):
every mechanism in Phase 5 must be local geometric dynamics, or a
measurement of one. No supervisor decides which subsystem wins. No
`if X then do Y` reads a metric and reroutes.

Explicit prohibitions for Phase 5:

| # | Prohibition | Why |
| -- | --- | --- |
| H1 | Schema-source ablation (section C) is **diagnostic-only** — the system MUST NOT switch schema sources based on robustness results | Adaptive rerouting reads a metric and selects a subsystem; textbook homunculus |
| H2 | Atom-splitting **diagnostic** is logged; the actual split *action* is a deferred sub-component | The current scope is measurement; the action mechanism needs its own anti-homunculus check |
| H3 | Branch selection is **energy-based only**, never metric-based (no "pick the branch with the best cap-coverage") | Per-branch diagnostics are logged, not used for selection; design §"Per-branch diagnostics" |
| H4 | No "if ΔE fails on C1 but passes on C2, declare graduation" — graduation is structural, not best-of-N over conditions | A robustness sweep that becomes a search for the condition that works is a homunculus by another name |

---

## I. Interpretation rule (the central question Phase 5 must answer)

> Does structural role-prior branching genuinely beat content-prior
> branching inside the settled energy landscape, or are we just seeing
> an artifact of the surviving Phase 4 atoms?

Phase 5 graduates cleanly **only if all of the following hold**:

1. **A1** passes — ΔE CI-disjoint from zero, n ≥ 10.
2. **All of B1–B4** behave as predicted (controls remove or shrink the effect).
3. **C1 AND at least one of {C2, C4}** show the effect — the effect survives at least one schema source that does not depend on late-run death survival.
4. **D1 + D2** pass — n ≥ 10 with LOSO CI excludes zero for every leave-one-out subset.
5. **G1** passes — W=3 meta-stable rate does not regress under role-prior branching.

If the effect depends on a fragile schema source (C fails), is carried
by one seed (D2 fails), or is purchased by substrate instability (G1
fails), **Phase 5 does not graduate cleanly** — the report documents
which failure mode and Phase 5 is re-scoped.

---

## J. Carried Phase-3/4 items (parked behind Phase 5; not gating)

These remain open from Phase 4 and earlier phases but do not gate Phase 5
graduation. Listed so they are not lost:

- STATUS blocker #4 — diagnostic-actuator dynamic-form session: **held 2026-05-20** ([note](../notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md)); A+B combined candidate passed anti-homunculus audit. Implementation commitment pending.
- STATUS blocker #5 — seed-23 idiosyncratic geometry diagnostic (3 runs
  identified it, never investigated). Phase 5 surfaces seed 23 via D3
  but does not diagnose it.
- STATUS blocker #6′ — top1 regression is a Phase 3 / Hebbian property;
  Phase 5 reports Δtop1 (F1) but does not fix it
- Pre-phase commitments: consolidation-geometry regime classifier;
  empirical θ′(β) calibration spike

---

## Status banner (one line for STATUS.md)

> Phase 5: **A+B death-mechanism dynamic-form note complete and audited**
> ([2026-05-20 diagnostic-actuator note](../notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md);
> anti-homunculus reviewer PASS after 4 fixes). Audit prereqs cleared
> (commit [ec3b95b](https://github.com/Dypatterson/Neuro-AI/commit/ec3b95b):
> CFL clamp, reproducibility lockfile, HAM deferred-sync). **A+B implementation
> in flight 2026-05-20.** Continuous coverage-weighted reinforcement (A) +
> -α log(d_eff) repulsion in substrate energy (B) replace the binary
> death step. Pre-committed falsification criteria binding (d_eff ≥ 25 at
> step 1800; K-branch state_divergence within 30% of pre-death; Phase 4 D1
> non-regression at n=10; α/λ fixed pre-retrain, not tuned). Cue-regime /
> role-prior bimodality (report 043's second axis) remains open — death
> redesign is necessary but not sufficient for A1 graduation.
