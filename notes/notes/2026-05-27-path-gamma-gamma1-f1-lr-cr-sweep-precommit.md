---
date: 2026-05-27
project: personal-ai
phase: Path γ — Phase 3 mechanism redesign
mechanism: Γ1.c — context-residual consolidation (F1 lr_cr sweep)
status: precommit (diagnostic, not graduation; smoke-scale only)
parent: 2026-05-27-path-gamma-gamma1-context-residual-precommit.md
tags:
  - notes
  - subject/cognitive-architecture
  - subject/path-gamma
  - subject/phase-3
---

# Path γ — Γ1.c F1 lr_cr sweep precommit (diagnostic, smoke n=3)

## Status

**Diagnostic-only precommit.** Inherits the anti-homunculus reviewer PASS
on the [Γ1 parent precommit](2026-05-27-path-gamma-gamma1-context-residual-precommit.md)
verbatim — F1 changes a hyperparameter, not the mechanism. No new
reviewer pass is required. **F1 results do NOT graduate Phase 3 under
any outcome** (binding per §"What this precommit does not permit").

## Lineage

[Report 113](../../reports/113_path_gamma_gamma1_headline_gate.md) closed
the Γ1.c headline gate at lr_cr=0.1 as FAIL on both clauses of the
revised C.3 criterion. Two candidate explanations for the attenuation
(Report 113 §"Finding 3"):

1. **Effective learning rate mismatch.** The Γ1.c update direction is
   `ε = codebook[target] − codebook[predicted]` (atom-space). For
   atoms that are *similar* (which is what happens during confusions),
   ‖ε‖ may be « 1. Pull/push uses `slot_query` with magnitude ≈ 1 by
   construction. So `lr_cr=0.1` and `lr_pull=0.1` produce different
   *effective* update magnitudes.
2. **Atom-vs-atom geometry carries less corpus signal than atom-vs-cue
   geometry.** WikiText's compositional structure may live more in
   cue geometry than in atom-pair geometry; Γ1.c is operating on the
   wrong axis regardless of magnitude.

F1 is the precommitted follow-up named in
[Γ1 precommit §"Pre-committed follow-ups"](2026-05-27-path-gamma-gamma1-context-residual-precommit.md):
sweep `lr_cr ∈ {0.01, 0.05, 0.1, 0.2, 0.5}` at smoke scale n=3 to
distinguish the two hypotheses.

## Architectural claim (inherited)

Unchanged from the parent precommit §"Architectural claim". The
mechanism is asymmetric gradient descent on the per-event repulsion
energy

> **E_cr(codebook | event) = − Σ_{j ∈ roles} 1[predicted_id_j ≠ target_id_j] · ½ · ‖codebook[target_id_j] − codebook[predicted_id_j]‖²**

with respect to `codebook[target_id_j]` (stop-gradient on
`codebook[predicted_id_j]`). F1 changes only the magnitude of the
descent step (`lr_cr`), not the mechanism. The H1–H7 anti-homunculus
discipline is unaffected.

## What F1 measures

For each `lr_cr ∈ {0.01, 0.05, 0.1, 0.2, 0.5}`, run n=3 seeds (0..2) of
the C.3 driver at the Γ1 headline operating point. From each seed JSON
extract per-stratum standard-vs-control Δ (with the existing Wilson
machinery). Aggregate per lr_cr:

- **per-seed mean Δ** (stratum-pooled default mode, seeds 0..2).
- **per-seed Δ distribution** (3 values per lr_cr).
- **comparison vs PathC baseline at the same seeds 0..2** (from the
  headline gate's `pathc_baseline_seed{0..2}/c3_summary.json` files
  already on Drive — no need to rerun PathC).

**The diagnostic question:** does any lr_cr ∈ {0.01, 0.05, 0.1, 0.2, 0.5}
produce a per-seed mean Δ close to PathC's +0.055 at the matched n=3
slice?

- **Hypothesis (a) confirmed** if `max_lr_cr mean(Δ_Γ1.c) ≳ 0.04` (≈
  within 30% of PathC) at any lr_cr. The shape change can carry signal
  comparable to pull/push *when the effective magnitude is matched*.
  Imply: a *separate* precommit may authorize a full n=10 headline
  gate at the winning lr_cr.
- **Hypothesis (b) confirmed** if `max_lr_cr mean(Δ_Γ1.c) ≲ 0.02` (≈
  twice the lr_cr=0.1 result, no clear lr_cr-dependent recovery). The
  shape change carries fundamentally less signal than pull/push
  regardless of magnitude.
  Imply: Γ1 family is closed. The next-candidate precommit (Γ2 or Γ3
  per the survey ranking) becomes the next deliverable.
- **Inconclusive** if the lr_cr→Δ curve is non-monotonic or noisy
  enough at n=3 that neither hypothesis is clearly supported.
  Imply: F1 is extended to n=5 or n=10 at the most promising lr_cr.

## Operating point

Identical to the [Γ1 headline gate](2026-05-27-path-gamma-gamma1-context-residual-precommit.md)
operating point in every dimension *except* `lr_cr`. The point is to
isolate the lr_cr lever.

| Knob | Value | Source |
|---|---|---|
| corpus | wikitext-2-raw-v1, vocab_cap=1000, window=8 | Γ1 headline |
| D | 4096 | Γ1 headline |
| β | 10 | Γ1 headline |
| K (Recall@K) | 5 | Γ1 headline |
| landscape_size | 64 | Γ1 headline |
| n_consolidation_events | 1000 | Γ1 headline |
| α_anti, repulsion_step_size | 0.01, 0.05 | Γ1 headline |
| λ_ac, μ_T, τ_T, λ_cc, obs_rate, drift_ema_rate | Path C C.2.x values | Γ1 headline |
| `use_context_residual`, `use_pull_push` | True, False | Γ1 headline |
| **`lr_cr` (the sweep dimension)** | **{0.01, 0.05, 0.1, 0.2, 0.5}** | This precommit |
| `n_seeds` | 3 (smoke) — seeds 0, 1, 2 | This precommit |

PathC baseline at the same seeds 0..2 is *not* re-run; it's loaded from
the Γ1 headline gate's Drive output at
`MyDrive/neuro-ai/results/gamma1_headline_2026-05-27/pathc_baseline_seed{0..2}/`.

## Anti-homunculus discipline (inherited)

F1 changes only `lr_cr` (a static config value set at construction).
H1–H7 of the [Γ1 precommit](2026-05-27-path-gamma-gamma1-context-residual-precommit.md)
remain in force; no new mechanism is introduced. The reviewer-PASS on
the parent precommit carries forward verbatim.

**Specific H3 safeguard.** The five lr_cr values produce five
*per-condition* Δ values. **No graduation evidence is taken from any
of them.** The reading is purely diagnostic — "does the lr_cr dial
recover PathC-magnitude effects, and if so at what value?" If F1
identifies a promising lr_cr, the *next* move is a separate precommit
that authorizes a full n=10 headline gate at that lr_cr. Reading F1's
best-lr_cr result *as graduation* would be exactly the H3-forbidden
best-of-N pattern.

## What this precommit does NOT permit

- **No graduation claim under any F1 outcome.** F1 is diagnostic.
- **No silent lr_cr promotion** to "the new Γ1 default." If F1
  identifies a promising lr_cr, that's the *content* of the next
  precommit, not an implicit policy change.
- **No corpus / D / β / window / vocab_cap variation.** All other
  knobs held at Γ1 headline values.
- **No new conditions beyond the five lr_cr points.** Specifically,
  *no* α_anti=0, λ_ac=0, or "all C.2.x off" ablations under F1.
  Those are post-graduation drill-downs (parent precommit F4); F1 is
  the lr_cr question only.
- **No pull/push baseline re-run.** PathC at seeds 0..2 is loaded
  from Drive from the headline gate's run. (Conserves Colab credit
  and ensures byte-identity with Report 113's PathC numbers.)
- **No symmetric Γ1.c variant.** That's parent precommit F2 — its own
  precommit if/when authorized.
- **No Phase 5 work of any kind.**

## Cost

- 5 conditions × 3 seeds = **15 procs** in parallel.
- Each proc ≈ same wall time as one Γ1 headline proc (~10–25 min on
  A100 wikitext-2 at n_events=1000; the Report 113 run took on the
  order of an hour for 20 procs).
- Total: under 30 min wall clock on A100, materially cheaper than the
  Γ1 headline gate.

## Falsifiable diagnostic criterion

For each `lr_cr ∈ {0.01, 0.05, 0.1, 0.2, 0.5}`, F1 reports:

- per-seed mean Δ (stratum-pooled default mode, seeds 0..2)
- per-seed Δ range (min, max across 3 seeds)
- per-seed Δ vs PathC at matched seeds 0..2

Then evaluates against the two hypotheses defined in §"What F1
measures":

| Hypothesis | Empirical signature | Implication |
|---|---|---|
| (a) effective-lr mismatch | max(lr_cr=0.2 or 0.5) per-seed mean Δ ≳ 0.04 | New precommit for full n=10 gate at the winning lr_cr |
| (b) atom-vs-atom geometry carries less signal | max across all lr_cr per-seed mean Δ ≲ 0.02 | Γ1 family closes; Γ2 or Γ3 precommit is next |
| inconclusive | curve is non-monotonic or noisy beyond resolution | F1 extension at n=5 or n=10 at most promising lr_cr |

The thresholds 0.04 and 0.02 are derived from Report 113: PathC
per-seed mean Δ +0.055; Γ1.c at lr_cr=0.1 per-seed mean Δ +0.0115.
A 30% margin around PathC (≈ 0.04) is the floor for "Γ1.c shape can
match pull/push signal magnitude at the right lr_cr"; 2× the lr_cr=0.1
result (≈ 0.02) is a generous ceiling for "lr_cr doesn't matter much."

## Pre-committed escalation paths

Named here so the F1 outcome doesn't trigger ad hoc design under the
post-mortem:

- **F1→a (hypothesis a confirmed):** new precommit "Γ1 headline gate
  v2 at lr_cr=`<winner>`" authorizes a full n=10 paired-baseline run
  (same as the headline notebook structure, just with the new lr_cr).
  Anti-homunculus reviewer is *not* required again (no mechanism
  change). experiment-result-auditor is required before any STATUS
  update.
- **F1→b (hypothesis b confirmed):** new precommit for Γ2
  (bundle-first scene memory per the [survey](../emergent-codebook/path-gamma-mechanism-family-survey.md))
  — full mp-grill-with-docs + anti-homunculus reviewer + experiment-
  result-auditor cycle, same as the Γ1 precommit itself was developed.
- **F1→inconclusive:** new precommit "F1 extension at n=5" or
  "F1 extension at lr_cr=`<promising value>` n=10" — diagnostic, not
  graduation.

## Implementation surface

No code changes needed. The C.3 driver already takes `--lr-cr` and
`--use-context-residual` and `--no-pull-push` from the Γ1 implementation
landing (2026-05-27). F1 is purely a notebook-level orchestration of
the existing driver across the five lr_cr values.

Files this precommit touches:

- [notes/notes/2026-05-27-path-gamma-gamma1-f1-lr-cr-sweep-precommit.md](.) (this file).
- [scripts/colab_gamma1_f1_lr_cr_sweep.ipynb](../../scripts/colab_gamma1_f1_lr_cr_sweep.ipynb)
  (companion Colab notebook).

## Implementation findings (running log)

### 2026-05-27 — F1 result: hypothesis (b) confirmed; Γ1 family closes

- **Verdict:** ✅ Hypothesis (b) per [Report 114](../../reports/114_path_gamma_gamma1_family_closure.md). Max per-seed mean Δ across all 5 lr_cr values = +0.0117 at lr_cr=0.10, well below the 0.02 threshold. The shape — atom-vs-atom geometry — does not carry corpus signal regardless of magnitude.
- **Additional finding (Report 114 §"Finding 2"):** lr_cr ∈ {0.20, 0.50} produce *negative* mean Δ (−0.010, −0.017). Stronger atom-vs-atom repulsion actively *degrades* the codebook below the shuffled-token-control baseline. This is a real curve shape (peak at 0.10, negative at higher magnitudes), not noise.
- **Constraint on future mechanisms (Report 114 §"Finding 3"):** any candidate that compositionally includes atom-vs-atom repulsion at non-trivial magnitude inherits this negative-Δ regime. Relevant for Γ4 EqProp's negative-phase term, Γ5 Hyperseed's competition-with-decay, and parent precommit F2 (symmetric Γ1.c). **F2 is retired by this finding.**
- **Test-harness reproducibility:** lr_cr=0.10 cell at seeds 0,1,2 reproduces Report 113 per-seed Δ values (+0.030, +0.005, +0.000) to three decimal places across separate Colab sessions. Determinism confirmed.
- **experiment-result-auditor 2026-05-27:** 5/6 done-gates PASS (gate 5 = STATUS update, landed this session).
- **Pre-committed escalation invoked:** F1→b path. **Γ2 (bundle-first scene memory) precommit is the next deliverable**, with full mp-grill-with-docs + anti-homunculus reviewer + experiment-result-auditor cycle, per [Path γ survey](../emergent-codebook/path-gamma-mechanism-family-survey.md) §"Γ2".

**Status:** Γ1 family closed. F1 precommit complete. Next session-step: Γ2 precommit drafting.

---

*See also: [Γ1 parent precommit](2026-05-27-path-gamma-gamma1-context-residual-precommit.md),
[Report 113](../../reports/113_path_gamma_gamma1_headline_gate.md),
[Path γ mechanism-family survey](../emergent-codebook/path-gamma-mechanism-family-survey.md),
[STATUS.md](../../STATUS.md).*
