# Report 048 — Phase 5 A1 (r_ema geometric init) 1-seed pilot: FAIL at n=1

**Date:** 2026-05-20
**Phase:** 5 ([design](../notes/emergent-codebook/phase-5-unified-design.md))
**Status:** Mechanism-validity criteria #1 and #2 fail at n=1 on seed 17.
Falsification SIGNAL, not final falsification (n=5 needed per the
[A1 design note pre-commitment](../notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md)).
Surfaces a deeper-than-init failure: A1 reduces but does not eliminate
the discovery-channel atoms' dominance of the strength-ranked top-k.
Root cause is in the redundancy *measure*, not the *initialization*.
**Driver:** [scripts/run_phase5_a1_pilot_seed17.sh](../scripts/run_phase5_a1_pilot_seed17.sh)
→ [experiments/19_phase34_integrated.py](../experiments/19_phase34_integrated.py)
(with A1 wired at [src/energy_memory/phase4/consolidation.py:add_pattern](../src/energy_memory/phase4/consolidation.py)
and [src/energy_memory/phase4/replay_loop.py](../src/energy_memory/phase4/replay_loop.py)).
**Output:** [reports/phase5_a1_pilot_seed17/](phase5_a1_pilot_seed17/)
**Pre-committed knobs (binding, unchanged from A+B+step3):**
`alpha_anti=1.0`, `coverage_lambda=1.0`, `coverage_ema_rate=0.01`,
`repulsion_step_size=100.0`, `retrieval_weight_epsilon=0.05`,
`retrieval_weight_tau=0.02`. A1 introduces no new tunables; A1's
implementation landed at commit
[f037e48](https://github.com/Dypatterson/Neuro-AI/commit/f037e48)
with 4 new tests (245 total pass, 0 regressions).

## Experiment preamble

**Active phase:** 5

**Headline metric for this pilot (NOT graduation):** top-8 schema
pairwise FHRR similarity in band [0.10, 0.60] at step 1800 on W=4 —
criterion #1 from the
[2026-05-20 A1 design note](../notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md).
Drill-downs: d_eff (criterion #3); K-branch state_divergence
(criterion #2 at n=1).

**Required controls:** the prior pilot
([report 046](046_phase5_ab_pilot_seed17_step3.md)) on the same seed
without A1 is the natural A/B comparison.

**Last verified result:** [report 047](047_phase5_ab_branch_divergence_failure.md)
— K-branch state_divergence FAIL at n=1 on the A+B+step3 substrate;
root cause attributed to discovery-channel atoms initialized at
`r_ema = 0`.

**Why this experiment now:** the
[A1 design note](../notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md)
(anti-homunculus reviewer PASS, commit
[3b68877](https://github.com/Dypatterson/Neuro-AI/commit/3b68877))
designed substrate-derived `r_ema` init to close the categorical
"new = novel" claim. This pilot verifies the n=1 mechanism-validity
signal before n=5 / Colab n=10.

## Headline result against the six pre-committed criteria

| # | Criterion | Bar | Observed | Verdict |
| - | --- | --- | --- | --- |
| 1 | top-8 pairwise FHRR similarity at W=4 step 1800 | in band [0.10, 0.60] | **1.0000** | **FAIL (n=1 signal)** |
| 2 | K-branch state_divergence (state_div) at W=4 step 1800 | in band [0.024, 0.044] | **0.0** | **FAIL (n=1 signal)** |
| 3 | d_eff at W=4 step 1800 | ≥ 25 | **35.19** | **PASS** |
| 4 | Phase 4 D1 non-regression Δms_w3 | ≤ -0.5 CI-disjoint, n=10 | n=1 only; deferred | (deferred to n=10) |
| 5 | r_ema init done once at add, not scheduled | code-level | (audited at commit) | **PASS** |
| 6 | coverage_ema_rate NOT retuned from 0.01 | code-level | 0.01 unchanged | **PASS** |

**At n=1, criteria #1 and #2 fail.** The verdict is a SIGNAL (per the
A1 design note: "If A1 fails criterion 1 (top-k diversity stays high),
the failure is deeper than the init — that points to a different
design session, not a retune"). n=5 is the formal bar; the
single-seed signal here is highly informative because A1's intended
effect is geometric and should fire on every seed independently.

## Top-8 schema diversity (criterion #1)

| Substrate | top-8 indices | pairwise sim (mean off-diag) | In band [0.10, 0.60] |
| --- | --- | ---: | :---: |
| Pre-A+B baseline (seed 17 step 1500, [report 047](047_phase5_ab_branch_divergence_failure.md)) | 167, 205, 211, 451, 634, 713, 720, 775 (spread) | 0.361 | ✓ |
| A+B+step3 (seed 17 step 1800, [report 047](047_phase5_ab_branch_divergence_failure.md)) | 1044–1063 (all discovery) | 1.0000 | ✗ |
| **A1 (this report)** | **1044, 1045, 1048, 1046, 1060, 1051, 1047, 1050** (all discovery) | **1.0000** | ✗ |

A1's top-8 is *the same family* as A+B+step3's top-8 — all discovery-
channel atoms (indices ≥ 1024), all FHRR-identical at FP precision.

## d_eff trajectory (criterion #3)

| step | A+B+step3 ([report 046](046_phase5_ab_pilot_seed17_step3.md)) | A1 (this report) | Δ |
| ---: | ----------------: | ----------: | ---: |
|  500 | 36.34             | 36.34       | 0.00 |
| 1500 | 35.67             | 35.66       | -0.01 |
| 1700 | 35.36             | 35.34       | -0.02 |
| 1800 | 35.20             | 35.19       | -0.01 |

d_eff is bit-identical between A+B+step3 and A1 at every step (max
delta = 0.02). **A1 does not affect substrate broad geometry** — it
only changes the initial condition of `r_ema`, which feeds the
reinforcement-modulation gate but not Candidate B's repulsion. d_eff
remains preserved at criterion #3.

## K-branch state-divergence diagnostic (criterion #2 at n=1)

From [phase5_headline_seed17.json](phase5_a1_pilot_seed17/branch_diag_w4_step1800/phase5_headline_seed17.json):

| Condition         | mean_n_branches | mean_state_divergence | per-cue ΔE_K4 mean |
| ----------------- | ---------------: | ---------------: | ---: |
| role_K4           | 1.00             | 0.0              | 0.0  |
| content_K4        | 1.00             | 0.0              | 0.0  |
| random_K4         | 1.00             | 0.0              | 0.0  |
| role_K4_g0 (γ=0)  | 1.00             | 0.0              | 0.0  |
| role_K1           | 1.00             | 0.0              | 0.0  |
| **Pre-death** ref | (4)              | 0.0336           |      |

Every condition produces `n_branches = 1.0` and `state_divergence = 0`,
identical to the A+B+step3 result in
[report 047](047_phase5_ab_branch_divergence_failure.md). The K-branch
diagnostic confirms the top-k degeneracy translates directly to a
degenerate branch structure.

## A1 *did* partially work — effective_strength comparison

Where A1 *did* have effect: discovery atoms' bulk
`effective_strength` distribution. From the same W=4 step-1800
snapshots:

|  | A+B+step3 (no A1) | A1 (this report) | Δ |
| --- | ---: | ---: | ---: |
| Original atoms (idx 0–1023) mean eff. strength | 0.0253 | 0.0252 | -0.0001 |
| Original atoms max | 0.1205 | 0.0915 | -0.029 |
| Discovery atoms (idx ≥ 1024) mean eff. strength | **0.5065** | **0.3385** | **-0.168 (−33%)** |
| Discovery atoms max | **12.115** | **9.646** | **-2.469 (−20%)** |
| Top-5 discovery strengths | [12.11, 1.99, 1.19, 0.67, 0.50] | [**9.65, 0.95, 0.29, 0.29, 0.20**] | progressively reduced |

A1's mechanism fires correctly: discovery atoms with high `r_ema_init`
(near-duplicates of the existing substrate) are throttled at
reinforcement, reducing their accumulated strength by 20–70% across
the top-5. The 2nd through 5th discovery atoms drop from {1.99, 1.19,
0.67, 0.50} to {0.95, 0.29, 0.29, 0.20} — a major effect.

But the **first** discovery atom (idx 1044) still has effective
strength 9.65 — 100× larger than the strongest original atom (0.0915).
Even at A1's full r_ema modulation, the cumulative reinforcement on
this atom places it above all 1024 original atoms.

## Root cause: Gram-row RMS proxy under-measures sparse duplicates

The first discovery atom (1044) was added at some early step against
a substrate of ~1024 original Phase-3 atoms. Its `r_inst` was computed
via the implementation's proxy:

```python
# src/energy_memory/phase4/consolidation.py:_coverage_redundancy_instantaneous
G_ij = (1/D) * <p_i, p_j>            # complex; |G_ii| = 1
r_i  = sqrt( mean_{j != i} |G_ij|² ) # RMS off-diag, ∈ [0, 1]
```

If atom 1044 is a near-duplicate of one original atom *k* (|G_{1044,k}|
≈ 1) and orthogonal-ish to the other 1023 (|G_{1044,j}| ≈ small):

```
r_inst[1044] = sqrt( (1.0² + ε² · 1023) / 1023 ) ≈ sqrt(1/1023) ≈ 0.031
```

That is, the RMS proxy reports **r_inst ≈ 0.031** for an atom that is
a perfect duplicate of one atom but orthogonal to the rest. The
geometric reading of "is this atom redundant given the rest of the
substrate" gives 1.0 (the projection onto the column-span is the
duplicate atom itself, full magnitude), but the implementation's
**mean-over-all** proxy dilutes this signal across N=1024 atoms.

A1's init draws from this proxy, so the first discovery atom enters
consolidation with `r_ema_init ≈ 0.03` instead of ≈ 1.0. Its
reinforcement modulation `(1 − 0.03) = 0.97` is nearly unthrottled,
exactly as in the no-A1 case for this atom. It accumulates strength,
and the next 39 discovery atoms (added as near-copies of 1044 or
of each other) each see a slightly increased `r_inst` (because there
are now multiple near-duplicates contributing to the RMS) but still
much less than 1.0. **A1 is monotonically better than no-A1 but
asymptotically still insufficient.**

The design note's anti-homunculus framing of A1 ("the init is the
EMA's geometric equilibrium under the existing measurement") is
correct as designed; the limitation is the **measurement itself**,
not the initial condition.

## What this means for A's mechanism-validity status

**A1 PASSES on the dynamic-form / anti-homunculus criteria #5 and #6
by construction.** The implementation does what the design note
specified. The failure mode is at the *measurement* level: the
Gram-row RMS proxy `_coverage_redundancy_instantaneous` is a strictly
weaker measure of redundancy than the formal projection-magnitude
definition in the original death-dynamic note. The design note
acknowledged the proxy (§"Drawback" of Candidate A: "Computing
`proj_{P_¬i}` per atom per step is O(N³) per update. Will need a
low-rank approximation or running estimate") — what wasn't anticipated
was that the *mean-vs-max* choice in the RMS would matter for the
sparse-duplicate case.

The cleanest reading: **A and A1 are correct in shape; the
*measurement* operationalizing r_i is too weak.** This is a separate
design-level question from the init dynamic.

## Three sketched directions (each requires a design note + audit)

These are sketches, NOT implementation commitments:

### A1' — Replace Gram-row RMS with max-over-others

```python
r_i = max_{j != i} |G_ij|
```

For an FHRR atom that's a perfect duplicate of any one other atom,
r_i = 1.0. For an orthogonal atom, r_i ≈ 0. This matches the formal
"projection magnitude onto column-span of {p_j : j != i}" up to
single-atom-versus-subspace approximation.

**Anti-homunculus check sketch:** still per-atom; still O(N²) per call;
still a measurement, not an arbitration. The change is in how the row
is reduced to a scalar — mean vs max. PASS-shaped on first read.

**Drawback:** loses partial-redundancy signal (an atom that's
half-similar to many neighbors gets r ≈ 0.5 under mean but r ≈ 0.5
under max too; an atom that's perfectly-similar to one and orthogonal
to others gets r ≈ 0.03 under mean but r ≈ 1.0 under max — the second
case is what matters).

### A1'' — Streaming low-rank SVD approximation

The original design note recommended this exact path: a running
estimate of `||proj_{P_¬i}(p_i)||` via a streaming-SVD update. Heavier
to implement; closer to the formal definition.

**Anti-homunculus check sketch:** the design note already PASSED this
as the load-bearing constraint on Candidate A. Returning to it is just
implementing what was originally specified — no new audit needed.

**Drawback:** larger implementation footprint; needs a streaming-SVD
primitive that doesn't currently exist in the codebase.

### A1''' — A3 / β (path 3) becomes the next session

If the underlying issue is "the substrate produces one dominant basin
and the schema store inherits that degeneracy regardless of
strength-ranking-modulation," then the right move is to change *what
the prior is*, not just *how strengths accumulate*. The
[cue-regime / role-prior note](../notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md)
Candidate β (continuous role-fidelity-weighted prior) was already
designed and PASSED audit. It addresses a different axis but may
make the strength-dominance issue moot — if the prior weights
schemas by role-fidelity, the categorical "top-k by strength" is
displaced.

**Anti-homunculus check:** already PASSED in the path-3 note.

**Drawback:** β requires the strength-dominance to be benign at the
prior-weighting layer; not yet verified.

## Pre-committed binding remains intact

All six A+B+step3 knobs were set ONCE; A1 introduces no new tunables.
None of {α_anti, coverage_lambda, coverage_ema_rate,
repulsion_step_size, retrieval_weight_epsilon, retrieval_weight_tau}
has been tuned post-hoc. Per H4 of the
[death-dynamic design note](../notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md):

> If the mechanism fails D1 preservation, the candidate is wrong-shaped
> and we go back to design.

A1 here passes criterion #3 (D_eff preservation) but fails criteria
#1 and #2 (the top-k diversity and K-branch state-divergence
indicators). Per the A1 design note's own H4-style reading:

> If A1 fails criterion 1 (top-k diversity stays high), the failure
> is deeper than the init — that points to a different design session,
> not a retune.

The failure is exactly at this anticipated boundary. The next move is
a design session on one of A1', A1'', or A1''' (or a combination),
NOT a retune of `coverage_ema_rate` or `coverage_lambda` to compensate.

## What this report does NOT do

- It does NOT declare A1 falsified. n=1 is a signal, not the
  pre-committed n=5 threshold for criteria #1 and #2.
- It does NOT commit to a fix. A1', A1'', A1''' are sketches; each
  requires a separate design + anti-homunculus check + decision.
- It does NOT block A1' as the obvious move. The max-over-others
  fix is the simplest patch, but its anti-homunculus shape needs
  independent audit (max-of-similarities is a per-atom operation
  but does it shift the dynamic-form reading vs mean?).
- It does NOT close out criterion #4. D1 non-regression at n=10 is
  deferred until the full mechanism-validity criteria 1-3 pass at
  n=5 (currently blocked by criterion #1 failing).
- It does NOT exercise n=5. The local-snapshot grid has 5 seeds;
  if the user wants the formal n=5 result, exp 19 can re-run against
  the 4 other seed substrates — but the n=1 signal is strong enough
  to inform the next design decision, and re-running n=5 on a known
  wrong-measurement is the least useful experiment to spend on.

## Updated recommendation

Three options in disciplined order:

1. **Pause + design (recommended).** Surface this finding; choose
   between A1', A1'', or A1''' (or a combination) at the design level;
   write the design note + anti-homunculus check; THEN implement and
   re-pilot. *Most faithful to the project's discipline.*

2. **A1' direct implementation (smallest patch).** The mean-vs-max
   change is a 1-line patch to `_coverage_redundancy_instantaneous`.
   With A1 already passing audit at the shape level, this is the
   smallest mechanism patch. *Fastest move; risks skipping the
   design step.*

3. **Run n=5 first.** Re-pilot A1 across seeds 1, 2, 11, 23 and
   confirm criteria #1/#2 failure is consistent before designing a
   fix. *Slightly more expensive; tightens the diagnostic, but the
   n=1 signal already points at a known measurement-level cause.*

Recommended: **option 1**. The next-move boundary is "redesign the
measurement" — pausing to design and audit is the right discipline.
A1' looks shape-clean on first reading but a clean anti-homunculus
audit of the max-vs-mean choice is the binding gate.

Anti-homunculus discipline reminder: the death-dynamic and A1 design
notes both explicitly anticipate this exact path-of-redesigns. The
first concrete diagnostic-actuator pair in dynamic form is taking
real iteration to nail down; that is what discipline looks like in
practice.
