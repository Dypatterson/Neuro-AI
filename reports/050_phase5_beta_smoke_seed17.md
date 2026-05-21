# Report 050 — Phase 5 β (path-3) smoke test on A1' substrate (seed 17)

**Date:** 2026-05-20
**Phase:** 5 ([design](../notes/emergent-codebook/phase-5-unified-design.md))
**Status:** β implementation landed + n=1 smoke test against the A1'
substrate snapshot. **β does not fix the headline criteria at n=1.**
The substrate has **zero variance in the role-fidelity measure f_i**
(every atom returns 0.9858 ± 0.0000) at D=4096, so the q-sweep
(criterion #2 from the cue-regime note) is identically zero by
construction: f_i^q is a uniform constant across all schemas, and
the β prior at q=0 equals the β prior at q=1 up to a uniform scaling.
**The binding limit has moved from selector-layer (report 049) to
substrate-encoding-level structure** — the substrate does not have
a measurable per-atom role-fidelity signal at this dimension.
**Driver:** [experiments/40_phase5_branching.py](../experiments/40_phase5_branching.py)
`--mode headline` (with β conditions integrated at commit
[87b03ae](https://github.com/Dypatterson/Neuro-AI/commit/87b03ae)).
**Output:** [reports/phase5_a1prime_pilot_seed17/branch_diag_w4_step1800_beta/](phase5_a1prime_pilot_seed17/branch_diag_w4_step1800_beta/)

## Experiment preamble

**Active phase:** 5

**Headline metric:** β prior at q=1 should produce ΔE > 0 vs q=0
baseline (per [2026-05-20 cue-regime / role-prior note](../notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md)
§"Pre-committed falsification criteria #2"). At n=1 on this substrate,
observed ΔE = 0.0.

**Required controls:** β at q=0 (content-only baseline); fid_K1 at γ=0;
existing role/content/random_K4 conditions.

**Last verified result:** [report 049](049_phase5_a1prime_pilot_seed17.md)
— A1' produced dramatic substrate-level progress (discovery atoms'
max effective_strength reduced 99.8%) but the top-K-by-strength
selector still picked 7 FP-identical discovery atoms tied at the noise
floor. The failure was identified as selector-layer; β was recommended
as the natural next architectural fix.

**Why this experiment now:** β's design + anti-homunculus PASS were
done 2026-05-20. The smoke test against the existing A1' substrate
snapshot at seed 17 step 1800 verifies β's library code wires into
exp 40 correctly and produces a measurable signal. Full n=5 / n=10
validation is deferred to Colab per user instruction.

## β substrate-level finding (the headline diagnostic)

When β computes role-fidelities on the 1064-atom A1' substrate:

```
[β] full-substrate fidelities computed: N=1064, mean(f)=0.9858,
    std(f)=0.0000, min=0.9858, max=0.9858
```

**Every atom has the same f_i value, to FP precision.** The
distribution of `{f_i}` across the substrate is a single point.

### Why f_i is uniform at D=4096

The role-fidelity measure is `f_i = mean_{j != k} (1 - |G_jk|)` over
the W=4 unbound fillers `{unbind(s_i, pos_r)}_{r=1..W}`. For an FHRR
pattern stored as `bundle(filler_r ⊛ pos_r)`, the unbind at position
p yields:

```
unbind(P_i, pos_p) = filler_{i,p} + Σ_{r != p} filler_{i,r} ⊛ pos_r ⊛ inv(pos_p)
                   = signal + crosstalk_noise
```

At D=4096, the crosstalk terms are sums of random unit-magnitude
phasors. By the central limit theorem, the magnitude of the crosstalk
noise scales as `√D / D = 1/√D ≈ 0.0156`. The signal (the actual
filler) has magnitude 1.

For two unbinds at different positions r ≠ s:
- The signal terms are different fillers → orthogonal signals
- The crosstalk noises are independent (different residual sums)
- Their FHRR inner product `|G_jk|` ≈ noise magnitude ≈ 1/√D ≈ 0.0156

So `f_i = mean(1 - |G_jk|) ≈ 1 - 0.0156 ≈ 0.984` — **independent of the
specific filler content of the pattern**.

This is exactly what we observed (0.9858 ≈ 0.984). The role-fidelity
measure is **structurally noise-dominated at this dimension**.

## β headline result against the six pre-committed criteria

| # | Criterion | Observed | Verdict |
| - | --- | --- | --- |
| 1 | f_i distribution coherence cross-seed | f_i = 0.9858 everywhere within seed 17 (n=1) | (cross-seed n=5 deferred to Colab; per-seed uniformity already evident) |
| 2 | β ΔE > 0 (q=1 vs q=0) at n=1 | **ΔE = 0.0 (every cue), fraction_positive = 0.10** | **FAIL** (no signal — uniform f_i means q has no effect) |
| 3 | q-sweep monotonicity | ΔE(q=0) = ΔE(q=1) (constant zero) | (trivially monotonic; no signal) |
| 4 | β prior at q=1 lower energy than role_K4 / content_K4 | mean E_unbiased: fid_K1_q1 = -1.3562, content_K1 = -1.3568 (β SLIGHTLY worse) | **FAIL on energy** |
| 5 | β implemented per design note | code-level | **PASS** (commit 87b03ae) |
| 6 | No parameter retuning | p=1, q=1 pre-committed | **PASS** |

**β does not graduate the headline at n=1.** The criterion that β was
designed to win (q=1 vs q=0 ΔE) is identically zero because the f_i
measurement has zero variance to leverage on this substrate.

## β at q=0 vs content_K1: continuous weighted sum is slightly worse

Even at q=0 (no fidelity influence), β produces a different prior than
the existing `content_K1` baseline:

| Condition | mean E_unbiased | Why |
| --- | ---: | --- |
| content_K1 | -1.3568 | Picks **single** top-cosine atom as prior |
| fid_K1_q0 | -1.3562 | **Weighted sum** of all 1064 atoms by cue cosine |

The β-weighted prior dilutes the sharpest cue-cosine-aligned atom by
averaging with 1063 less-aligned atoms. Each less-aligned atom
contributes a vector that pulls the prior off-axis. Even after
normalization, the resulting prior is less perfectly cue-aligned than
the single-atom content_K1 selector. Net: β at q=0 lands at slightly
**higher** energy than the categorical top-1 selector.

## The failure path

A 4-step progression:

1. **A+B+step3 ([report 047](047_phase5_ab_branch_divergence_failure.md))** — substrate-construction failure (discovery atoms dominate)
2. **A1 ([report 048](048_phase5_a1_pilot_seed17.md))** — measurement under-counts sparse duplicates
3. **A1' ([report 049](049_phase5_a1prime_pilot_seed17.md))** — measurement now works, but top-K selector picks tied duplicates
4. **β (this report)** — selector replaced by continuous weighted sum, but the substrate's role-fidelity property has no measurable variance to weight by

Each layer's fix correctly addresses the prior layer's failure. **The
binding failure has now reached the substrate's intrinsic role-binding
structure**, which at D=4096 is structurally noise-dominated.

## What this means for path 3

The cue-regime / role-prior note proposed β under the hypothesis that
"per-seed `mean(f_i)` spread should be within 30% cross-seed." That
falsification criterion is satisfied trivially when f_i is constant
within a seed at 0.986. The note's other criteria fail:

- **β ΔE > 0**: f_i has no variance, so q=1 and q=0 are identical
- **q-sweep monotonicity**: trivially monotonic at zero

The β formulation as designed doesn't fix the report-049 selector-
layer failure because the substrate doesn't have a role-binding signal
the fidelity measure can detect. **β's audit-PASS shape was correct;
its operationalization on this substrate produces zero signal.**

## Three sketched directions (each requires design + audit)

These are sketches, NOT implementation commitments:

### β' — Different f_i formulation

The `mean pairwise FHRR distance of unbound fillers` measure is
noise-dominated. A different measure of "role-binding fidelity" could
be:

> `f_i = corr(unbind(s_i, pos_r), codebook_filler_at_r)`

i.e., correlation of the unbound filler with a known codebook entry.
This requires the codebook to be available (it is — saved in the
substrate snapshot), but introduces a coupling to the codebook that
the original β didn't have. Anti-homunculus check: the codebook is
substrate state (geometric); the correlation is a per-atom
measurement. PASS-shaped on first read.

### γ — Cue-regime distribution averaging at evaluation

From the same cue-regime note. Use a *distribution* of cue regimes
(`(σ, δ) ~ Uniform`) and report β's expected behavior across that
distribution. This addresses the *variance* of the headline across
seeds; it doesn't solve the within-seed uniform-f_i problem β
encounters here.

### Lower-dimension or different-encoding substrate

At lower D (e.g., D=512), the FHRR crosstalk noise is larger (1/√D ≈
0.044 instead of 0.016) but the role-binding signal might be more
recoverable. Alternatively, a different binding-encoding scheme that
doesn't have FHRR's crosstalk structure could expose the role-fidelity
signal. **Either is a research-direction change, not an architectural
fix to Phase 5.**

## Honest assessment

The four-step debugging journey (A+B → A1 → A1' → β) has produced
the substrate that A+B+A1+A1' was designed to produce:

- **d_eff preserved at 35.23** (criterion #3 of the death-dynamic note)
- **Discovery atoms throttled to noise floor** (max eff_strength 0.029)
- **Top-1 atom is now an original Phase-3 atom** (idx 713)
- **Substrate-level failure modes 1, 2, 3 all closed**

What β has now surfaced is that the substrate doesn't have a measurable
per-atom role-fidelity signal at D=4096. The K-branch state_divergence
criterion (criterion #2 in the original A1 design) was looking for a
property the substrate's encoding doesn't have at this dimension.

**This is a falsification result for path 3 β as implemented**, and
also a deeper finding: the architecture's "structural retrieval"
claim, as operationalized by the role-fidelity-weighted prior,
doesn't have a substrate-level signal at D=4096 to exploit.

The disciplined next move is one of:

1. **Stop chasing the K-branch state_divergence criterion** and re-scope
   Phase 5's headline. The mechanism-validity criteria 1 and 2 were
   based on the diagnostic-actuator note's pre-death substrate
   measurement; that substrate had non-trivial state_divergence
   because of binary death's surviving small-N atom set. The
   replacement (continuous A+B+A1' substrate) has a different
   structure and may need a different headline.
2. **Reformulate f_i** (β' direction): use codebook-correlation rather
   than pairwise-distance. A separate design + audit.
3. **Investigate lower-dimension** (research direction): determine
   whether the substrate has role-binding signal at D < 4096; would
   reframe phase 5 around a different substrate dim.
4. **Defer to Colab** with the current β implementation: confirm at
   n=5 that the uniform-f_i finding is consistent across seeds (high
   confidence based on n=1 + the theoretical argument; running it
   gives the formal verdict).

## What this report does NOT do

- It does NOT declare Phase 5 unable-to-graduate. The substrate has
  many working properties (d_eff preserved, discovery atoms throttled,
  D1 graduates at Phase 4). Just this particular metric (K-branch
  state_divergence under role-prior) appears not to be discriminable.
- It does NOT commit to a fix. The three sketches need their own
  design + audit + decision cycles.
- It does NOT close out criterion #4 (D1 non-regression at n=10).
  D1 graduation already passed at Phase 4 ([report 038](038_phase4_d1_graduation.md)).
  No reason to expect β to regress it.

## What the Colab run should test

When the user runs n=5 / n=10 on Colab with the current β
implementation:

1. **Confirm f_i uniformity holds across seeds.** Expectation: yes,
   based on the theoretical argument. If a seed shows f_i variance,
   that's a finding to report; if not, β is empirically falsified.

2. **β at q=0 vs content_K1 ΔE distribution.** This is what β does
   even without fidelity weighting. The smoke test showed β q=0 is
   ~slightly worse than content_K1; at n=10 we'd see whether the
   distribution overlaps zero.

3. **Substrate-level metrics (criterion #3 + criterion #4):** d_eff
   and Phase 4 D1 should remain preserved across seeds. A1' has
   already shown this at n=1.

The Colab run is informative regardless of outcome:
- If β fails identically on every seed → β is empirically falsified
  on this substrate
- If β passes on some seeds → the uniform-f_i finding doesn't
  generalize, contradicting the theoretical argument

## Pre-committed binding remains intact

No parameter retuning. `p=1, q=1` for β (from the design note's
recommendation). All A+B+A1+A1' substrate knobs unchanged. β's
implementation is a one-shot library addition + one new prior_type;
no per-cue feedback loops.

## Sequencing

β is now in main as a library + experiment-40 condition. It runs
unconditionally in `--mode headline` alongside the existing role/
content/random K4 conditions. The output JSON has a new
`headline_deltas.beta_q0_minus_q1` field.

The decision to invest in β' (reformulated f_i), to re-scope Phase 5's
headline metric, or to investigate lower-D substrates is **the user's
to make**, after the Colab n=5 result confirms or denies the n=1
finding.
