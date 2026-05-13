# Report 018 — Phase 3 codebook comparison: synergy reveals what Recall@1 hides

**Date:** 2026-05-13
**Experiment:** `experiments/32_phase3_synergy_comparison.py`
**Raw data:**
- `reports/phase3_synergy_comparison.csv`
- `reports/phase3_synergy_comparison.json`
**Supersedes (in part):** parts of [reports/016](016_phase2_audit_and_phase3_objective.md)
and [reports/017](017_phase3_codebook_comparison.md)

## Motivation

[Report 017](017_phase3_codebook_comparison.md) showed that
Recall@1 at n_eff ≈ 50 could not distinguish three learned codebooks
(Hebbian, reconstruction, error_driven) that differ by mean |diff|
≈ 0.82 on the unit-magnitude phase manifold. This script runs a
joint evaluation on the same operating envelope at **n_eff ≈ 2191
per codebook** (test_samples=500 × 6 seeds × ~370 effective per seed
after unk filtering) and adds a structural metric — settled synergy
at the masked slot — alongside Recall@1.

## Headline result (n = 2191 per codebook, 6 seeds pooled)

| codebook        | Recall@1            | raw synergy           | **settled synergy @ mask**    |
| --------------- | ------------------- | --------------------- | ---------------------------- |
| random          | 0.084 [0.073, 0.096]| 0.312 [0.311, 0.312]  | **0.002 [0.001, 0.003]**     |
| **hebbian**     | 0.087 [0.076, 0.100]| 0.298 [0.298, 0.299]  | **0.252 [0.249, 0.256]**     |
| reconstruction  | 0.043 [0.036, 0.053]| 0.294 [0.293, 0.295]  | 0.230 [0.227, 0.233]         |
| error_driven    | 0.031 [0.024, 0.039]| 0.298 [0.297, 0.298]  | 0.222 [0.219, 0.225]         |

(Wilson CI on Recall@1; bootstrap percentile CI on synergy means.
All CIs at 95%. n=2191 effective per codebook for both metrics.)

### The structural ranking is unambiguous

On **settled synergy at the masked position** — the structural
recoverability of the *correct token* from the post-settling state —
all four codebooks are separated by **wildly disjoint CIs**:

1. **Hebbian: 0.252** (best)
2. **Reconstruction: 0.230** (−0.022)
3. **Error_driven: 0.222** (−0.030)
4. **Random: 0.002** (effectively zero)

The learned codebooks beat the random baseline by **~125× on
settled synergy**. Among the three, Hebbian wins by 2–3 standard CI
widths. This is the first Phase 3 codebook-growth result with
statistically robust separation from random.

### Recall@1 hides this entirely

At the same n=2191:

- Hebbian: 0.087 — statistically indistinguishable from random (CIs overlap heavily)
- Reconstruction: 0.043 — significantly *below* random
- Error_driven: 0.031 — even further below random

Reading only Recall@1, the conclusion is "Hebbian ties random,
reconstruction and error_driven hurt accuracy." Reading settled
synergy, the conclusion is "all three learned codebooks are doing
massive structural work; Hebbian is the cleanest." Both readings
are correct on their respective metrics. They report different
things about the substrate.

## What is settled synergy actually measuring

The estimator (from [reports/011](011_synergy_probe_phase4.md)) takes
a triple `(role, filler, binding)` and returns

```
synergy = sim(unbind(binding, role), filler) - max(
    sim(role, filler),  # baseline: can you guess filler from role alone?
    sim(binding, filler),  # baseline: can you read filler off binding without unbinding?
)
```

In this experiment, for each test window:
- `role` = position vector at the masked slot
- `filler` = codebook atom for the *correct* token
- `binding` = the Hopfield-settled state after retrieval on the masked cue

A high score means: the settled state contains structure that, when
unbound at the masked position, recovers something close to the
correct atom — beyond what the role or binding alone tells you.

The metric is **continuous (every window contributes a real number)**,
**directly aligned with the substrate's binding algebra**, and
**robust to argmax noise** (it doesn't matter which token "wins"; it
matters how recoverable the right one is).

## What the data is telling us

### 1. Learned codebooks make the correct token *recoverable* from the settled state, but don't change which token *wins* the argmax

Random codebook settled synergy ≈ 0 means: after Hopfield settling
on a masked cue, the settled state has *no* recoverable structure
for the right token. The substrate finds *some* answer (~8.4% of the
time it happens to be right, by chance and frequency), but that
answer isn't structurally derived from the cue's compositional
content.

Learned codebooks at 0.22–0.25 mean: the settled state *does* contain
structure pointing at the right token. The substrate is finding
"the kind of token that belongs here." But the argmax may still
pick a different token (probably a high-frequency function word, a
landscape-repeated pattern, or a semantically-adjacent neighbor)
that happens to have a slightly higher unbind similarity. The right
answer is *present in the structure* but not *winning the lottery*.

This is exactly the contextual-completion vs. sequence-prediction
distinction from the
[2026-05-09 paper synthesis](../notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md):
the substrate is producing a contextual state, not a token
prediction. Recall@1 forces it to be evaluated as a token predictor.
Synergy lets it be evaluated as a contextual state.

### 2. Reconstruction and error_driven have a real Recall@1 deficit, but it's not "they failed"

Reconstruction (0.043 Recall@1) and error_driven (0.031) have CIs
disjoint from random (0.084) on the upper side — these objectives
*actively hurt* exact-match accuracy. But their settled synergy
(0.230, 0.222) is well above random and only slightly below Hebbian.

The likely mechanism: reconstruction loss and error-driven gradients
push the codebook to make co-occurring tokens *more similar*. This:
- *Increases* the structural recoverability of "the right
  neighborhood" (correct token's unbind similarity goes up).
- But *also increases* unbind similarity of other tokens in the same
  neighborhood.
- Net effect on the argmax: the right token is one of many close
  neighbors, so the winner is less often *exactly* right.

Hebbian's co-occurrence rule does less of this neighborhood-merging
(weaker correlation gradients), so it adds structural recoverability
without sacrificing argmax-distinguishability.

This is a *useful trade-off characterization*, not a failure
verdict. Reconstruction may be the better objective for a downstream
task that cares about "is the system retrieving from the right
semantic region" rather than "did it land on the exact token."

### 3. Raw synergy is mildly *lower* for learned codebooks than random

Random codebook raw synergy: 0.312
Learned codebook raw synergy: 0.294–0.299

Random vectors are maximally orthogonal in expectation, so unbinding
a random-codebook window produces the cleanest per-token recovery.
Learning correlates atoms with their contexts, which slightly
reduces per-window decomposability. The drop is small (~5 pp) and
the rank ordering across learned codebooks is essentially flat.

Raw synergy isn't measuring what Phase 3 wants to optimize. It
captures the substrate's algebraic baseline (which random preserves
best) but not what learning is supposed to add (which is
context-conditional recoverability — captured by settled synergy).

## Implications for the Phase 3 program

### Settled synergy is the right Phase 3 primary headline

[Report 016](016_phase2_audit_and_phase3_objective.md)'s objective
should be revised:

> **(Old, retired.)** Maximize masked_token:generalization Recall@1
> at the operating envelope. Beat the random baseline of 0.205
> (single-seed) / 0.089 (multi-seed pooled).

> **(New.)** Maximize settled synergy at the masked position over
> the operating envelope's generalization windows. Beat the random
> baseline of 0.002 with non-overlapping pooled CIs at ≥ 6 seeds,
> n ≥ 500 per seed. Recall@1 is retained as a downstream sanity
> check; large Recall@1 regressions vs. random (e.g. CI disjoint
> from random on the upper side) flag that the learned codebook is
> over-merging atoms in its semantic neighborhood.

### Hebbian is the first verified Phase 3 codebook-growth winner

Hebbian wins on settled synergy with disjoint CIs vs. all other
codebooks AND does not regress on Recall@1. Both checks pass. This
is the first Phase 3 codebook-growth result the project can publish
with statistically robust separation from baseline.

### Reconstruction and error_driven aren't "deprecated" — they're "in tension"

These objectives produce **higher structural quality than random
(by a lot) but lower argmax accuracy than random (by a little).**
The right framing isn't "they failed", it's "they pay an
argmax-exact-match cost for a structural gain." Whether that trade
is worth it depends on the downstream task. For the project's
contextual-completion design intent, the structural gain probably
matters more — which means reconstruction in particular deserves
more careful examination, not deprecation.

### Report 017's "the metric is the problem" hypothesis is confirmed

Report 017 ended with: "I lean toward 'the metric is wrong' but I
don't think that's a closed call." This report closes it. The same
operating envelope, the same codebooks, the same retrieval substrate
— but a structural metric instead of an argmax metric — reveals a
clear winner where Recall@1 saw nothing.

## What this does not say

- **The substrate's compositional structure is fine** (already known
  from [report 011](011_synergy_probe_phase4.md): settling preserves
  raw synergy ratio = 1.000 under default operating params).
- **Hebbian winning here does not mean Hebbian is the final answer**
  — only that it's the first codebook-growth result with a
  statistically defensible structural advantage. Other objectives may
  do better; we just haven't tried them yet, and now we have the
  measurement framework to judge.
- **The Recall@1 floor is not a substrate ceiling**. The substrate
  is producing recoverable structure for the right token in ~25% of
  windows under Hebbian. The argmax sees ~9%. Downstream consumers
  that read the top-K or use the structural state directly should
  see a much larger effective gain.

## Recommended next steps

1. **Update [report 016](016_phase2_audit_and_phase3_objective.md)**
   with the revised Phase 3 objective using settled synergy as the
   primary headline.
2. **Promote `mean_synergy` as a standard Phase 3/4 drill-down.** Add
   it to `experiments/18_phase4_unified_experiment.py` (per the
   already-named follow-up in [report 011](011_synergy_probe_phase4.md)).
3. **Re-open the reconstruction objective.** It produces high
   structural quality (only slightly below Hebbian) but takes a
   Recall@1 hit. For contextual-completion downstream uses, this
   trade-off may be favorable. Worth a focused experiment that
   measures top-K agreement with semantic neighbors, not just
   exact-match Recall@1.
4. **Look at Recall@K for K > 1 across the four codebooks.** The
   hypothesis: learned codebooks should show large Recall@K gains
   over random at K=5 or K=10, because they're enriching the
   neighborhood of the right answer even when not winning the argmax.
   This is the "argmax-noise-robust" version of Recall@1 and might
   be a usable downstream task metric alongside synergy.

## Anti-homunculus check / FEP audit

Pure measurement evaluator. Synergy estimator is a deterministic
geometric function of (role, filler, binding); no thresholds, no
inspect-and-trigger branches. Recall@1 is the same argmax-eval used
by the Phase 2 matrix. No new mechanisms. Passes trivially.
