# Phase 3+4 online codebook regression — diagnostic findings

## Question

Phase 3+4 integration (experiment 19, `reports/phase34_medium/`) shows the
online codebook updater collapsing retrieval to chance by ~2000 cues
when started from random codebook init. Dylan's commit message on
3aa23fc named this as "consolidations on slot-queries from non-stationary
retrievals oscillate the codebook into noisy territory."

The goal of experiments 22 and 23 was to localize the cause and propose
a fix.

## What we established

### 1. Pull is the load-bearing collapse driver, not push.

Experiment 22 (partial, single seed, W=2, random init, 5000 cues):

| Condition | top-1 at 5000 cues | trajectory |
|---|---:|---|
| no updates | 0.015 | flat, by design |
| pull-only | 0.015 | climbs to 0.039 by ~1500 cues, collapses to chance by ~3000 |
| push-only | 0.015 | flat; drift ≈ 0 — push barely moves atoms |

The contrastive push formula `(1+lr)·old − lr·avg → normalize` is
~5% of pull's per-event magnitude and never changes the codebook
materially. Removing it would not change downstream behaviour.

### 2. The collapse signature is *not* pairwise convergence.

Throughout the collapse, mean pairwise |cos| stays ~0.009, p95 only
moves 0.022 → 0.028. Atoms aren't drifting together — they're losing
target-specific information while staying pairwise diverse. Repulsion
at the deep-dive's default threshold of 0.7 never fires (experiment 23
condition C, `reports/phase34_stable/`).

### 3. The collapse mechanism is position-bias amplification.

Slot queries arrive as `slot_query = unbind(retrieved_state, position_mask)
= retrieved_state · conj(position_mask)`. With random init, retrieved
state is mostly noise → `slot_query ≈ noise · conj(position_mask)`.
Averaging across the failure buffer cancels the noise but preserves
`conj(position_mask)`. **Every target's pull centroid converges to
approximately the same direction**, so pull yanks every codebook entry
toward a common position-induced bias direction. Diversity in
high-similarity neighborhoods erodes; pairwise mean stays low because
atoms are still mostly random in their unbiased components.

### 4. Lower lr_pull *delays* the collapse but does not fix it.

Experiment 23 V2 run, lr_pull=0.02 (vs deep-dive default 0.1):

| step | top-1 | topk |
|---:|---:|---:|
| 0 | 0.015 | 0.015 |
| 500 | 0.073 | 0.239 |
| 1000 | 0.044 | 0.268 |
| 1500 | 0.049 | 0.278 |
| 2000 | 0.049 | 0.332 |
| 2500 | 0.049 | 0.293 |
| 3000 | 0.020 | 0.288 |
| 3500 | 0.054 | 0.268 |

vs lr_pull=0.1 (same updater): collapse to chance (0.015, 0.044) by
cues=3000.

At lr=0.02 the run is in an *oscillating plateau* rather than a
catastrophic collapse — same underlying instability, slower rate. Not a
fix; a delay.

### 5. The combined V2 fix prevents collapse but at a lower ceiling.

V2 = lr_pull=0.02 + mean-subtracted pull target + repulsion at threshold
0.05 / strength 0.05:

| step | top-1 | topk | repulsion events |
|---:|---:|---:|---:|
| 0 | 0.015 | 0.015 | 0 |
| 500 | 0.073 | 0.073 | 5 |
| 1000 | 0.073 | 0.073 | 8 |
| 1500 | 0.073 | 0.098 | 11 |
| 2000 | 0.073 | 0.107 | 11 |
| 2500 | 0.107 | 0.107 | 13 |
| 3000 | 0.044 | 0.132 | 26 |
| 3500 | 0.078 | 0.107 | 49 |

Monotonically more stable top-1 than B (lr=0.02 unmodified); top-1 stays
above chance throughout. Topk ceiling is lower than B's. Mean-subtract
removes the common-mode bias direction but also strips out signal that
helps the top-K candidate diversity.

## The deeper finding

Re-reading [phase-3-deep-dive.md:140-145](notes/emergent-codebook/phase-3-deep-dive.md):

> The error-driven pathway requires a known correct retrieval target —
> the actual masked token. This is available during training (over
> corpus or replay buffer) but not at runtime (live conversation has no
> ground truth).
>
> **Implication: the codebook's significant learning happens during
> dedicated training passes. Runtime use produces only Hebbian
> adjustments — small reinforcement of patterns that successfully
> retrieve.** This matches the existing project architecture: replay
> drives most consolidation, live use is mostly retrieval with light
> reinforcement.

The deep-dive specifies two modes:

1. **Batch error-driven training** (offline, full corpus) — implemented
   as `ErrorDrivenLearner` (Phase 3b), `ReconstructionLearner` (Phase 3c).
   Output: `phase3c_codebook_reconstruction.pt`.
2. **Online Hebbian reinforcement** (runtime, no ground truth) — *not
   implemented*. Reserved for live use.

The current `OnlineCodebookUpdater` does **error-driven contrastive
updates inside a streaming cue loop**. That is a third mode the
deep-dive never specified, sitting between the two intended ones. It
runs the error-driven update online from random init, which:

- Requires ground-truth target_ids (so it's not "runtime" in the
  deep-dive's sense), but
- Is interleaved with retrieval rather than separated batch processing
  (so it's not "dedicated training" either), and
- Couples each update to the just-perturbed retrieval state (closed
  feedback loop that the batch learners avoid by holding the codebook
  fixed for the duration of training).

**The regression is in a mode the architectural design did not
sanction.** The diagnostics-and-candidates work above confirms that
each individual fix (repulsion, lr decay, mean-subtract) addresses a
symptom but not the underlying mode mismatch.

## Recommendation

Three options, in order of project alignment:

### Option 1 — Honour the deep-dive's runtime-vs-training asymmetry

Discontinue online error-driven updates as currently configured. Replace
with the two pathways the deep-dive actually specifies:

- **Batch error-driven** (already implemented as Phase 3c
  reconstruction; the `phase3c_codebook_reconstruction.pt` artifact is
  load-bearing for all downstream phases).
- **Online Hebbian reinforcement** — *needs to be implemented.* The
  deep-dive specifies it at line 100:
  > **If q is high (above success threshold):** apply Hebbian update
  > with magnitude proportional to q. For each atom that participated
  > in the cue, drift slightly toward the *clean context bag* (bundle
  > of other atoms in the cue, no position bindings).

  Concretely: when retrieval succeeds (q above threshold), pull each
  participating atom toward `normalize(bundle of co-cue atoms − the
  atom itself)`. Magnitude small (deep-dive starting point 0.01,
  vs 0.1 for error-driven). No ground truth needed; the signal is
  retrieval success.

- **Replay-driven consolidation** (Phase 4 unified design) is the
  channel for continued learning beyond initial batch training. Stored
  patterns re-encoded through the current codebook handle drift; new
  patterns from replay introduce new structure.

### Option 2 — Keep the online error-driven mode, but only on top of pretrained codebook

The phase34_medium experiment used `--random-codebook`. The same
experiment with `phase3c_codebook_reconstruction.pt` was never run as
the headline test. If the pretrained codebook is stable enough that
online error-driven updates don't destabilize it, this is a viable
intermediate path. Worth a one-condition validation run before
committing to Option 1.

### Option 3 — Accept that online error-driven from random init is out of scope

Document the regression as a known boundary of the current architecture.
Update the phase notes so the random-init + online error-driven mode
isn't re-attempted. Move on to the Tier 1 next steps from the original
plan (Phase 4 headline-with-drift, HAM 5-seed validation).

## Recommendation order

**Option 2 first** as a small validation (one condition run, ~30 min):
load `phase3c_codebook_reconstruction.pt`, run the unmodified
`OnlineCodebookUpdater` at lr=0.1 for 3500 cues, see whether top-1
degrades from 0.121 (the established Phase 3c baseline) over time. If
top-1 stays at or near 0.121: online updates atop pretrained codebook
are safe, and the regression was specific to random init. If top-1
degrades: the issue is structural, and Option 1 (implement Hebbian) is
the correct path.

**Option 1 then** if Option 2 shows degradation, because the deep-dive
already specifies the runtime pathway that's missing.

**Option 3** is the abandonment path; only worth taking if Option 1 also
turns out to be infeasible.
