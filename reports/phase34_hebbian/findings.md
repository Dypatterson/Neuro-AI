# Phase 3+4 online codebook regression — final findings

Companion to `reports/phase34_stable_v2/findings.md`. The earlier
document characterized the collapse mechanism (position-bias
amplification from random init) and proposed three paths. The user
chose Option 1 — implement the Hebbian success-pathway from
`notes/emergent-codebook/phase-3-deep-dive.md:100`. This document
reports the validation result.

## Headline

The online error-driven contrastive updater (`OnlineCodebookUpdater`)
is **unsafe regardless of codebook initialization**. The Hebbian
success-pathway updater (`HebbianOnlineCodebookUpdater`) is **safe and
preserves the codebook**, but at its default success threshold of 0.5
provides essentially no continued learning signal — it fires on ~0.06%
of cues with the pretrained Phase 3c codebook.

## Experiment 24b setup

- Substrate: TorchFHRR, D=4096, MPS
- Initial codebook: `phase3c_codebook_reconstruction.pt` (pretrained,
  pair_sim ≈ 0.41)
- Window size W=2, landscape L=4096, β=10
- Single scale (W=2), 3500 cues per condition, seed=11
- Held-out test set: 300 windows
- Conditions, identical seeds and cue stream:
  - **A_baseline** — no online updates
  - **B_hebbian** — Hebbian updater, lr=0.01, success_threshold=0.5
  - **C_error_driven** — original `OnlineCodebookUpdater`, lr_pull=0.1,
    lr_push=0.05, quality_threshold=0.15

## Results

Final state at cues=3500:

| Condition | top-1 | top-K | cap_t_05 | drift | pair_sim_abs_mean | events |
|---|---:|---:|---:|---:|---:|---|
| A_baseline | 0.107 | 0.322 | 0.015 | 0.0000 | 0.4091 | — |
| B_hebbian | 0.107 | 0.322 | 0.015 | 0.0000 | 0.4091 | 2 successes |
| C_error_driven | **0.020** | **0.205** | 0.015 | 0.0042 | 0.4070 | 17 consolidations |

Trajectory of C_error_driven across cues:

| cues_seen | top-1 | top-K |
|---:|---:|---:|
| 0 | 0.107 | 0.322 |
| 500 | 0.044 | 0.317 |
| 1000 | 0.044 | 0.322 |
| 1500 | 0.049 | 0.312 |
| 2000 | 0.039 | 0.293 |
| 2500 | 0.024 | 0.293 |
| 3000 | 0.024 | 0.224 |
| 3500 | 0.020 | 0.205 |

Top-1 drops 60% within the first 500 cues (3 consolidation events) and
keeps degrading. Top-K starts preserved but degrades from 2500 cues
onward.

## What this resolves

1. **The original online error-driven updater is structurally unsafe.**
   Tested on two qualitatively different starting points (random init,
   trained codebook with pair_sim 0.41) it degrades both. The "noise
   amplification" diagnosis from `reports/phase34_stable_v2/findings.md`
   was specific to one symptom; the underlying instability is more
   general — applying contrastive pull/push to non-stationary
   slot-queries in a streaming loop is hostile to whatever structure
   the codebook already has, regardless of how much that is.

2. **Hebbian preserves the codebook safely.** Across 3500 cues, top-1,
   top-K, cap-coverage, drift, and pairwise statistics are identical
   to the no-update baseline to within measurement noise. The Hebbian
   update's "pull each cue atom toward the clean context bag" is, at
   default thresholds, conservative enough to be effectively a no-op on
   the pretrained codebook.

3. **The runtime-vs-training asymmetry in the deep-dive is validated.**
   `phase-3-deep-dive.md:140-145` says batch error-driven for codebook
   acquisition and online Hebbian for runtime reinforcement. Both
   halves of this prescription are confirmed by the experiment: batch
   error-driven (Phase 3c) gave us pair_sim=0.41 baseline performance;
   online Hebbian preserves it; online error-driven destroys it.

## What this leaves open

The Hebbian fire rate at success_threshold=0.5 was 2 events in 3500
cues. That's preservation, not continued learning. Two related
questions sit ahead but did not need to be resolved to make the
architectural call:

1. **How to make Hebbian a meaningful continued-learning signal.**
   Lower the success threshold? Use a different q proxy at runtime
   (user feedback, replay coherence, retrieval entropy)? Both candidates
   are worth investigating, but neither is a Phase 4 blocker. The
   immediate Phase 4 path uses the pretrained codebook unchanged.

2. **Where the error-driven pathway lives now.** The deep-dive says
   "batch training" — meaning `ErrorDrivenLearner` and
   `ReconstructionLearner` run offline over corpus. Those exist and
   work. Phase 4 replay's re-encoding handles codebook drift on stored
   patterns; it does not run error-driven updates. With the online
   error-driven path removed, the system's only error-driven updates
   are batch, which is what the deep-dive intends.

## Recommended actions

### Immediate (1-2 hour items)

1. **Remove `OnlineCodebookUpdater` from the integrated experiment
   path.** Specifically, `experiments/19_phase34_integrated.py` invokes
   it via `updaters_b` and `updaters_c` for conditions B and C. Either
   replace those with `HebbianOnlineCodebookUpdater` or delete the
   conditions entirely. The original phase34_medium result with
   `--random-codebook` is documenting a misapplication; the pretrained
   version of the same experiment is the architecturally-correct test.

2. **Re-run `experiments/19_phase34_integrated.py` with the pretrained
   codebook and Hebbian online**, to give Phase 4's headline experiment
   a stable codebook to operate against. The previous integrated run
   showed `phase3_phase4` produced only 2 replay candidates over 2000
   cues — likely because the codebook was being actively destabilized
   under it, so replay never saw a stable landscape to discover from.
   With Hebbian replacing error-driven, replay should see the
   pretrained landscape and produce meaningful candidates.

3. **Update the phase notes** with the runtime-vs-training resolution.
   Specifically:
   - `notes/emergent-codebook/phase-4-unified-design.md` — clarify that
     the "codebook drift" the design assumes refers to drift from
     replay-driven re-encoding plus future batch retrains, not from
     online error-driven updates.
   - Add a short note under "Phase 3 specific failure modes" in the
     deep-dive: "Online error-driven contrastive updates from any
     init: confirmed unsafe across both random init and pretrained
     init in `reports/phase34_hebbian/`. Reserve error-driven updates
     for batch passes only; runtime uses Hebbian."

### Follow-up (1-2 day items, lower priority)

4. **Sweep Hebbian success_threshold** (0.2, 0.3, 0.4) to find a regime
   where Hebbian fires often enough to provide continued learning
   without re-introducing instability. The current 0.5 threshold is
   safe but light-touch.

5. **Consider runtime-mode Hebbian q proxy** (entropy-based or
   top-2-margin-based) for when ground truth isn't available. The
   current implementation uses target-similarity, which is a
   training-time signal. The runtime equivalent would need to be a
   property of the retrieval distribution itself.

### Tier 2 work this unblocks

With the online updater question resolved, the Tier 1 list from the
earlier review can resume:

- **Phase 4 headline-with-drift experiment** (was blocked on having a
  stable codebook for replay to operate against — now unblocked).
- **5-seed HAM arithmetic validation** (independent of online learning).
- **Layer-2 admission rule tightening** (also independent).
