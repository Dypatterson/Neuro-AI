# Report 020 — Per-scale codebooks do not share the atom-collapse pathology

**Date:** 2026-05-13
**Experiment:** `experiments/34_per_scale_atom_geometry.py`
**Raw data:** `reports/per_scale_atom_geometry.json`
**Follows up:** [reports/019_reconstruction_characterization.md](019_reconstruction_characterization.md)

## Motivation

Report 019 found that all three single-scale (W=8) Phase 3 codebooks —
Hebbian, reconstruction, error-driven — collapsed to mean pairwise atom
similarity ≈ 0.41, despite training under three independent objectives.
The per-scale codebooks under [reports/phase4_per_scale/](phase4_per_scale/)
(trained by [experiments/13_per_scale_codebooks.py](../experiments/13_per_scale_codebooks.py)
for W ∈ {2, 3, 4}) predate report 019 — the question was open whether
they share the same pathology.

This report answers that question with one diagnostic: sampled pairwise
cosine similarity across each codebook's decodable atoms
(200k pairs, seed 11, n_atoms = 2048, dim = 4096).

## Headline result

Per-scale codebooks **do not collapse**. They sit ~0.005–0.008 above
the random baseline's near-zero mean — orders of magnitude away from the
0.41 single-scale collapse.

| codebook              | mean    | std    | p50     | p95     | p99     | max     |
| --------------------- | ------- | ------ | ------- | ------- | ------- | ------- |
| random_baseline       | -0.0000 | 0.0110 |  0.0000 |  0.0182 |  0.0256 |  0.0497 |
| hebbian_w8            | **0.4094** | 0.0330 |  0.4133 |  0.4518 |  0.4769 |  0.6628 |
| reconstruction_w8     | **0.4102** | 0.0322 |  0.4145 |  0.4515 |  0.4692 |  0.6613 |
| error_driven_w8       | **0.4060** | 0.0322 |  0.4107 |  0.4468 |  0.4635 |  0.6540 |
| per_scale_w2          |  0.0076 | 0.0161 |  0.0058 |  0.0365 |  0.0613 |  0.1240 |
| per_scale_w3          |  0.0075 | 0.0172 |  0.0054 |  0.0357 |  0.0701 |  0.1651 |
| per_scale_w4          |  0.0054 | 0.0163 |  0.0038 |  0.0299 |  0.0598 |  0.2745 |

The per-scale codebooks' p95/p99/max are modestly elevated over random
(max 0.12–0.27 vs random's 0.05), so there *is* some learned structure
showing up in the tail — but the bulk of the distribution remains
orthogonal.

## Two interpretations

The naive reading is "per-scale training preserves orthogonality, single-
scale collapses it — therefore per-scale is the right objective." That
interpretation should be resisted until R@K and settled synergy are
measured for these codebooks. There are two qualitatively different
mechanisms that could produce this geometry:

**Interpretation A: structural — small W carries less reconstruction
signal.** At W=2, the binding `pos_0 ⊛ tok_0 + pos_1 ⊛ tok_1` produces a
weaker "neighborhood pull" gradient per atom than the W=8 reconstruction
loss does. The codebook drifts less from random init *and* fails to
acquire context-conditional structure. Orthogonal codebook, weak
retriever.

**Interpretation B: failure-rate driven — atoms were rarely updated.**
The per-scale `summary.json` records final failure rates of
0.53/0.62/0.71 for W=2/3/4. Over half the training windows didn't pass
the quality threshold, so atoms in those windows got no gradient. Atoms
drift far less when training fails most of the time. Orthogonal because
under-trained, not because the objective is structurally better.

Same training class (`ReconstructionLearner`) at smaller W producing
geometry this different is informative *only* paired with a retrieval
benchmark. Geometry alone can't separate "orthogonality-preserving by
virtue of structure" from "orthogonality-preserving by virtue of doing
very little."

## What this updates

- **The atom-collapse finding is plausibly W- or training-volume-
  dependent, not objective-shape-dependent.** Three objectives at W=8
  all collapse; the same reconstruction objective at W ∈ {2,3,4}
  doesn't. A clean experiment to pin down the cause would re-run
  single-scale W=8 reconstruction with the same `quality_threshold` and
  number of consolidations as per-scale W=4, see whether the W=8
  geometry stays orthogonal under low effective training volume. If it
  does, the FEP-audit story about "unbalanced gradient pulling atoms
  together" is too simple — collapse is dosage-dependent.

- **Phase 4's multi-scale retrieval result becomes more interesting.**
  Report from `reports/phase4_per_scale/summary.json` notes per-scale
  training did *not* outperform the shared W=8 codebook (memory recall:
  the validated multi-scale architecture uses the *shared* W=8
  reconstruction codebook). So Phase 4 retrieval is winning with the
  *collapsed* W=8 codebook even though the *orthogonal* per-scale
  codebooks exist. The substrate's Hopfield settling does a lot of work
  on top of catastrophically collapsed atoms.

- **The Spisak & Friston self-orthogonalization direction is still
  motivated**, but the case for it shifts from "all three Phase 3
  objectives collapse, so something fundamental is wrong" to "the
  Phase 3 objectives collapse when trained to convergence, and we need
  a counter-gradient that prevents this without sacrificing the
  retrieval gain we get with the collapsed codebooks." Quantifying
  that retrieval gain explicitly is the prerequisite.

## Recommended next steps

1. **Run experiment 33 on the per-scale codebooks.** R@K + settled
   synergy + target rank, same harness, same test set. This is the
   missing measurement: without it we don't know if orthogonal per-
   scale codebooks are useful retrievers or just under-trained
   noise. ~30 min on MPS.

2. **Replicate the report-019 atom-collapse finding with a training-
   volume control.** Train W=8 reconstruction under per-scale-like
   hyperparameters (high quality threshold, fewer epochs) and re-
   measure atom-pair geometry. If geometry stays orthogonal, collapse
   is dosage-driven; if it still collapses, the W=8 binding objective
   is the cause. ~1 h.

3. **Only then design the orthogonality-preserving objective**
   (Spisak-Friston style or Hebbian + dissimilarity counterweight).
   The motivation is much sharper once we know whether collapse is an
   objective-shape problem or a training-volume problem.

## Anti-homunculus / FEP audit

Pure measurement. Atom-pair cosine similarity is a deterministic
geometric quantity. No mechanisms added, no thresholds, no inspect-and-
trigger branches. Passes trivially.
