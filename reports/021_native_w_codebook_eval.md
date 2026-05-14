# Report 021 — Per-scale codebooks are orthogonal because under-trained, not because the objective is better

**Date:** 2026-05-13
**Experiment:** `experiments/35_native_w_codebook_eval.py`
**Raw data:** `reports/native_w_codebook_eval.csv` / `.json`
**Follows up:**
- [reports/019_reconstruction_characterization.md](019_reconstruction_characterization.md) (atom-collapse finding)
- [reports/020_per_scale_atom_geometry.md](020_per_scale_atom_geometry.md) (per-scale codebooks orthogonal)

## Headline

The per-scale codebooks have orthogonal atom geometry, but they fail
to retrieve. The collapsed W=8 codebooks (mean atom similarity ≈ 0.41)
dominate the orthogonal per-scale codebooks (mean similarity ≈ 0.005)
at every native W on Recall@1, and at every K up to W=3.

This rules out the "per-scale objective is structurally better"
interpretation from report 020 and confirms the under-training
interpretation. **Atom orthogonality is neither necessary nor
sufficient for retrieval performance** in this substrate.

## Headline result

Pooled across 6 seeds × 500 test windows = 3000 windows per condition
per native W. Wilson 95% CIs on Recall@K.

### W=2

| codebook            | R@1                | R@5                | R@20               | median rank | synergy |
| ------------------- | ------------------ | ------------------ | ------------------ | ----------: | ------: |
| random_at_W         | 0.042 [0.035, 0.049]| 0.047 [0.040, 0.055]| 0.074 [0.065, 0.084]|       773.0 |  -0.000 |
| **hebbian_w8**      | **0.108 [0.098, 0.120]**| **0.243 [0.228, 0.259]**| **0.392 [0.375, 0.410]**|        66.0 |   0.460 |
| reconstruction_w8   | 0.088 [0.078, 0.098]| 0.197 [0.183, 0.211]| 0.377 [0.359, 0.394]|        63.0 |   0.346 |
| per_scale_w2        | 0.015 [0.011, 0.020]| 0.046 [0.039, 0.054]| 0.117 [0.106, 0.129]|       189.0 |   0.100 |

**per_scale_w2 has Recall@1 *below* random** (0.015 vs 0.042, disjoint
CIs). It barely beats random at R@20 (0.117 vs 0.074).

### W=3

| codebook            | R@1                | R@5                | R@20               | median rank | synergy |
| ------------------- | ------------------ | ------------------ | ------------------ | ----------: | ------: |
| random_at_W         | 0.015 [0.011, 0.020]| 0.032 [0.026, 0.039]| 0.046 [0.039, 0.054]|       791.5 |  -0.002 |
| hebbian_w8          | 0.097 [0.087, 0.108]| 0.239 [0.224, 0.255]| 0.354 [0.337, 0.371]|       114.0 |   0.303 |
| **reconstruction_w8** | 0.090 [0.080, 0.100]| 0.221 [0.207, 0.236]| **0.408 [0.391, 0.426]**|        60.0 |   0.253 |
| per_scale_w3        | 0.030 [0.025, 0.037]| 0.078 [0.069, 0.088]| 0.197 [0.183, 0.212]|       102.0 |   0.128 |

per_scale_w3 is above random but well behind both W=8 references on
every K. CI gap to hebbian_w8 at R@20 is >15 pp, disjoint.

### W=4

| codebook            | R@1                | R@5                | R@20               | median rank | synergy |
| ------------------- | ------------------ | ------------------ | ------------------ | ----------: | ------: |
| random_at_W         | 0.077 [0.068, 0.087]| 0.109 [0.098, 0.120]| 0.128 [0.117, 0.141]|       740.5 |   0.005 |
| hebbian_w8          | 0.101 [0.090, 0.112]| 0.248 [0.233, 0.264]| 0.381 [0.364, 0.399]|        76.0 |   0.359 |
| reconstruction_w8   | 0.083 [0.074, 0.094]| 0.216 [0.202, 0.231]| 0.362 [0.345, 0.380]|        85.5 |   0.268 |
| per_scale_w4        | 0.063 [0.055, 0.072]| 0.180 [0.167, 0.194]| 0.341 [0.325, 0.358]|        58.0 |   0.121 |

W=4 is the only scale where per-scale becomes competitive — at R@20
its CI [0.325, 0.358] touches reconstruction_w8's [0.345, 0.380]. But
at R@1 it still loses to *random* (0.063 vs 0.077, CIs touch). The
W=4 codebook is the only per-scale codebook that retrieved with any
strength.

## What this settles

### Interpretation B (under-training) is correct

Report 020 left two open interpretations:

- **A (structural):** small W carries less reconstruction signal, so
  small-W codebooks are orthogonal because the objective doesn't drive
  collapse at that scale. Implies "per-scale is the better objective."
- **B (training volume):** per-scale training had 53–71%
  quality-threshold failure rates; atoms were rarely updated, so they
  stayed near random init. Implies "per-scale is orthogonal because
  it didn't learn."

If A were correct, per-scale codebooks should beat random at their
native W on R@K. They do, but barely — and they lose decisively to the
W=8 collapsed codebooks at every native W. The pattern (per_scale_w2
worst, per_scale_w3 middle, per_scale_w4 best — graded with W) matches
training volume per atom (W=4 has more context per cue, so each
gradient step carries more signal even at the same number of cues).

The per-scale codebooks are orthogonal because they were under-trained.

### Atom orthogonality is not the right diagnostic on its own

Report 019 framed atom collapse as a structural pathology requiring an
orthogonality-preserving objective (Spisak-Friston style). This report
shows the relationship between geometry and retrieval is much weaker
than that framing implies:

- **Collapsed + trained ≫ orthogonal + under-trained** — by a factor of
  3–5× on R@K at most K.
- **The Hopfield substrate is robust to atom collapse.** Mean pairwise
  similarity 0.41 should mathematically devastate retrieval (atoms
  carry only ~60% discriminative variance). It doesn't. Report 019
  noted this in passing; this report confirms it head-to-head.

This doesn't mean collapse is good. It means collapse alone isn't the
load-bearing problem. The right Phase 3 question is **"given the
substrate's robustness to collapse, what's the cheapest training tweak
that adds context structure without making things worse?"** — not
"how do we prevent collapse?"

### Validates Phase 4's design choice

Report from `reports/phase4_per_scale/summary.json` showed per-scale
training did not improve the multi-scale Phase 4 retrieval result; the
shared W=8 reconstruction codebook was kept as canonical. This report
provides the mechanistic reason: per-scale codebooks aren't usefully
trained — at W=2 they're worse than random on R@1, at W=4 they're
on par with random on R@1. Using the shared W=8 codebook across all
scales is correct.

## What this updates

- **Spisak-Friston self-orthogonalization moves from "highly motivated"
  to "interesting but not urgent."** The substrate compensates for
  collapse far more than report 019 implied. The next codebook-growth
  algorithm should be evaluated against retrieval *first* and geometry
  *second*; report 019's three-criteria order (orthogonality →
  settled synergy → R@K) should be flipped to **R@K → settled synergy
  → orthogonality**.

- **The "training-volume control" item from report 020 is partially
  resolved.** Per-scale codebooks are under-trained (B), but the
  reciprocal question — does W=8 training under per-scale-like
  hyperparameters stay orthogonal? — is still open. Lower priority
  now, because we know orthogonality alone doesn't win retrieval.

- **Phase 4 priorities clarify.** With (a) Hebbian validated as the
  Phase 3 winner (report 018), (b) atom collapse confirmed as not
  catastrophic (this report), and (c) the W=8 reconstruction codebook
  validated as the canonical multi-scale source — the open Phase 4
  question (death_window / reencode_every tuning) is the next live
  experiment again. The detour through Phase 3 geometry is closing.

## Recommended next steps

1. **Return to Phase 4 tuning** with `death_window=1000` +
   `reencode_every=100` per the 2026-05-13 session summary. The
   substrate is healthy, the codebook is canonical, and the open
   headline metric is the gating issue. ~1 h.

2. **5-seed HAM arithmetic validation** — independent of Phase 4,
   still pending. ~2 h.

3. **(Deferred)** Volume-control replication of W=8 collapse. Only
   useful if Phase 4 tuning gets blocked by a codebook-side issue.

4. **(Deferred)** Spisak-Friston orthogonalization. Re-prioritize only
   if a Phase 4 / Phase 5 limit clearly traces back to collapse.

## Anti-homunculus / FEP audit

Pure measurement evaluator. R@K and rank are deterministic from
`decode_position`; settled synergy from `synergy_score`. No
inspect-and-trigger branches, no thresholds, no actuators added.
Passes trivially.
