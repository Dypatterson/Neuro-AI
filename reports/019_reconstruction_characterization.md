# Report 019 — Reconstruction characterization: atom collapse, Recall@K, and Hebbian's dominance

**Date:** 2026-05-13
**Experiment:** `experiments/33_reconstruction_characterization.py`
**Raw data:**
- `reports/reconstruction_characterization.csv` (per-window records)
- `reports/reconstruction_characterization.json` (atom profiles, aggregates)
**Follows up:** [reports/018_phase3_synergy_comparison.md](018_phase3_synergy_comparison.md) (which surfaced the Recall@1 vs. settled synergy split)

## Motivation

Report 018 left two open questions:

1. Is reconstruction's Recall@1 deficit caused by neighborhood-merging,
   such that the right answer is structurally close but not the
   argmax winner? (Test: Recall@K for K > 1 should show large gains
   over random.)
2. Is reconstruction better than Hebbian on some axis we haven't
   measured?

This report addresses both via a 4-codebook × 6-seed run at n=500
per seed with three additional measurements: **pairwise atom
similarity geometry**, **Recall@K for K ∈ {1,5,10,20,50}**, and
**target-rank distribution**.

## Headline result

### Atom-pair geometry is dramatically worse than "over-merging"

Pairwise atom cosine similarity across the entire 2050-atom vocabulary
(200k sampled pairs per codebook):

| codebook        | mean    | std    | p95    | p99    | max    |
| --------------- | ------- | ------ | ------ | ------ | ------ |
| random          | 0.0000  | 0.0110 | 0.0182 | 0.0256 | 0.0497 |
| hebbian         | **0.4094** | 0.0330 | 0.4518 | 0.4769 | 0.6628 |
| reconstruction  | **0.4102** | 0.0322 | 0.4515 | 0.4692 | 0.6613 |
| error_driven    | **0.4060** | 0.0322 | 0.4468 | 0.4635 | 0.6540 |

Random behaves exactly as theory predicts: mean ≈ 0, std ≈ 1/√(2D) ≈
0.011, max ≈ 0.05. Pristine orthogonality.

**All three learned codebooks have mean pairwise similarity ≈ 0.41**,
with very tight std (~0.033). This isn't "co-occurring atoms have
merged neighborhoods" — it's *atom collapse*: every atom in the
2050-token vocabulary is on average 41% similar to every other atom.
The differences across the three learning algorithms are ~0.004 — all
three produce essentially the same collapsed cluster.

This radically updates Report 018's "over-merging" framing. The
codebooks aren't sculpting semantic neighborhoods; they've sacrificed
the substrate's orthogonality almost entirely.

### Recall@K shows Hebbian dominating, gap widening with K

Pooled n=2191 per codebook, Wilson 95% CIs:

| codebook        | R@1                | R@5                | R@20               | R@50               |
| --------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| random          | 0.084 [0.073, 0.096]| 0.107 [0.095, 0.121]| 0.136 [0.123, 0.151]| 0.152 [0.138, 0.168]|
| **hebbian**     | 0.087 [0.076, 0.100]| **0.219 [0.202, 0.237]**| **0.343 [0.323, 0.363]**| **0.428 [0.407, 0.448]** |
| reconstruction  | 0.043 [0.036, 0.053]| 0.120 [0.108, 0.135]| 0.220 [0.203, 0.237]| 0.276 [0.258, 0.295]|
| error_driven    | 0.031 [0.024, 0.039]| 0.132 [0.119, 0.147]| 0.234 [0.217, 0.252]| 0.316 [0.297, 0.336]|

Three observations:

1. **Hebbian's R@K is 2.0–2.8× random's at every K ≥ 5.** Gap is
   already disjoint at K=5 (CI bounds don't touch) and grows with K:
   +11 pp at K=5, +21 pp at K=20, +28 pp at K=50.

2. **Reconstruction and error_driven also beat random at K ≥ 5**, but
   by less than Hebbian. At K=5, reconstruction barely separates
   from random (0.120 vs 0.107, CIs touch). Both pull ahead at K=20
   and K=50.

3. **Hebbian beats reconstruction at every K, with disjoint CIs.** No
   point in the K-sweep does reconstruction approach Hebbian's
   Recall@K performance. Report 018's "in tension" framing of
   reconstruction is no longer ambiguous: **Hebbian dominates
   reconstruction on every measurable retrieval metric** — settled
   synergy *and* Recall@K at every K tested.

### Target-rank distribution: same story

Median rank of the true target in the full decoded ranking
(~2050 candidates):

| codebook        | mean    | **median** | p25     | p75     |
| --------------- | ------- | ---------- | ------- | ------- |
| random          | 878.9   | **802.0**  | 229.0   | 1504.0  |
| hebbian         | 452.3   | **117.0**  | **6.0** | 790.0   |
| reconstruction  | 658.0   | **377.0**  | 32.0    | 1245.0  |
| error_driven    | 527.4   | **294.0**  | 23.0    | 922.0   |

Random's median rank of 802 is approximately "uniform across the
vocabulary" — the substrate has no idea where the right answer
sits. Hebbian's median of 117 (out of ~2050) means the right
answer is in the top ~6% of the ranked list on a median draw, and
Hebbian's p25 of 6 means a quarter of the time it's in the top 6.

Reconstruction's median rank (377) is better than random but
considerably worse than Hebbian. Error_driven (294) sits between
them.

## Three findings worth keeping in working memory

### 1. The atom-collapse finding is structurally important

The fact that *all three* Phase 3 learning algorithms — three
independent objective functions, three independent training paths —
converge on essentially identical pairwise-similarity profiles
(0.4094, 0.4102, 0.4060) is striking. It suggests they're not finding
different solutions; they're all driven toward the same local minimum
where atoms collapse into a tight cluster.

This is consistent with a known failure mode in similarity-learning:
if the loss rewards "similar things should have similar atoms"
without an explicit *dissimilarity* counterweight, atoms drift
toward each other and the codebook loses discriminative power. The
FEP audit framework should flag this: there's an unbalanced gradient
pulling atoms together with no countervailing gradient pushing them
apart. The 2026-05-13 brainstorm idea about
[Spisak & Friston (2025) self-orthogonalizing attractor networks](https://arxiv.org/abs/2505.22749)
is directly relevant — they prove orthogonalization is a free-energy
consequence under the right formulation. The current Phase 3
objectives don't have that consequence.

### 2. Hebbian's win is now triply verified

| metric                          | random | hebbian | rec   | err   |
| ------------------------------- | ------ | ------- | ----- | ----- |
| settled synergy (report 018)    | 0.002  | **0.252** | 0.230 | 0.222 |
| Recall@5 (this report)          | 0.107  | **0.219** | 0.120 | 0.132 |
| Recall@20                       | 0.136  | **0.343** | 0.220 | 0.234 |
| median target rank (lower better)| 802   | **117**  | 377   | 294   |

Hebbian wins on every metric with disjoint CIs vs. all alternatives.
The "first Phase 3 codebook-growth winner" claim from report 018
isn't just defensible — it's overdetermined.

### 3. Reconstruction is not a secret winner with a Recall@1 cost

Report 018 ended with a hedge: reconstruction's high settled synergy
might mean it's the better codebook for a downstream consumer that
reads structural state directly rather than top-1 argmax. This
report rules that out empirically. Reconstruction:

- Lower settled synergy than Hebbian (0.230 vs 0.252).
- Lower Recall@K than Hebbian at every K.
- Higher median target rank (worse) than Hebbian.

There is no axis on which reconstruction outperforms Hebbian in this
operating envelope. The "in tension" framing was real (reconstruction
beats random on settled synergy and Recall@K≥10) but the comparison
is between an algorithm that beats random by a lot (Hebbian) and an
algorithm that beats random by less (reconstruction). Hebbian wins
on every comparison.

That said, **error_driven outperforms reconstruction on Recall@K for
K ≥ 20** (0.234 vs 0.220 at K=20, 0.316 vs 0.276 at K=50). If the
project ever cares about deep top-K coverage specifically,
error_driven is the runner-up, not reconstruction.

## A methodological footnote

Frequency-bucket analysis returned only `q1_most_frequent` for every
test target — every single one of the 2191 test windows had a target
in the most-frequent quartile. This is because
`build_frequency_buckets` in [src/energy_memory/phase2/metrics.py](../src/energy_memory/phase2/metrics.py)
carves the bucket boundaries by *token rank*, not by *occurrence
mass*. With wikitext-2's Zipfian distribution, the top 25% of tokens
by rank are also the top ~99% of occurrences. The buckets aren't
doing what their name suggests.

This isn't a bug — the function name says "frequency buckets" and it
buckets tokens by frequency rank. But it's not useful for the
question "do learned codebooks help rare vs frequent tokens
differently" because the bucket assignment doesn't capture rarity in
a way that matters at evaluation time. The right fix: split tokens
by *occurrence-weighted* quantiles (so each bucket contains the same
number of *occurrences*, not the same number of *unique tokens*).
That's a separate small follow-up, not something this report can
resolve in place.

## Implications

### For the Phase 3 codebook-growth program

- **Hebbian is the winner**, full stop. The right next move for
  Phase 3 isn't another comparison of existing algorithms; it's
  designing the *next* algorithm — one that preserves orthogonality
  while gaining context-conditional structure.

- The atom-collapse finding is the most actionable Phase 3
  observation in any report this session. Any new codebook-learning
  objective should be evaluated against three criteria, in order:
  1. **Does it preserve orthogonality?** (atom-pair geometry mean
     should stay close to 0, not drift to 0.4+)
  2. **Does it improve settled synergy at the masked position?**
     (the structural recovery metric)
  3. **Does it improve Recall@K?** (sanity check on the
     decoding side)

  An algorithm that collapses atoms and still beats random is
  encouraging but suboptimal. An algorithm that preserves
  orthogonality *and* beats Hebbian on settled synergy and Recall@K
  is the goal.

### For the project's measurement framework

- **Recall@K is the right downstream metric**, not Recall@1. The
  K=5–20 range cleanly separates codebooks while K=1 is dominated by
  argmax noise. Future Phase 3/4 comparisons should report R@K for
  multiple K alongside settled synergy.

- **Atom-pair geometry needs to be a standard diagnostic.** It took
  this report to surface a structural pathology that all three
  Phase 3 algorithms share. It's a one-shot computation per codebook
  (no test data needed), should be cheap to add to any Phase 3
  experiment going forward.

### For the broader project

- The substrate is doing more work than the codebooks deserve. A
  codebook with mean atom similarity 0.41 *should* produce
  near-random retrieval. The fact that Hebbian gets the right answer
  into the top-20 over 34% of the time despite this codebook
  pathology is testament to the Hopfield settling's
  structure-extraction capacity. This is consistent with
  [report 011](011_synergy_probe_phase4.md)'s finding that the
  substrate preserves compositional structure through retrieval
  exactly (ratio 1.000).

- The Spisak & Friston paper (May 2025) referenced in the 2026-05-13
  brainstorm becomes directly actionable. Their attractor networks
  orthogonalize as a free-energy consequence. The next codebook-
  growth experiment should test whether their formulation (or an
  equivalent FEP-compatible counterweight) prevents the atom collapse
  observed here without sacrificing context-conditional recovery.

## Recommended next steps (Path C from the prior discussion)

1. **Lock in the new Phase 3 measurement framework.** Add
   atom-pair geometry, settled synergy at the masked position, and
   Recall@K for K ∈ {1,5,10,20,50} as standard outputs of
   `experiments/18_phase4_unified_experiment.py`. Replace the
   existing single-cell Recall@1 with this triple.
2. **Re-evaluate the per-scale codebooks** at
   `reports/phase4_per_scale/codebook_w*.pt` against the same
   framework. They predate report 018; we don't know whether they
   share the atom-collapse pathology.
3. **Fix `build_frequency_buckets`** to use occurrence-weighted
   bucketing so the rare-vs-frequent question becomes answerable.
   ~10 lines, no algorithmic change.
4. **Design the next codebook-growth algorithm.** Hebbian's
   atom-collapse pathology motivates an orthogonality-preserving
   variant — either Spisak-Friston style, or a Hebbian variant with
   an explicit dissimilarity counterweight (e.g., decorrelating
   gradient on unbound co-occurrence statistics). This is the
   substantive Phase 3 follow-up, not a measurement task.

## Anti-homunculus check / FEP audit

Pure measurement evaluator. No mechanisms added. Atom-pair similarity
is a deterministic geometric quantity. Recall@K and rank are
deterministic from `decode_position`'s scores. No inspect-and-trigger
branches. Passes trivially.
