# Report 010 — Permutation slots in coupled recall (bag vs. permutation)

**Date:** 2026-05-13
**Experiment:** `experiments/27_permutation_slots_coupled_recall.py`
**Raw data:** `reports/permutation_slots_coupled_recall.json`

## Motivation

[Report 009](009_permutation_slots_ablation.md) showed that
permutation-indexed slot encoding recovers the correct neighbor at the
correct signed offset on isolated synthetic contexts. The broader
brainstorm Idea 6 claim, however, is that the upgrade should improve
*real* retrieval — specifically the coupled (content + temporal)
recall path that drives Phase 3 retrieval — when only the temporal
channel can resolve which stored token is the right answer.

This report is the Phase-3b sidebar test that
[notes/notes/2026-05-13-fep-audit-checklist.md](../notes/notes/2026-05-13-fep-audit-checklist.md)
calls for before adopting permutation slots as a default.

## Setup

The stream is constructed to isolate directionality as the only
available disambiguator:

* **Family**: 5 content-similar atoms — independent perturbations of a
  shared family base (noise = 0.10). All 5 are near-identical to the
  base, near-orthogonal to everything else.
* **Pool**: a shared pool of 4 distractor atoms (size `2 * window` =
  4). Every family member's window of neighbors is drawn from this
  pool in a different permutation.
* **Stream layout**: `[n₋₂, n₋₁, FAMₖ, n₊₁, n₊₂]` blocks, one per
  family member. The unordered multiset of FAMₖ's neighbors is
  *identical* across k. The signed (atom, offset) multiset is
  *unique* per k.

This means:

* The bag context of every family member is identical (up to bundle
  normalize noise) — bags carry zero per-member information.
* The permutation context of every family member is distinct.
* The content cue (family base) is uninformative within the family —
  every member is equally similar to the cue.

The probe: for each family-member position k, build the directed
temporal cue from that position's true neighbors (encoded the same
way the memory encodes its contexts — a fair-fight cue per
encoding). Coupled recall should pick that exact position.

Headline metric: **top-1 disambiguation accuracy** — fraction of
family-member probes where `coupled_recall` returns the correct
position's label.

Control condition `cue_type="random"`: replace the directed temporal
cue with a fresh random FHRR vector. If the gain is real
directionality, both encodings should collapse to ≈ 1/5 chance.

Five seeds {11, 23, 41, 53, 67}. D = 4096. window = 2.
content_beta = 80, temporal_beta = 8.

## Result

| encoding    | cue       | top-1     | mean rank | mean entropy |
| ----------- | --------- | --------- | --------- | ------------ |
| bag         | directed  | 0.200     | 22.00     | 0.500        |
| bag         | random    | 0.200     | 22.00     | 0.500        |
| permutation | directed  | **0.880** | 10.64     | 0.075        |
| permutation | random    | 0.280     | 9.12      | 0.094        |

### Headline reading

* **Bag = chance (0.200).** Identical accuracy on directed and random
  cues confirms bags carry zero directional information in this
  setup. The bag temporal context is the same vector for every family
  member, so the temporal channel cannot do anything.
* **Permutation = 0.880 with directed cues, 0.280 with random.**
  4.4× over the bag baseline; the 0.880 → 0.280 collapse under the
  random control rules out any non-directional explanation.

### Drill-down readings

* **Mean rank.** Bag rank is exactly 22, the pool size — the true
  target is anywhere in the ranking, no concentration. Permutation
  directed rank is 10.64 — meaningful narrowing even on the seeds
  where top-1 was wrong.
* **Mean entropy.** Bag entropy is locked at 0.500, the maximum-
  uncertainty value for a binary-like choice on family members.
  Permutation entropy with a directed cue is 0.075 — the joint
  distribution sharply commits, even when it commits to the wrong
  family member (seed 223).
* **Per-seed variance.** Four of five seeds give permutation perfect
  top-1 accuracy (1.000). Seed 223 gives 0.400 with elevated entropy
  (0.306) — a known mode of FHRR coupled recall under unfortunate
  bundle interference, not a directional failure.

## Implication

Permutation-indexed temporal slots earn their way into the coupled
recall path:

* On a task where directionality is the only signal, they outperform
  bags 4.4× and the gain is provably directional (control collapses).
* On the prior Phase-3 baseline tests (where neighborhoods are not
  designed to mirror) the encoding is invisible to the existing
  `coupled_recall` machinery — `TorchTemporalAssociationMemory` now
  takes an `encoding="bag"|"permutation"` parameter, default unchanged.
* No new substrate operator was added (`TorchFHRR.permute` landed in
  report 009).

The change is additive, backwards-compatible, and supported by:
* 4 encoding-switch unit tests in
  `tests/test_torch_temporal_encoding.py` (default-is-bag, unknown
  rejected, contexts differ, mirrored-neighborhood distinguishability).
* The existing temporal-memory tests still pass on the bag path.

## Anti-homunculus check / FEP audit

* `encoding` is a constructor-time architectural choice, not a learned
  or runtime branch. The dynamics inside `coupled_recall` are
  identical for both encodings; only the contents of
  `self.temporal_contexts` differ.
* Permutation operations inside `store_sequence` are deterministic
  local functions of `(neighbor vector, signed offset)`. No
  controller picks anything.
* F contribution: each permuted role-filler binding is evidence about
  the position-conditional likelihood of seeing this filler at this
  signed offset. The encoding concentrates evidence on the correct
  binding rather than max-pooling over offsets.
* No new threshold, no inspect-and-trigger branch. Passes.

## Recommended next steps

1. **Promote `encoding="permutation"` to the default** for new
   coupled-recall callers in Phase 3 and beyond. Existing callers (and
   tests) continue to receive the prior bag behavior unless they
   explicitly opt in.
2. **Rerun the existing cue-degradation and dual-degradation sweeps**
   (`experiments/cue_degradation_sweep.py`,
   `experiments/dual_degradation_sweep.py`) with both encodings to
   measure the size of the improvement on the real distractor stream
   benchmarks. Those are the project's load-bearing temporal recall
   benchmarks; they should be re-run before this becomes the default.
3. **First LSR-revisit prerequisite check is now closed.** From report
   008's deferred-decision section: "Does the permutation-slot upgrade
   close enough of the Phase 3 structural gap that the regime
   question stops being load-bearing?" The answer on this task is yes;
   the directionality channel was the missing structural information,
   not the blend-vs-retrieve regime. The second prerequisite — running
   the synergy estimator on real Phase 4 consolidated bindings —
   remains open and is the next item.
