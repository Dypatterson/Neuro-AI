# Report 009 — Permutation-indexed temporal slots vs. context bags

**Date:** 2026-05-13
**Experiment:** `experiments/26_permutation_slots_ablation.py`
**Raw data:** `reports/permutation_slots_ablation.json`

## Motivation

Brainstorm Idea 6 (2026-05-13) proposes replacing the unordered context
bag

    context(i) = bundle({ v_j : j in window, j != i })

with a directed slot encoding

    context(i) = bundle({ permute(v_j, j - i) : j in window, j != i })

The substrate-level upgrade should eliminate the two named failure modes
of bags from the 2026-05-09 paper-synthesis note: temporal inaccuracy
(simultaneous vs. sequential) and temporal fragmentation (lost
ordering). The discrete permutation variant is the lower-risk first
implementation — algebraically exact, no continuous decoder.

## Setup

- D = 4096, length 32 random FHRR atoms, window 2 (offsets ±1, ±2)
- For each anchor index, probe the recovered neighbor at each of the
  four signed offsets:
  - Bag memory: nearest-neighbor of context against vocab (no offset
    awareness — best the bag can do).
  - Permutation memory: ``permute(context, -offset)`` then
    nearest-neighbor against vocab.
- Conditions: ordered (natural sequence) and shuffled (random
  permutation of sequence positions before encoding).

## Result

| encoding    | condition | mean accuracy |
| ----------- | --------- | ------------- |
| bag         | ordered   | 0.262         |
| permutation | ordered   | **1.000**     |
| bag         | shuffled  | 0.262         |
| permutation | shuffled  | **1.000**     |

Permutation slots recover the correct neighbor at the correct signed
offset on every probe. The bag baseline sits near 1/(2·window) =
0.25 — exactly chance over the four window neighbors, since the bag
cannot route by offset.

The shuffled condition produces identical accuracy in both encodings
because the shuffle becomes the new "ordered" sequence at encoding
time. This is the expected sanity check: nothing in the permutation
algorithm depends on the underlying token labels.

## Implication

The two failure modes of bags are eliminated:

- **Temporal inaccuracy**: the offset axis distinguishes simultaneous
  (no offset binding) from sequential (signed offset binding).
- **Temporal fragmentation**: each neighbor's position is recoverable
  by name via ``permute(context, -k)``.

The improvement is large enough (4× on synthetic data) and clean
enough that the permutation-slot encoding is a defensible drop-in
replacement for context bags at any phase where the current bag
encoding is used.

## Anti-homunculus check

The encoding `bundle({ permute(v_j, j-i) })` is a deterministic local
function of (anchor index, neighbor vectors, signed offsets). The
offset is a property of the temporal axis, not a controller decision.
Query routing through `permute(context, -k)` is also local: the offset
is supplied by the caller, not inferred by an internal module.
Passes.

## FEP audit (per 2026-05-13 checklist)

- **F contribution**: each role-filler pair contributes to a likelihood
  that the temporal-context surface explains observed neighbors at
  their offsets. Bags max-pool over offsets; permutation slots
  separate offsets and so concentrate the likelihood on the correct
  binding.
- **Gradient response**: not yet integrated; the encoding alone is a
  measurement upgrade. Downstream consumers (Hopfield retrieval,
  Phase 3 codebook) still descend their existing gradients on the
  improved context vectors.
- **Branching on metrics**: none. The shift is constant from offset.
- **Threshold constants**: window size only (architectural prior).

## Recommended next steps

1. **Phase 3b sidebar experiment**: rerun the masked-token recall
   experiments with `TorchTemporalAssociationMemory` swapped for
   `PermutationSlotTemporalMemory`. The bag-context-driven coupled
   recall path in `torch_temporal.py` is the natural integration
   point — but that wires in coupled settling, so it's a larger PR
   and should be its own decision after we've seen this isolated
   result.
2. **Window-size sweep**: does the permutation gain hold at window 4
   and 8? Bags degrade gracefully as window grows; permutation slots
   may degrade differently as cross-talk between offsets accumulates.
3. **Phase 5 readiness**: the permutation operator is the same algebra
   the Kymn et al. NeurIPS 2024 paper uses for hippocampal
   compositionality. The codebase now has the operator wired in;
   Phase 5 layer-2 binding work can use it for both temporal *and*
   role-filler slots without further substrate changes.
