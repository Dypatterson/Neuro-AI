# Report 015 — Phase 0 sweeps on the Torch hot path (Tier C item 2)

**Date:** 2026-05-13
**Experiment:** `experiments/cue_degradation_sweep_torch.py`
**Artifacts:**
- [reports/005_cue_degradation_sweep_torch.md](005_cue_degradation_sweep_torch.md) — MPS D=4096 bag
- [reports/005_cue_degradation_sweep_torch_perm.md](005_cue_degradation_sweep_torch_perm.md) — MPS D=4096 permutation
- [reports/005_cue_degradation_sweep.md](005_cue_degradation_sweep.md) — reference, CPU D=512 (unchanged)

## Motivation

PROJECT_PLAN Tier C item 2 calls for porting Phase 0 sweeps to the
Torch hot path. Seven experiments under [experiments/](../experiments/)
still import from the reference path (`energy_memory.substrate.FHRR`,
`energy_memory.memory.TemporalAssociationMemory`). This report ports
the canonical one — `cue_degradation_sweep.py` — and documents the
pattern so the remaining six can follow.

## Strategy: parallel ports, not replacements

The reference path is *the spec*. It stays. The torch port lives
alongside it as `*_torch.py`, takes a superset of the reference's
CLI flags (adds `--device` and `--encoding`), and writes to a
parallel report path (e.g. `005_cue_degradation_sweep_torch.md`).

Verification: with the same seed, dim, and `--encoding bag`, the
**summary table is identical** between the two ports.

| Cue mode             | First committed β (ref) | First committed β (torch) |
| -------------------- | ----------------------- | ------------------------- |
| exact_pair           | 4.0                     | 4.0                       |
| single_neighbor      | 4.0                     | 4.0                       |
| neighbor_plus_noise  | 4.0                     | 4.0                       |
| wrong_pair           | none                    | none                      |
| noise_only           | 4.0                     | 4.0                       |

Per-row numerical details differ at the third decimal place (e.g.
final top weight 0.414 vs. 0.426 for exact_pair β=0.5) — this is the
expected fingerprint of the two RNGs (`random.Random` vs.
`torch.Generator`). Qualitative conclusions match exactly.

## Findings from running the port

### 1. D=4096 / MPS / bag matches D=512 / CPU / bag

The summary table at the project's working scale (D=4096 on MPS) is
**identical** to the reference summary at D=512. The substrate at
D=4096 is no more committed or less committed than at D=512 on this
small-landscape sweep. This confirms the reference results scale to
the project's working dimension.

### 2. Permutation encoding requires permutation-compatible queries

Running with `--encoding permutation` at D=4096 MPS produced an
across-the-board failure mode: every cue mode shows
`ambiguous_low_recall`, `flipped`, or `wrong_or_sparse` regimes.

The cause is structural and was *not* a bug in the encoding: the
temporal queries used in this experiment are bag-style constructions
(e.g. `bundle([vectors["slip"], vectors["doctor"]])`). They match the
*format* of stored bag contexts. They do **not** match the format of
stored permutation contexts, which are bundles of `permute(neighbor,
offset)` terms — algebraically different objects.

**Implication for the encoding switch.** The `encoding="permutation"`
option in `TorchTemporalAssociationMemory` introduces a *query-side
contract*: callers must construct temporal queries that include the
same `permute(_, signed_offset)` operations the memory uses to build
its contexts. The directionality benefit shown in
[report 010](010_permutation_slots_coupled_recall.md) (4.4× over bag)
depends on that contract being honored.

Adopting `encoding="permutation"` as the default in future Phase 3
work means *also* updating every caller that constructs a temporal
query. For this Phase 0 sweep — whose temporal queries are designed
to probe the *bag-context* algebra — leaving the default as `bag`
is correct.

### 3. MPS is a wash for this sweep size, as predicted

Per [report 014](014_mps_benchmark_d4096.md), MPS speedup only kicks
in at landscape size ≥ 1024. The DISTRACTOR_STREAM has 20 atoms.
This sweep runs fine on MPS but doesn't benefit from it. The
practical recommendation: this script is a portability artifact
(can be run on either backend, same code), not an MPS-speedup
artifact. Small-landscape sweeps will get faster turnaround with
`--device cpu`.

## Pattern for porting the remaining six sweeps

The other reference-path experiments are:

- `experiments/content_vs_temporal_distractors.py`
- `experiments/coupled_settling.py`
- `experiments/dual_degradation_sweep.py`
- `experiments/joint_energy_disambiguation.py`
- `experiments/regime_sweep.py`
- `experiments/synthetic_temporal_recall.py`

Each follows the same pattern:

1. Copy to `*_torch.py`.
2. Replace `from energy_memory.substrate import FHRR, Vector` with
   `from energy_memory.substrate.torch_fhrr import TorchFHRR` (plus
   `import torch`).
3. Replace `from energy_memory.memory.TemporalAssociationMemory`
   imports with `from energy_memory.memory.torch_temporal import
   TorchTemporalAssociationMemory`.
4. Replace `build_memory` (which constructs a reference memory) with
   the inline torch equivalent: `TorchTemporalAssociationMemory(...).
   store_sequence(...)`. (`build_memory` in
   `energy_memory/experiments/synthetic_worlds.py` only works with
   the reference substrate; either inline its 4-line body or
   factor a torch-equivalent helper.)
5. Add `--device {cpu, mps}` and optionally `--encoding {bag,
   permutation}` flags.
6. Verify summary-table parity with the reference at the same seed
   and `--device cpu --dim 512 --encoding bag` before declaring the
   port done.

The reference scripts can stay as the canonical spec; the torch
ports are the fast / MPS-capable alternative. No deletion needed.

## Recommended next steps

1. **Don't port the remaining six in this session.** The port pattern
   is now established and validated. The remaining ports are
   mechanical work that earns its keep when one of those reference
   experiments actually needs to scale up — bulk-porting now would
   write a lot of code that may not get re-run. Port on demand.
2. **Move to Tier C item 3** — audit the Phase 2 matrix state and
   propose what's left to complete. The matrix is partially run
   (see [reports/phase2_full_matrix/](phase2_full_matrix/)) and item
   4 (Phase 3 codebook-growth objective) depends on it.

## Anti-homunculus check

`TorchTemporalAssociationMemory` is the same algorithm the reference
uses, with the same dynamics. The port is a backend swap, not a
mechanism change. No inspect-and-trigger logic added. Passes.
