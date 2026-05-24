# Phase 5 M1 Codebook-Prior Protocol - Phase 3c Lineage Audit

Date: 2026-05-24

Branch: `codex/phase5-m1-codebook-prior-protocol`

## Scope

This is the Step B audit for the opt-in M1 `codebook_prior_density`
diagnostic. It asks whether the registered Phase 3c reconstruction codebook can
be treated as a legitimate same-lineage reference for the seed-17 protocol run.
It does not make the codebook lineage-independent, does not change the default
count gate, and does not authorize n=3 or n=10.

Reference bytes:

- Path observed locally:
  `/Users/dypatterson/Desktop/Neuro-AI/reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt`
- SHA-256:
  `863d2ae49c8baf33e8041fe8cc0cf54939ac338f23296fb0be783db4e0412aa6`
- Size bytes: `67176273`
- Shape/dtype from the portable M1 audit: `[2050, 4096]`,
  `torch.complex64`

Audited M1 snapshot:

- Path:
  `reports/phase5_m1_provenance_seed17/snapshots/phase3_phase4_w4_step1800.pt`
- Snapshot metadata: seed `17`, scale `4`, cues seen `1800`, rows `1064`
- Snapshot tensors: `patterns` shape `[1064, 4096]`, `positions` shape
  `[4, 4096]`

## Verdict

Step B passes only as a same-lineage reference audit.

The Phase 3c reconstruction codebook is an acceptable registered reference for
the seed-17 same-lineage diagnostic: it is the pre-existing Phase 4/5 substrate
codebook, not a codebook constructed from the M1 audit rows after seeing the M1
result. The selection-circularity risk is therefore controlled enough for the
single-seed protocol checkpoint.

The reference is not lineage-independent. It shares the same corpus family,
seed-17 substrate lineage, and downstream Phase 3c -> Phase 4 -> Phase 5 path
as the audited snapshot. The seed-17 codebook-prior PASS remains a
wiring/provenance diagnostic, not representative Phase 5 evidence.

Proceed to Step C: a lineage-independent or non-load-bearing random/unit
codebook characterization. Do not run n=3 or n=10 from Step B alone.

## Direct Circularity

Finding: PASS.

The supplied codebook is not built from the M1 audit rows. The audited snapshot
contains stored substrate patterns, positions, consolidation metadata,
per-pattern encoder terms, and row labels. It does not contain a
`phase3c_codebook_reconstruction.pt` tensor or a 2050-row codebook tensor.

The direct connection is upstream encoding: the Phase 3+4 integrated experiment
loads a codebook path, encodes source windows with
`encode_window_with_provenance`, and later saves memory patterns plus encoder
provenance into the substrate snapshot. That is same-lineage use, not a
post-hoc codebook derived from the rows being audited.

Evidence:

- `reports/phase5_m1_provenance_seed17/phase34_results.json` records
  `codebook_path` as the Phase 3c reconstruction artifact and seed `17`.
- `experiments/19_phase34_integrated.py` encodes initial rows from the supplied
  codebook with `encode_window_with_provenance` and saves snapshots containing
  memory patterns, positions, and encoder-term metadata.
- `src/energy_memory/phase2/encoding.py` defines encoder provenance as the
  structural `(position_index, token_id)` tuples before bundling, not as a
  post-hoc unbinding result.

## Lineage Circularity

Finding: ACCEPTED WITH LIMIT.

The Phase 3c codebook and the M1 seed-17 snapshot are in the same lineage.
Phase 3c reports Wikitext, dim `4096`, seed `17`, five reconstruction epochs,
and vocab size `2050`. The M1 provenance snapshot report records Wikitext,
seed `17`, and the same local Phase 3c codebook path. This means the
codebook-prior PASS can show that the explicit same-lineage Phase 3c reference
breaks row-role symmetry in the audit metric. It cannot show that an
independent codebook would do so.

This limitation is expected and is why Step C remains required.

Evidence:

- `reports/phase3c_reconstruction/04_phase3c_reconstruction.md` records the
  Phase 3c training config: Wikitext, dim `4096`, seed `17`, five epochs, and
  vocab size `2050`.
- `experiments/04_phase3c_reconstruction.py` loads Phase 3a random/Hebbian
  artifacts, trains `ReconstructionLearner`, and saves
  `phase3c_codebook_reconstruction.pt`.
- `reports/039_phase3_codebook_comparison_integrity.md` records that
  `phase3c_codebook_reconstruction.pt` is genuinely unique among the Phase
  3-era codebook tensors and was the Phase 4 input.

## Selection Circularity

Finding: PASS FOR SAME-LINEAGE PROTOCOL USE.

The Phase 3c reconstruction artifact was not introduced because it made the M1
codebook-prior probe pass. It predates the M1 codebook-prior branch and was
already the downstream substrate codebook for Phase 4/Phase 5 work.

Evidence:

- Git history places the Phase 3c report in commit `2cc362c` and the Phase 3
  codebook-integrity disposition in commit `af08f9e`.
- Git history places M1 codebook-prior implementation/protocol commits later:
  `7ab68bf` for the geometric probe and `f12325b` for this protocol branch.
- `notes/emergent-codebook/phase-5-unified-design.md` states that Phase 5 wraps
  the Phase 4 substrate and leaves the codebook unchanged as the Phase 3c
  reconstruction codebook, or whatever Phase 4 later produces.
- Phase 5 Colab notebooks stage
  `MyDrive/neuro-ai/phase3c_codebook_reconstruction.pt` before running the
  headline code path, showing this was the standing substrate artifact
  convention rather than a codebook selected by the M1 probe.

## Portability Limit

The registered SHA and size make the rerun identity-checkable, but they do not
transport the 64 MB `.pt` artifact. Current evidence still depends on the local
external artifact path. This is acceptable for the Step B same-lineage audit
only. A load-bearing future gate needs either a documented artifact source or a
portable regeneration/manifest convention.

## Next Step

Continue to Step C before any cross-seed run:

- Characterize a lineage-independent codebook if available, or a shape-matched
  FHRR random/unit control as explicitly non-load-bearing.
- Compare it against the portable same-lineage Phase 3c PASS and the Step A
  scrambled FAIL.
- If the independent control is much weaker or fails, keep the Phase 3c
  codebook-prior PASS labeled same-lineage only and do not escalate to n=3.
