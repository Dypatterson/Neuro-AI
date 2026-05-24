---
date: 2026-05-24
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
status: protocol-note
session-closes: phase5-m1-codebook-prior-protocol
---

# Phase 5 M1 Codebook-Prior Protocol

This note scopes the `codebook_prior_density` probe added on the M1 branch.
It is a portability and anti-homunculus protocol for one opt-in diagnostic,
not a Phase 5 graduation claim and not a change to the default M1 gate.

## Current Scope

- Branch scope: `codex/phase5-m1-codebook-prior-protocol`, based on
  `phase5-m1-role-energy-stack`.
- Active Phase 5 headline remains the design-spec metric:
  `Delta E = E_content-prior - E_role-prior`, paired per cue, with the
  magnitude floor and controls named in
  [phase-5-unified-design.md:256-315](../../notes/emergent-codebook/phase-5-unified-design.md).
- The default M1 provenance audit gate remains count-based and is still
  degenerate on seed 17.
- Seed 17 is wiring, provenance, and degeneracy smoke only. It is not
  representative Phase 5 evidence and does not authorize n=3, n=10, or any
  path claim.
- `codebook_prior_density` is opt-in and diagnostic-only. It must not enter
  `E_total`, K-branch settling, retrieval, branch competition, replay,
  consolidation, or any runtime weighting path. Any future runtime use reopens
  the Path C/arbitration-shape audit and invalidates the present gate.

## What The Probe Measures

The count gate asks whether each stored pattern row carries non-degenerate
encoder-time role provenance. Seed 17 fails this gate because full-window rows
have symmetric role counts.

The same-role filler geometry probe asks whether row-local unbinding against
same-role fillers breaks that symmetry. Seed 17 and the synthetic fixture show
that it does not: the geometry is still full-window symmetric.

The codebook-prior probe asks a narrower diagnostic question: if the audit is
given an explicit reference codebook, do row-domain role scores become
non-degenerate when each role-unbound row is compared directly against that
codebook? The answer can be useful only after the reference codebook's identity,
lineage, and selection order are explicit.

## Current Reference Bytes

The provisional registry entry records these observed bytes:

- Basename: `phase3c_codebook_reconstruction.pt`
- SHA-256:
  `863d2ae49c8baf33e8041fe8cc0cf54939ac338f23296fb0be783db4e0412aa6`
- Size bytes: `67176273`
- Local observed path:
  `/Users/dypatterson/Desktop/Neuro-AI/reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt`

These bytes are not committed to this branch. Heavy `*.pt` artifacts over 50
MB are intentionally not committed per [CLAUDE.md](../../CLAUDE.md). The
registry entry in `config/phase5_m1_codebook_registry.json` is therefore an
identity check, not an artifact transport mechanism and not an approval mark.

The code that generated the Phase 3c codebook is checked in at
[experiments/04_phase3c_reconstruction.py](../../experiments/04_phase3c_reconstruction.py).
It starts from Phase 3a random and Hebbian codebooks, applies reconstruction
training, and saves `phase3c_codebook_reconstruction.pt`
([experiments/04_phase3c_reconstruction.py:1-6](../../experiments/04_phase3c_reconstruction.py),
[experiments/04_phase3c_reconstruction.py:475-548](../../experiments/04_phase3c_reconstruction.py)).
The generated report records Wikitext, dim 4096, seed 17, 5 epochs, and the
training/evaluation settings
([reports/phase3c_reconstruction/04_phase3c_reconstruction.md:1-24](../../reports/phase3c_reconstruction/04_phase3c_reconstruction.md)).

This establishes a plausible generation path. The Step B lineage audit below
upgrades the reference to acceptable same-lineage diagnostic use, but does not
make it lineage-independent from the seed-17 M1 smoke.

## Portability Rules

The audit records both content identity and location:

- `codebook_identity.sha256`: load-bearing content identity.
- `codebook_identity.size_bytes`: a secondary sanity check.
- `codebook_location.resolved_path`: local diagnostic only.
- `codebook_location.relpath_from_repo_root`: portable only when non-null.
- `codebook_registry.matched`: whether the SHA is known to the registry.

Warnings have the following meanings:

- `codebook_path_outside_repo`: the file was supplied from outside the current
  worktree. This is a location fact and should warn even when the codebook-prior
  mode is only being inspected as a side diagnostic.
- `codebook_registry_not_supplied`: an active geometric codebook-prior gate was
  run without a registry. The result may be useful during development but is not
  portable evidence.
- `codebook_not_in_registry`: a registry was supplied, but the loaded bytes were
  not listed by SHA.
- `codebook_registry_load_failed`: the registry could not be parsed or had no
  valid SHA entries.

For an active codebook-prior rerun, a malformed or unloadable registry should
be a failure, not only a warning. Otherwise the audit can look portable while
silently skipping the identity check. The implementation should be tightened
before the next seed-17 rerun is treated as protocol evidence.

## Anti-Homunculus Checks

**Direct circularity.** The probe must not build a reference codebook from the
same snapshot rows being audited. If the supplied codebook is derived from the
M1 audit snapshot, the result is circular and cannot be used as evidence.

**Lineage circularity.** A Phase 3c codebook generated from Wikitext and seed 17
is not automatically invalid, but it is not lineage-clean by assertion. Before
load-bearing use, the branch must document whether the reference bytes share
seed, corpus, replay traces, death survivors, or downstream selection pressure
with the M1 seed-17 snapshot. If this cannot be established from checked-in
artifacts, the load-bearing path must switch to a documented substrate-
independent regeneration or a separately generated control codebook.

**Selection circularity.** The reference SHA must be selected by a rule that
precedes any PASS result it is meant to support. "Use the codebook that made
seed 17 pass" is not a protocol. The acceptable rule in this branch is
"use the registered Phase 3c reconstruction artifact with this SHA as a
provisional diagnostic, then test controls before any escalation."

**No controller.** The codebook-prior score is a measurement of row/codebook
geometry. It must not choose between role and content branches, change a
retrieval parameter after reading an outcome, or promote/demote rows based on
an audit metric.

## Required Seed-17 Sequence Before n=3

Do not run n=3 or n=10 from the current state. The minimum sequence is:

1. Code-level portability check: active codebook-prior mode requires a readable
   registry match before it can pass as protocol evidence. Location warnings may
   remain if the artifact is still external.
2. Seed-17 portable rerun with the registered SHA and explicit evidence scope.
   Expected: same diagnostic non-degeneracy as the earlier smoke, plus registry
   metadata and any outside-repo warning.
3. Seed-17 scrambled-codebook control. The scrambled codebook must preserve the
   tensor shape and norm distribution while breaking token/codebook geometry,
   using a fixed rowwise coordinate permutation recorded in the manifest. A
   pure codebook-row permutation is not a valid control for the current
   `codebook_prior_density` metric because the metric treats the codebook as an
   unordered nearest-neighbor reference set. Expected: fail or materially
   degrade the codebook-prior gate. If it passes cleanly, n=3 is blocked and
   this protocol must be revised.
4. Seed-17 lineage-independent codebook characterization. Acceptable forms are
   a separately generated Phase 3c artifact with different seed/source lineage
   or a shape-matched FHRR random/unit control documented as non-load-bearing.
   If this is much weaker than the provisional Phase 3c SHA, n=3 is blocked
   until the lineage/selection story is resolved. If it fails, investigate
   before escalation.
5. Only after the preceding seed-17 checks have the expected outcomes should
   seeds 17/11/23 be considered for n=3 sanity. Even then, n=3 is still sanity,
   not representative Phase 5 evidence.

## Step A - Scrambled-Codebook Control (2026-05-24)

The Step A control uses
`scripts/phase5_m1_codebook_scramble.py --method rowwise_coordinate_permutation`
with seed `4242`. The helper loads the Phase 3c codebook, independently
permutes coordinates within every row, writes the generated `.pt` artifact to
`/private/tmp/`, and writes a JSON manifest recording method, seed, input/output
SHA-256 identities, file sizes, tensor shape, dtype, and row/global norm
differences.

The coordinate permutation is rowwise rather than a permutation of codebook
rows. In the current implementation, `codebook_prior_density` computes top-k
cosine density from role-unbound fillers against the supplied codebook as an
unordered reference set. Row order is never consulted, so a pure row-order
permutation would be invariant and would not falsify the metric. Rowwise
coordinate permutation preserves shape, dtype, and per-row norms while breaking
the query/codebook geometry the audit actually measures.

The scrambled output SHA must not be added to
`config/phase5_m1_codebook_registry.json`. That registry is reserved for
approved reference codebooks. Scrambled controls are intentionally not approved
references, so a final scrambled-control audit should warn with
`codebook_not_in_registry` and `codebook_path_outside_repo` when run from
`/private/tmp/`.

The pre-declared seed-17 reference baseline for this control is:

- `geometric_entropy.mean_normalized = 0.8866774142628774`
- `geometric_entropy.uniform_row_fraction = 0.02537593984962406`

Material degradation is satisfied if either the existing active audit gate
fails, or both secondary metrics co-move materially:

- `delta mean_normalized >= 0.03`
- `delta uniform_row_fraction >= 0.10`

If the audit fails but only one secondary metric moves materially while the
other stays close to baseline, report the asymmetry as a finding and stop for
review before declaring the control discriminating. If the scrambled control
does not meet the material-degradation threshold, stop; do not proceed to the
Phase 3c lineage audit or lineage-independent codebook characterization until
this protocol is revised.

## Step B - Phase 3c Lineage Audit (2026-05-24)

The Step B audit is recorded in
[reports/phase5_m1_codebook_prior_protocol_lineage/phase3c_lineage_audit.md](../../reports/phase5_m1_codebook_prior_protocol_lineage/phase3c_lineage_audit.md).

Verdict: the registered Phase 3c reconstruction codebook is acceptable for
same-lineage seed-17 diagnostic use, but not lineage-independent evidence.

Direct circularity is controlled. The audited M1 snapshot does not contain the
Phase 3c codebook tensor and the reference codebook was not built from the
1064 stored rows being audited. The snapshot rows were, however, encoded
upstream with the Phase 3c codebook, so the relationship is same-lineage.

Selection circularity is controlled for this protocol checkpoint. The Phase 3c
reconstruction artifact predates the M1 codebook-prior probe and was already
the standing Phase 4/Phase 5 substrate codebook convention. It was not selected
because it made the M1 codebook-prior audit pass.

Lineage independence remains unresolved by design. The reference and snapshot
share the seed-17/Wikitext Phase 3c -> Phase 4 -> Phase 5 lineage. Therefore
the portable seed-17 codebook-prior PASS may be described only as a
same-lineage diagnostic result. It still does not authorize n=3 or n=10.

## Step C - Random/Unit Codebook Control (2026-05-24)

The Step C control is recorded in
[reports/phase5_m1_codebook_prior_protocol_seed17_random_unit/random_unit_control_report.md](../../reports/phase5_m1_codebook_prior_protocol_seed17_random_unit/random_unit_control_report.md).

The control uses
`scripts/phase5_m1_codebook_random_control.py --method iid_unit_complex_phases`
with seed `20260524` to generate a shape-matched `[2050, 4096]`
`torch.complex64` FHRR unit-codebook in `/private/tmp/`. It uses no corpus,
vocabulary, Phase 3c training, or M1 snapshot rows, and its SHA is intentionally
not added to `config/phase5_m1_codebook_registry.json`.

The random/unit audit fails with
`geometric_mean_role_entropy_degenerate` and
`geometric_uniform_role_rows_degenerate`:

- `geometric_entropy.mean_normalized = 0.9996990434321246`
- `geometric_entropy.uniform_row_fraction = 1.0`

This matches the scrambled-control behavior and not the same-lineage Phase 3c
PASS. The result argues against the codebook-prior PASS being caused by generic
shape, dtype, row-norm, or IID FHRR unit-vector density. The non-degenerate
seed-17 result depends on alignment with the actual Phase 3c reference bytes.

Step C still does not make the result lineage-independent. From this branch
state, do not run n=3 or n=10. A future escalation should first decide whether
the same-lineage Phase 3c codebook is an acceptable explicit gate input, or
generate a separate Phase 3c-style artifact from an independent seed/source
lineage.

## Falsifiers And Retractions

- If the registered SHA cannot be regenerated or otherwise obtained from a
  documented artifact source, prior codebook-prior passes remain local smoke.
- If a scrambled codebook passes the gate, the probe is measuring generic
  codebook density or row norm artifacts, not role geometry.
- If an independent-lineage codebook fails while the provisional Phase 3c SHA
  passes, the provisional pass is selection-sensitive and not load-bearing.
- If future work routes the codebook-prior weights into runtime energy,
  retrieval, or branch selection, this protocol no longer applies.
- If a future Phase 5 codebook lineage replaces the provisional SHA, older
  PASSes must be labeled with the old SHA and cannot be pooled with the new
  lineage without an explicit bridge/control report.

## STATUS.md Discipline

No `STATUS.md` update is required for this protocol note by itself because it
does not change a blocker, claim a result, or complete a Phase 5 evidence run.
If the seed-17 protocol rerun is completed later, `STATUS.md` should receive
one short bookmark entry only. Full detail belongs in the report and in
[notes/status-log/2026-05.md](../../notes/status-log/2026-05.md).

## Out Of Scope

- Merging to `main`.
- Treating seed 17 as representative evidence.
- Running n=3 or n=10 before the seed-17 control sequence passes.
- Copying the 64 MB codebook artifact into the repo without a deliberate
  artifact policy.
- Staging ambient local files under
  `reports/phase5_log_prior_controls_local_smoke/`.
