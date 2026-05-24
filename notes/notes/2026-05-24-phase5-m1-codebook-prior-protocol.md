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

This establishes a plausible generation path. It does not yet establish that
the bytes are lineage-independent from the seed-17 M1 smoke or that their
selection was pre-registered before observing the M1 codebook-prior pass.

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
   tensor shape and norm distribution while breaking token identity, for example
   by a fixed random row permutation recorded in the report. Expected: fail or
   materially degrade the codebook-prior gate. If it passes cleanly, n=3 is
   blocked and this protocol must be revised.
4. Seed-17 lineage-independent codebook characterization. Acceptable forms are
   a separately generated Phase 3c artifact with different seed/source lineage
   or a shape-matched FHRR random/unit control documented as non-load-bearing.
   If this is much weaker than the provisional Phase 3c SHA, n=3 is blocked
   until the lineage/selection story is resolved. If it fails, investigate
   before escalation.
5. Only after the preceding seed-17 checks have the expected outcomes should
   seeds 17/11/23 be considered for n=3 sanity. Even then, n=3 is still sanity,
   not representative Phase 5 evidence.

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
