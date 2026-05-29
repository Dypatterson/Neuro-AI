# Issue tracker: Local Markdown (reports/ + STATUS.md)

Units of work for this repo do **not** live in GitHub Issues. They live as
numbered markdown reports under `reports/` and are tracked in `STATUS.md`
and per-phase checklists.

This is the convention codified in `CLAUDE.md` — every experiment that
informs a phase-graduation decision must be written up as a numbered
report, and `STATUS.md` is the bookmark that names the active phase,
the current headline metric, the last verified result, and the active
blockers.

## Conventions

- **Reports** — `reports/<NNN>_<slug>.md`, where `<NNN>` is the next
  zero-padded sequence number. Find the next number with
  `ls reports/ | grep -E '^[0-9]+_' | sort | tail -1`.
- **Status bookmark** — `STATUS.md` at the repo root. Holds the active
  phase, headline metric per design spec, recent updates (~5 lines per
  session, linking to the per-session report), and an active-blockers list.
- **Phase checklists** — `notes/emergent-codebook/phase-<N>-checklist.md`.
  Line items here are the closest analogue to "open issues" for a phase.
- **Status archive** — long-form narrative and walk-back chains live in
  `notes/status-log/<YYYY-MM>.md`, not in `STATUS.md` itself.

## When a skill says "publish to the issue tracker"

Don't call `gh issue create`. Instead:

1. **If it's an experiment that produced a result** — create
   `reports/<NNN>_<slug>.md`, then update `STATUS.md` Recent updates
   with one line linking to it.
2. **If it's a checklist item / blocker for the active phase** — add a
   line to the active phase checklist under
   `notes/emergent-codebook/phase-<N>-checklist.md` and (if it changes
   project state) add a corresponding line to the `STATUS.md` blockers
   list.
3. **If it's a PRD-style design doc for a new mechanism** — create
   `notes/emergent-codebook/<descriptive-slug>.md` (matching the style
   of the existing `phase-5-unified-design.md`,
   `consolidation-geometry-diagnostic.md`, etc.).

Always close the loop by updating `STATUS.md` **before** ending the session
(per CLAUDE.md's "walk-back is the first edit, not the last" rule).

## When a skill says "fetch the relevant ticket"

The user will normally pass either:

- A report path (`reports/040_freq_weighted_alpha_sweep.md`), or
- A checklist line reference (e.g. "the open `cap-coverage > 0.85` line in
  the phase-4 checklist"), or
- A `STATUS.md` blocker by its leading bullet.

Read the referenced file(s). Per CLAUDE.md, if you find a numbered report
that resolves an apparently-open blocker, read the report before doing
anything else — STATUS.md's open-commitments list has drifted before.

## What this means for specific skills

- **`/to-issues`** — emits new lines into the active phase checklist and/or
  new `notes/emergent-codebook/<slug>.md` design stubs, not GitHub Issues.
- **`/to-prd`** — emits a new `notes/emergent-codebook/<slug>.md` design doc.
- **`/triage`** — operates on `STATUS.md` blocker entries and
  phase-checklist lines; updates the `Status:` field per
  `triage-labels.md`.
- **`/qa`** — verifies against the report's headline metric and required
  controls, per the CLAUDE.md "What 'done' looks like" gates.
