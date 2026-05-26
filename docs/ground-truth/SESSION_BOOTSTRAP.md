# Ground Truth Bootstrap

Read this after `STATUS.md`, `CLAUDE.md`, and the active phase spec/checklist.
This pack exists so future sessions can combine the source corpus with the
project's core principles without loading every brainstorm file.

## What To Read

1. `docs/ground-truth/principles.md`
2. `docs/ground-truth/literature_matrix.md`
3. `docs/ground-truth/source_manifest.jsonl`
4. Relevant cards under `docs/ground-truth/source_cards/`

## How To Use It

- Treat `source_manifest.jsonl` as the canonical source index.
- Treat source cards as compressed routing notes, not as proof.
- Drill into the primary PDF or URL before using a source as load-bearing.
- For a new mechanism, cite at least one `source_id` plus one principle check.
- For a phase claim, ignore this pack unless the active design spec and
  required controls also support the claim.

## Corpus Status

- `stale_pdf`: local PDFs from `/Users/dypatterson/Desktop/Neuro-AI/research`,
  indexed by absolute path and SHA-256.
- `link_only`: explicit sources surfaced by brainstorm/audit sessions that are
  not local PDFs in the stale research folder.

## Useful Audit

Run the read-only audit before relying on the pack:

```bash
python3 scripts/audit_ground_truth.py
```

The audit checks JSONL validity, duplicate IDs, local PDF paths, stale PDF
coverage, and card path existence.

## Scope Boundaries

This pack does not authorize Phase 5 work while `STATUS.md` says Phase 5 is
paused. It does not replace report evidence, raw artifacts, or phase checklists.
It is a literature and principle router.
