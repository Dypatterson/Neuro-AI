# Domain Docs

How the engineering skills should consume this repo's domain documentation
when exploring the codebase.

## Layout

This is a **single-context** repo. As of 2026-05-31 there is a root
**`CONTEXT.md`** — the stable charter (the bet, the original per-phase gates,
the invariants, the current crux); read it first. There is **no** `docs/adr/`.
The canonical domain-doc surface is:

```
/
├── CONTEXT.md                          ← stable charter: bet + original gates + invariants + crux (read FIRST)
├── STATUS.md                           ← bookmark: current position (volatile)
├── CLAUDE.md                           ← working agreement + non-negotiable rules
├── docs/
│   └── PROJECT_PLAN.md                 ← top-level phase plan
└── notes/
    ├── emergent-codebook/              ← per-phase design specs + checklists
    │   ├── phase-<N>-unified-design.md ← load-bearing design spec for phase N
    │   ├── phase-<N>-checklist.md      ← phase exit criteria
    │   └── <mechanism>-diagnostic.md   ← operationalization of specific metrics
    └── notes/                          ← dated cross-paper syntheses & decisions
        └── YYYY-MM-DD-<topic>.md
```

Per `CLAUDE.md`'s session-start protocol, the order to read before any
non-trivial work is:

1. `CONTEXT.md` — the stable charter (bet + original per-phase gates + invariants + crux)
2. `STATUS.md`
3. The active phase checklist under `notes/emergent-codebook/`
4. The §"Headline metric" + §"Required controls" of the active phase's
   design doc (with explicit line-number citations)
5. `docs/PROJECT_PLAN.md` for the current phase + non-negotiable rules
6. Relevant dated notes under `notes/notes/`

## Before exploring, read these

For any skill that would normally read `CONTEXT.md` for domain vocabulary:

- Read `docs/PROJECT_PLAN.md` for the phase plan and non-negotiable design
  rules.
- Read the active phase's design spec under
  `notes/emergent-codebook/phase-<N>-*.md` for headline-metric definitions
  and operationalization.
- Skim `notes/notes/` for the most recent dated synthesis notes touching
  the topic.

For any skill that would normally read `docs/adr/` for past decisions:

- The closest equivalents are the dated synthesis notes under `notes/notes/`
  (e.g. `2026-05-09-papers-diagnostics-and-actuator-dynamics.md`) and the
  per-phase design docs under `notes/emergent-codebook/`.
- Numbered reports under `reports/` are the empirical record of what was
  tried and what the verdict was — read these before proposing to revive
  an apparently-untouched mechanism.

The root **`CONTEXT.md`** (added 2026-05-31) is the charter a skill looking for
domain vocabulary should read first; `docs/PROJECT_PLAN.md` and the per-phase
design specs remain the detailed surface. This project's canonical surface is the
one above, not the Pocock-skill default.

## Use the existing vocabulary

When naming a domain concept, use the terms already established in
`PROJECT_PLAN.md`, `notes/emergent-codebook/`, and the dated notes.
Examples of load-bearing terms in this repo:

- *substrate*, *FHRR*, *Modern Hopfield Network*, *emergent codebook*
- *cap-coverage*, *meta-stable rate*, *participation ratio*
- *headline metric* vs. *drill-down metric*
- *anti-homunculus filter* (a non-negotiable architectural constraint)
- knob verdicts: *falsified*, *inert*, *integrated*, *empirically null*

If the concept you need isn't already established, note the gap — but
don't invent a synonym for something that already has a name in the notes.

## Flag conflicts with the working agreement

`CLAUDE.md` and `notes/emergent-codebook/<phase>-unified-design.md` are
load-bearing. If your output contradicts them — for example, you propose
a supervisor module that arbitrates between subsystems (which `CLAUDE.md`
forbids under the anti-homunculus rule), or you propose a headline metric
different from the one in the design spec — surface the conflict
explicitly rather than silently overriding:

> _Contradicts the anti-homunculus rule in CLAUDE.md /
> notes/notes/2026-05-09-... — proposing anyway because…_
