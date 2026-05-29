# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps
those roles to the actual label strings used in this repo's tracker
(numbered reports + `STATUS.md` blockers + phase checklists).

Since this repo has no GitHub Issues labels, a "label" here is the
`Status:` field at the top of a report file, or the leading marker on a
`STATUS.md` blocker bullet / phase-checklist line item.

| Canonical role    | String used in this repo  | Meaning                                                        |
| ----------------- | ------------------------- | -------------------------------------------------------------- |
| `needs-triage`    | `open: needs evaluation`  | Maintainer needs to evaluate — anomaly observed, no plan yet   |
| `needs-info`      | `open: waiting on user`   | Blocked on a decision or input only the user can provide       |
| `ready-for-agent` | `ready: AFK-ready`        | Fully specified, an agent can pick it up with no human context |
| `ready-for-human` | `ready: needs human`      | Needs human judgement / wet-lab thinking, not an agent run     |
| `wontfix`         | `closed: wontfix`         | Investigated and explicitly rejected (with a report linked)    |

When a skill mentions a canonical role (e.g. "apply the AFK-ready triage
label"), write the corresponding string into the `Status:` line of the
report or into the leading marker of the `STATUS.md` bullet /
phase-checklist line.

## Resolved entries

For closed work that *was* actioned (not wontfix), use the existing project
vocabulary already in use across reports:

- `falsified` — mechanism was tested and disproved
- `inert` — mechanism ran but had no measurable effect
- `integrated` — mechanism was tested and adopted
- `empirically null` — mechanism produced no signal vs. control

These four come straight from `CLAUDE.md`'s knob-verdict vocabulary and
should be preferred over a generic `closed: done` when the entry refers to
an experiment.
