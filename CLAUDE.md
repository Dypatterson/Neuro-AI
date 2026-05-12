# Claude Working Agreement — Neuro-AI

Project context and rules for working on this codebase. Read this before any
non-trivial work.

## Project shape

This is a research project building a neuroscience-inspired cognitive
substrate (FHRR + Modern Hopfield Networks + emergent codebook + replay/
consolidation). The architectural target is a contextual-completion system,
**not a sequence-prediction system**. The user reads primary literature and
keeps detailed design notes.

The phase plan lives in [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md). The
emergent-codebook subsystem has its own multi-phase plan under
[notes/emergent-codebook/](notes/emergent-codebook/). Recent design decisions
and synthesis from cross-paper reviews live in dated notes under
[notes/notes/](notes/notes/).

## Before designing or building anything non-trivial

**Mandatory check order** — do this *before* writing a spec, before writing
code, and before recommending an approach:

1. Read [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the current phase and
   non-negotiable design rules.
2. Search [notes/](notes/) for any document that names or specifies what
   you're about to build. Phase 4 work has a Phase 4 design note; Phase 3 work
   has a Phase 3 deep-dive; etc. **A design document existing for a feature
   is the strongest signal that someone already thought hard about it.**
   `grep -rln "<keyword>" notes/` is your friend.
3. Skim relevant dated notes under [notes/notes/](notes/notes/) — these
   capture cross-paper syntheses and architectural decisions that don't
   always make it into the phase docs.
4. Check [research/](research/) for papers the project has bookmarked.
   Extracted text is in [tmp/pdf_text/](tmp/pdf_text/). The 2026-05-09
   paper synthesis note ([notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md](notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md))
   catalogues which papers are load-bearing and why.

Do not assume a design from first principles when a design document exists.
If you find a relevant note partway through implementation, stop, read it,
and re-plan.

## The anti-homunculus filter

From [notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md](notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md):

> Every proposed addition to the architecture must either be a local
> geometric dynamic or be expressible as a measurement of one — never an
> arbitration over them.

Concretely:
- No supervisor module decides which subsystem wins.
- No `if X then do Y` rule that reads a metric and triggers a response.
- "Apparent decisions" are local geometry, energy, settling, tension, or
  consolidation dynamics.

When you propose a mechanism, explicitly write out the anti-homunculus check:
who decides X, who decides Y, where the "decision" actually lives in the
dynamics. If you can't make this check pass cleanly, the mechanism is the
wrong shape.

## Headline-vs-drill-down metric structure

From the same 2026-05-09 note:

> Each phase has one headline metric that defines whether the phase crossed
> its viability threshold, plus a panel of drill-down metrics that explain
> why the headline moved in unexpected ways.

When designing an experiment:
- Name the headline metric explicitly.
- The other metrics are drill-downs that explain the headline, not
  competing definitions of success.
- Multi-metric panels with no headline let any outcome rationalize as
  success. Don't do this.

## Substrate and metrics that already exist

Don't reinvent these:

- [src/energy_memory/substrate/torch_fhrr.py](src/energy_memory/substrate/torch_fhrr.py) — FHRR operations
- [src/energy_memory/memory/torch_hopfield.py](src/energy_memory/memory/torch_hopfield.py) — Modern Hopfield retrieval
- [src/energy_memory/phase2/metrics.py](src/energy_memory/phase2/metrics.py) — cap-coverage, meta-stable rate, entropy, Wilson CIs
- [src/energy_memory/phase2/encoding.py](src/energy_memory/phase2/encoding.py) — window encoding, position vectors, decode

If the metric you want already exists, use it. If it doesn't, check whether
the project's notes have already specified its operationalization (e.g., the
[notes/emergent-codebook/consolidation-geometry-diagnostic.md](notes/emergent-codebook/consolidation-geometry-diagnostic.md)
file specifies how cap-coverage is calculated for this project).

## Non-negotiable design rules (from PROJECT_PLAN.md)

- Do not add a module that decides which subsystem should win.
- Do not solve instability with ad hoc if/then supervisory routing.
- Do not collapse the memory into a vector database plus summaries.
- Do not make the LLM the source of persistence or identity.
- Do not remove the pure-Python reference backend.
- Do not trust a new mechanism until it survives a control condition.

## Environment

- Python: `.venv/bin/python` (has torch, MPS available)
- Set `PYTHONPATH=/Users/dypatterson/Desktop/Neuro-AI/src` for imports
- Run tests: `PYTHONPATH=src .venv/bin/python -m unittest tests.<module> -v`
- Heavy artifacts (`*.pt` files >50MB) are gitignored — don't try to commit them

## What "done" looks like for an experiment

A phase result is not "done" until:
1. The headline metric is reported with confidence intervals.
2. A control condition (random codebook, shuffled tokens, no-replay, etc.)
   has been run on the same test set.
3. Drill-down metrics explain anomalies in the headline.
4. The result is written up as a markdown report under `reports/`.
5. The relevant memory or status note is updated.

## Common failure modes to watch for

- **Building from first principles when a design exists.** Always check
  notes first. (This rule exists because it has been violated.)
- **Optimizing the wrong metric.** Check whether the project has specified
  *the* metric for this phase before picking one yourself.
- **Sneaking past phase order.** If you're doing Phase N work, verify Phase
  N-1 is done. Skipping ahead is allowed but should be acknowledged.
- **Shared random state between conditions.** Save and restore RNG state
  when comparing across conditions, or use independent substrates per
  condition.
- **Reporting confidence based on a single seed.** Multi-seed is the bar,
  with bootstrap or Wilson CIs.
