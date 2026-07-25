# Claude Working Agreement — Neuro-AI

Rules for working on this codebase. Short on purpose.

*Rewritten 2026-07-25. The previous version imposed ~30 separately-checkable
obligations and ~122 KB of mandatory session reading. An audit of the nine false
positives this project actually caught (the retrospective's false-positive
catalogue) found that **every one** was caught by a **measurement** rule —
multi-seed, a competent matched control, adversarial verification, log-scale for
heavy tails. **None** was caught by a read order, a `file:line` citation, a byte
budget, or a triage label. This file now holds the measurement rules and little
else. What was cut is in `archive/`, not deleted.*

## Session start — two reads

1. **[STATUS.md](STATUS.md)** — the bookmark: active front, headline metric, last
   verified result, blockers.
2. **[CONTEXT-B.md](CONTEXT-B.md)** — the binding charter for the live line
   (Bet B). Read
   [notes/RETROSPECTIVE-two-bets-2026-06-06.md](notes/RETROSPECTIVE-two-bets-2026-06-06.md)
   once for orientation if you haven't.

**Bet A is paused.** Its charter (`CONTEXT.md`) and phase notes are history, not
law. Do not import its mechanism bans (no-backprop, local-only, no-global) into
Bet-B work — that re-import happened three times in one session and is why
`CONTEXT-B.md` exists. Bet A's 121–132 local-vs-global bound stands, unfalsified.

If STATUS.md contradicts a report, that contradiction is itself a finding.
Surface it; don't paper over it.

## The measurement rules (binding — these are the ones that caught real errors)

1. **Multi-seed ≥8, with CIs.** Single-seed and n=2 smokes have produced false
   PASSes here (137). Use bootstrap CIs, and the **log** scale for ratio metrics
   (speedups are heavy-tailed — 138).
2. **A competent, matched control on the same test set.** The single
   highest-yield rule in this repo: it deflated k-WTA→global k-means (127) and
   Benna-Fusi→plain-L2-anchor (139). A control weaker than the mechanism proves
   nothing.
3. **Adversarially verify before banking a positive.** Try to refute your own
   result first. Every "first real win" here that skipped this was later deflated.
4. **One named headline gate.** Everything else is explicitly labelled a
   drill-down. Multi-metric panels with no headline let any outcome rationalize
   as success.
5. **Grep before re-running a mechanism by name.** Read
   [reports/INDEX.md](reports/INDEX.md) first, then grep **whole-repo** if absent.
   Scoped greps have returned false "never built" answers — that is how
   Benna-Fusi got rebuilt from scratch when a tested implementation existed.
6. **Anti-homunculus — enforced as a test, not a docstring.** No supervisor that
   reads a metric and branches (`if metric > threshold: do Y`). Apparent
   decisions must be local geometry, energy, or settling dynamics. Three modules
   in `legacy/` claim compliance in a docstring while violating it in code —
   which is why a prose claim is not enough. Assert it.

## Experiment preamble

Before an experiment that produces a numbered report or informs a gate, state:

> **Active capability:** …
> **Headline metric** per `<file> §<section-header>`: …
> **Required controls** per `<file> §<section-header>`: …
> **Last verified result:** [report]
> **Why now:** one sentence tying it to a STATUS.md blocker

**Cite by section header, not line number.** Line ranges rot — three root-charter
citations now point at the wrong text, including one inside the rule that existed
to prevent that drift.

If the metric isn't the design spec's headline, the experiment is a
**drill-down** and the report must say so. If you can't fill a field, ask.

## Runs must be auditable

Every experiment writes a provenance envelope (git SHA, argv, seeds, config,
declared controls) into its output JSON — see
[src/energy_memory/betb/runner.py](src/energy_memory/betb/runner.py). A run that
omits a declared control fails loudly. This replaces attestation with
enforcement.

Merged headline JSON is committed (`reports/**/headline*.json` is un-ignored);
shard intermediates stay ignored.

## What "done" means for an experiment

1. Headline reported with CIs. 2. A competent control on the same test set.
3. Drill-downs explain anomalies. 4. Written up under `reports/`.
5. `STATUS.md` updated — including the walk-back, if the session walked one back.

## Non-negotiable design rules

- No module that decides which subsystem wins.
- No ad hoc if/then supervisory routing to fix instability.
- Don't collapse the memory into a vector DB plus summaries.
- Don't make an LLM the source of persistence or identity.
- Keep the pure-Python reference backend.
- Don't trust a new mechanism until it survives a control condition.

## Don't reinvent these

- [src/energy_memory/betb/continual.py](src/energy_memory/betb/continual.py) —
  the continual-learning harness. **The task family is an injected parameter**;
  add a task family, don't fork the harness.
- [src/energy_memory/betb/tasks.py](src/energy_memory/betb/tasks.py) — task
  families, including the compositional regime.
- [src/energy_memory/substrate/torch_fhrr.py](src/energy_memory/substrate/torch_fhrr.py) — FHRR ops
- [src/energy_memory/memory/torch_hopfield.py](src/energy_memory/memory/torch_hopfield.py) — Hopfield retrieval
- [src/energy_memory/phase2/metrics.py](src/energy_memory/phase2/metrics.py) — cap-coverage, Wilson CIs
- `src/energy_memory/legacy/` — the Bet-A arc (phase3/34/4/5), frozen 2026-06-06.
  `legacy/phase4/consolidation.py` holds a tested multi-timescale consolidator.

## Environment

- Python: `.venv/bin/python`. Set `PYTHONPATH=src`.
- Tests: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t tests`
- Fast smoke: every `betb` experiment accepts `--tiny`.
- `*.pt` >50MB are gitignored.

## GPU rule

No `.item()`, `.cpu()`, `float(tensor)`, or tensor-interpolating `print` inside
hot loops — each forces a CPU↔GPU sync. Accumulate as tensors; sync once at the
end. (True on MPS and CUDA alike.)

## Failure modes that have actually bitten

- **Building from first principles when a design exists.** Check `reports/INDEX.md`.
- **`default = 0.0` ≠ "never run."** Knobs default to 0.0 to preserve
  reproducibility. Ask "which report exercises this knob?", never "what's the
  default?"
- **Testing in isolation what is really an interaction.** A mechanism can null
  alone and work coupled. Combination experiments are first-class.
- **Shared RNG state between conditions.** Independent substrates per condition.
- **The task-selection confound (standing).** If the task is already solvable by
  the simple method, the fancy mechanism *cannot* show a necessary advantage —
  the null is expected by construction and says nothing about the mechanism. Ask
  of every experiment: **does the simple baseline fail here?** If not, you are
  not testing what you think you are testing.
