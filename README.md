# Neuro-AI

A research project building a **neuroscience-inspired cognitive substrate** —
FHRR hypervectors + Modern Hopfield associative memory + an emergent codebook +
replay/consolidation. The architectural target is **contextual completion**
("what does this remind me of, and what fills this gap?"), **not** sequence
prediction ("what token comes next?"). It is explicitly *not* a vector database
plus summaries.

This repo is a lab notebook as much as a codebase: progress is recorded as
~140 numbered experiment reports under [`reports/`](reports/), each with a
headline metric, control conditions, and multi-seed confidence intervals. The
culture is **anti-overclaim** — results get walked back when a control overturns
them, and the walk-backs are part of the record.

> **Start here:** [`STATUS.md`](STATUS.md) is the live bookmark (where the work
> is *right now*). [`CONTEXT.md`](CONTEXT.md) and [`CONTEXT-B.md`](CONTEXT-B.md)
> are the stable charters. [`CLAUDE.md`](CLAUDE.md) is the working agreement.

---

## The core idea

Modern LLMs are powerful but (a) energy-hungry, (b) **frozen** after training —
they can't learn continually, forget nothing new, and need retraining to acquire
a skill, and (c) data-hungry. The brain is the opposite: it starts not-smart,
learns **continually**, from **little** data, on ~20W, with **no homunculus**
arbitrating its parts.

The substrate is built from brain-analogous primitives:

- **FHRR** (Fourier Holographic Reduced Representations) — bind/unbind/bundle of
  random unit-complex hypervectors, for role–filler structure at D=4096.
- **Modern Hopfield retrieval** — iterative softmax settling into attractors,
  with energy/entropy diagnostics (retrieval is *generative*, not lookup).
- **Emergent codebook** — discrete atoms grown from experience rather than
  hand-specified.
- **Replay / consolidation** — offline re-presentation and restructuring of
  stored experience.

A guiding invariant runs through all of it — the **anti-homunculus filter**:
every mechanism must be a *local geometric/energy dynamic or a measurement of
one*, never a supervisor that reads a metric and picks a winner.

## Two bets

As of 2026-06-05 the repo runs **two parallel bets** with **different mechanism
rules but the same discipline**:

- **Bet A** — charter [`CONTEXT.md`](CONTEXT.md). *Biology **is** the answer:*
  no backprop, no global-computation-as-mechanism, local-only. Global methods
  (SVD/NMF/k-means) are diagnostic *flashlights*, never the mechanism.
- **Bet B** — charter [`CONTEXT-B.md`](CONTEXT-B.md). *Biology is the
  **spotlight**, not the box:* backprop / global-as-mechanism / multi-layer are
  **allowed** (the one still-fenced shortcut is a one-shot closed-form
  SVD/eig). Forked because Bet A's mechanism bans kept getting silently
  re-imported as binding after they were explicitly lifted.

The **discipline** is identical and binding for both: a single named headline
gate per experiment, multi-seed with CIs, mandatory control conditions, the
anti-homunculus filter, and *grep `reports/` before re-running any mechanism by
name*.

## What's been established (honest tiers)

Per the post-139 reckoning in
[`notes/RETROSPECTIVE-two-bets-2026-06-06.md`](notes/RETROSPECTIVE-two-bets-2026-06-06.md):

**Durable:**
- **The Consolidation-write FLOOR is cleared (Bet A, Reports 055–058).** A
  surgical heteroassociative write + L2 decorrelator recalls role→target
  bindings role-selectively at the information ceiling, multi-seed, integrated
  into the substrate bit-identically. This is *contextual completion* working —
  the floor, not the ceiling.
- **A real local-vs-global bound (Bet A, Reports 121–132).** Paradigmatic
  (substitutional, *king/queen*) structure lives in the **subdominant modes** of
  the co-occurrence operator: reachable by *global* computation (SVD/NMF/k-means,
  +0.11–0.25) but by **no local single-projection dynamic over a flat code**
  (route-invariant across 7 operators; confirmed *capability*-level by an
  independent behavioral probe). The single-layer/local/no-backprop writer
  family is genuinely exhausted.
- **A reusable experimental discipline + a false-positive catalogue** — the
  named metric traps (gauge leak, glob-double-subtraction, contraction artifact,
  low-dim cosine inflation, incompetent-control, para-set hubness) that have
  repeatedly caught this project's own over-claims.

**Suggestive but confounded:**
- "Brain-distinctive mechanisms reduce to simpler ones" — observed repeatedly,
  but **confounded with task selection**: the toys were solvable by simple
  means, so this is as much about the toys as about biology.

**Unconfirmed (the open question):**
- The central thesis — *does an emergent consolidation step manufacture
  transferable structure that simpler methods (plain replay, a soft weight
  anchor, a one-shot factorization) cannot?* — is **untested in a discriminating
  regime where the simple methods fail.** Bet B's continual compounding-transfer
  gate was *cleared* by **soft weight-anchor + replay** (Report 139), but a
  matched **EWC control overturned** the brain-distinctive (multi-timescale)
  claim → the result is **recombinant, not brain-distinctive**. Building the
  discriminating/compositional regime is the live next step (see CONTEXT-B §6/§8).

## Repository map

| Path | What's there |
|---|---|
| [`src/energy_memory/`](src/energy_memory/) | The substrate: [`substrate/torch_fhrr.py`](src/energy_memory/substrate/torch_fhrr.py) (FHRR), [`memory/torch_hopfield.py`](src/energy_memory/memory/torch_hopfield.py) (Modern Hopfield), `phase2/` (encoding + metrics: cap-coverage, entropy, Wilson CIs), `phase3/` (consolidation/regime diagnostics). A **pure-Python reference backend** is kept alongside the Torch/MPS path by design. |
| [`experiments/`](experiments/) | Numbered, runnable experiment scripts (each pairs with a report). |
| [`reports/`](reports/) | ~140 numbered narrative reports (`NNN_*.md`). Generated data (`*.json/csv/log/err/out`, `shards/`) is gitignored — the markdown is the durable record. |
| [`notes/emergent-codebook/`](notes/emergent-codebook/) | Per-capability design docs (the `phase-N-*` files) — the load-bearing specs for headline metrics + required controls. |
| [`notes/notes/`](notes/notes/) | Dated cross-paper syntheses and architectural decisions. |
| [`notes/status-log/`](notes/status-log/) | Monthly archive of long-form status narrative. |
| [`docs/`](docs/) | [`PROJECT_PLAN.md`](docs/PROJECT_PLAN.md), `ground-truth/` (literature matrix, principles, source manifest), `agents/` (skill configs). |
| [`scripts/`](scripts/), [`tests/`](tests/) | Helper scripts; the `unittest` suite. |
| [`brainstorm-workspace/`](brainstorm-workspace/), [`notebooks/`](notebooks/) | Dated ideation sessions; Colab notebooks. |

## Running it

```bash
# Environment: a local .venv has torch (MPS available); src is on the path.
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests        # full suite

# A single test module:
PYTHONPATH=src .venv/bin/python -m unittest tests.<module> -v

# Experiments are numbered and self-contained, e.g.:
PYTHONPATH=src .venv/bin/python experiments/<NNN>_<name>.py
```

The pure-Python reference backend runs without torch; the Torch/MPS path is for
larger validation runs. Heavy artifacts (`*.pt` >50MB) and generated experiment
data are gitignored. For large compute, some runs use a sharded Colab workflow
(`--device cuda`).

## Reading order

1. [`STATUS.md`](STATUS.md) — the volatile bookmark: active front, headline
   metric, last verified result, blockers.
2. [`CONTEXT.md`](CONTEXT.md) (Bet A) / [`CONTEXT-B.md`](CONTEXT-B.md) (Bet B) —
   the stable charters: the bet, the capability map + gates, the invariants.
3. [`notes/RETROSPECTIVE-two-bets-2026-06-06.md`](notes/RETROSPECTIVE-two-bets-2026-06-06.md)
   — the honest big-picture reckoning across both bets.
4. The active capability's design doc under
   [`notes/emergent-codebook/`](notes/emergent-codebook/) — the source of truth
   for what "graduation" means.
5. [`CLAUDE.md`](CLAUDE.md) — the working agreement (experiment preamble,
   headline-vs-drill-down, controls, the anti-homunculus filter).
