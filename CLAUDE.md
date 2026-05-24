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

## Session-start protocol (read first, every session)

Before doing anything else in a session — before reading reports, before
planning, before answering a question that involves project state — read:

1. [STATUS.md](STATUS.md) at the repo root. This is the bookmark. It names
   the active phase, the current headline metric, the last verified result,
   and the active blockers.
2. The exit checklist for the active phase (e.g.
   [notes/emergent-codebook/phase-4-checklist.md](notes/emergent-codebook/phase-4-checklist.md))
   if one exists for the active phase. Every line item with a non-✅ status
   is potentially relevant to the current session.
3. **The §"Headline metric" + §"Required controls" sections of the active
   phase's design document** (e.g.
   [notes/emergent-codebook/phase-5-unified-design.md](notes/emergent-codebook/phase-5-unified-design.md)).
   Cite the exact line numbers — STATUS.md banners can drift away from
   the design spec over multiple sessions, and the design spec is the
   load-bearing source of truth for what "graduation" means.

   *Why this rule exists.* Phase 5 spent two sessions chasing a
   diagnostic instrumentation chain (K-branch state_divergence) that
   STATUS.md had drifted into describing as the headline. The actual
   design-spec headline (ΔE between role-prior and content-prior in
   [phase-5-unified-design.md:256-281](notes/emergent-codebook/phase-5-unified-design.md))
   was sitting unread the whole time. Re-skim the active design doc at
   session start; STATUS.md alone is not enough.

`STATUS.md` and the active checklist are *binding*, not advisory. If you
catch a contradiction between them and another document mid-session
— **including the active phase's design spec** — that is itself a
finding to surface, not paper over.

**Don't trust a "never built / never run" flag without grepping
`reports/` for the mechanism by name.** STATUS.md's
"Pre-phase commitments still open" list has drifted before — items
that were resolved by a numbered report were left flagged as open for
a week. Before treating any open commitment as actionable:

```bash
grep -rln '<mechanism_name_or_knob>' reports/ notes/emergent-codebook/
```

If a numbered report exists, read it before doing anything else.

*Why this rule exists.* The 2026-05-24 unconsidered-paths brainstorm
session built three downstream documents and a recommended action list
on top of a stale STATUS.md claim ("freq-weighted α never built") that
[Report 040](reports/040_freq_weighted_alpha_sweep.md) had resolved on
2026-05-17. A follow-up audit found four more knobs in the same
position — all of them already exercised in numbered reports. See
[brainstorm-workspace/2026-05-24-unconsidered-paths/wired-but-unrun-audit.md](brainstorm-workspace/2026-05-24-unconsidered-paths/wired-but-unrun-audit.md)
for the full per-knob breakdown.

If a session causes any status change (a blocker becomes done, a new
blocker surfaces, an audit fails), update `STATUS.md` and the checklist
**before** ending the session. The walk-back is the first edit, not the
last.

**STATUS.md banner discipline.** `STATUS.md` is a *bookmark*, not a log.
Each session adds one "Recent updates" entry of ~5 lines max: one-line
summary + link to the per-session report + link to the monthly archive
section. Long-form narrative, walk-back chains, and nested audits live
in the per-session report or in the monthly archive at
[notes/status-log/](notes/status-log/) (e.g.
[notes/status-log/2026-05.md](notes/status-log/2026-05.md)). Do not
inline multi-paragraph banners into `STATUS.md` itself — it grew to
87 KB / one-line-per-banner before the 2026-05-24 restructure and
became unreadable for both humans and the Read tool.

## Experiment preamble requirement

Before running any experiment that produces a numbered report or that
informs a phase-graduation decision, the agent must state, in plain text:

> **Active phase:** [N]
> **Headline metric per [spec file:line]:** [exact metric, e.g. "Δ Recall@K
> + Δ cap-coverage with active drift"]
> **Required controls per [spec file:line]:** [list]
> **Last verified result:** [report]
> **Why this experiment now:** [one sentence tying it to a STATUS.md
> blocker or checklist line item]

The `[spec file:line]` citation is **mandatory**, not optional. It must
point at the active phase's design document (typically under
[notes/emergent-codebook/](notes/emergent-codebook/)) — NOT at STATUS.md
and NOT at the most recent report's "what we're measuring" framing.
If the experiment is measuring something that does NOT appear as the
headline in the design spec, the experiment is by definition a
**drill-down**, not a graduation experiment, and the report must
explicitly label it as such.

*Why this rule exists.* Diagnostic measurements designed to investigate
a failure mode can drift into being treated as the phase's headline
over multiple sessions of debugging. The line-number citation is the
forcing function that catches the drift. If you can't find the metric
in the design spec, that's itself a finding — either the spec needs
updating (with explicit user agreement) or the experiment isn't a
graduation experiment.

If you cannot fill in any field — for example you do not know what the
headline metric should be, or you cannot identify a STATUS.md blocker
the experiment addresses — **stop** and ask the user. Do not improvise.

This preamble is a forcing function against the failure mode of "report
top1 first because exp 18 prints it first." It is cheap. The cost of
skipping it is hours of work against the wrong metric.

## Before designing or building anything non-trivial

**Mandatory check order** — do this *before* writing a spec, before writing
code, and before recommending an approach:

1. Read [STATUS.md](STATUS.md) and the active phase checklist (see
   session-start protocol above).
2. Read [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the current phase
   and non-negotiable design rules.
3. Search [notes/](notes/) for any document that names or specifies what
   you're about to build. Phase 4 work has a Phase 4 design note; Phase 3 work
   has a Phase 3 deep-dive; etc. **A design document existing for a feature
   is the strongest signal that someone already thought hard about it.**
   `grep -rln "<keyword>" notes/` is your friend.
4. Skim relevant dated notes under [notes/notes/](notes/notes/) — these
   capture cross-paper syntheses and architectural decisions that don't
   always make it into the phase docs.
5. Check [research/](research/) for papers the project has bookmarked.
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
- Set `PYTHONPATH="$(pwd)/src"` for imports (worktree-relative — works from either the `Neuro-AI-main/` worktree or the `Neuro-AI/` product-branch worktree without crossover)
- Run tests: `PYTHONPATH=src .venv/bin/python -m unittest tests.<module> -v`
- Heavy artifacts (`*.pt` files >50MB) are gitignored — don't try to commit them

## GPU performance rule of thumb (MPS / CUDA)

> **Every `.cpu()`, `float(tensor)`, `int(tensor)`, `tensor.item()`, or
> `tensor.tolist()` is a stop sign for the GPU pipeline.** Put them at
> the end of hot loops, not inside them.

PyTorch GPU work is *asynchronous*: the CPU queues commands and the GPU
runs them in the background. Any operation that converts a tensor to a
Python value forces a CPU↔GPU synchronization, which on MPS costs ~7ms
of pure waiting per sync. If you sync once per iteration of a 12-iter
settling loop, you pay ~84ms of waiting on top of ~50ms of actual work
— the waiting becomes bigger than the work.

When writing or reviewing tensor code in hot loops:
- Accumulate intermediate values as **tensors**, not Python floats.
- If a Python-level branch needs a value (e.g. early-exit convergence
  check), prefer to run all iterations on-device and replay the branch
  after a single batched `.cpu()` sync at the end. Capture intermediate
  states if the branch decision selects one of them.
- Provide a `_foo_tensor()` variant of any helper that currently returns
  a Python float, so the hot path can call the tensor version.
- `print(tensor)` and any logging that interpolates a tensor also syncs.

Reference: the 2026-05-15 `torch_hopfield.retrieve()` refactor cut
plain-retrieve time 17% and traced-retrieve 30% by deferring all
mid-loop syncs to one batched sync. Bit-identical user-facing metrics
preserved by capturing per-iteration states and selecting the converged
one retrospectively.

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
- **Equating `default = 0.0` in code with "never run."** This project's
  `ConsolidationConfig` knobs (`alpha_freq_lambda`, `coverage_lambda`,
  `retrieval_weight_epsilon`/`_tau`, `metastability_obs_rate`,
  `inhibition_gain`) default to `0.0` to preserve prior-substrate
  reproducibility, **not** because the mechanism has never been
  exercised. Every such knob has at least one numbered report that
  turned it on, measured the headline, and recorded a verdict
  (falsified, inert, integrated, or empirically null). The canonical
  query is always *"which numbered report exercises this knob?"*, never
  *"what's the default in the dataclass?"* This is the grep-by-default
  failure mode that caused the 2026-05-24 walk-back chain.
