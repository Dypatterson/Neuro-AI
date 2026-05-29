# 2026-05-26 — Architecture-review grilling: decisions

Outcomes from the `/mp-improve-codebase-architecture` grilling session on the
six candidates surfaced by the architecture review report
(`/var/folders/h8/vr5w2f4s7w57nfv2wx78d38c0000gn/T/architecture-review-20260526-192509.html`).

This note records *decisions*, not implementation. Decisions are written in the
order the grilling reached them, so the early ones (e.g. naming) are visible
to the later ones (e.g. cross-phase seams that need to reference the new name).

Vocabulary follows
[notes/agents/LANGUAGE.md](../../.claude/skills/mp-improve-codebase-architecture/LANGUAGE.md):
module, interface, implementation, depth, seam, adapter, leverage, locality.
Domain terms follow `docs/PROJECT_PLAN.md` and prior notes in this folder.

## Card 5 — `phase34/` identity

**Decision.** `src/energy_memory/phase34/` is a **long-lived subsystem**, not
a transient bridge. Rename it to **`src/energy_memory/online_consolidation/`**.

**Why.** The module's three exports (`OnlineCodebookUpdater`, `HebbianOnline`,
`StableCodebook`) describe a concept — *consolidation that runs continuously
as replay observations arrive*, as opposed to a separate batch pass over a
stored buffer. That concept will outlive the Phase 3 / Phase 4 bookmark
distinction and deserves a concept name, not a phase-pair number.

**Project glossary entry (new).** *Online consolidation* — the codebook-update
behaviour that fires as replay events arrive, rather than as a batch pass.
Lives under `src/energy_memory/online_consolidation/`. Distinct from Phase 3
codebook *learning* (which builds the codebook from scratch) and from Phase 4
*replay scheduling* (which decides when to consolidate).

**What changes (later — not in this session).**
- Move `src/energy_memory/phase34/` → `src/energy_memory/online_consolidation/`.
- Update all `from energy_memory.phase34 import …` to the new path.
- Update STATUS.md / phase-checklist references that point at `phase34/`.
- Drop the term *online consolidation* into `docs/PROJECT_PLAN.md` once it's
  used in user-facing prose more than once.

**ADR-shaped note.** This decision is recorded so future architecture-review
runs don't re-suggest "decide what phase34 is" — the name and shape are now
fixed.

## Card 4 — Anti-homunculus rule enforcement

**Decision.** Add a **lint/grep rule** as the structural backstop. **Keep** the
existing CLAUDE.md prose + reviewer-PASS process + monkey-patching tests.
**Do not** introduce a typed `LocalDynamicsModule` base class.

**Why not the typed seam.** A runtime-checking base class is itself reading
mechanism state to make decisions about mechanism shape — a homunculus in the
mirror. A shape-constraining base class (one whose interface simply can't
accept cross-subsystem reads at the type level) is honest but heavyweight,
requires every C.2.x dynamic to be retrofitted into a class hierarchy, and
adds an inheritance ceremony that doesn't pay back at this project's size.
The reviewer-PASS process has held for C.2.1–C.2.5; the next move is to make
that process cheaper, not replace it.

**What the lint rule checks (sketch — exact form deferred to implementation).**
For modules registered as "mechanism modules" (initially the five C.2.x
dynamics + any future ones added under `phase3/dynamics/` or
`online_consolidation/`):

- No imports from `phase4.scheduling`, `phase5.bridge_readouts`, or any
  module marked as a "downstream consumer" in the catalog.
- No reads from a metric snapshot type (e.g. `MetricsBundle`,
  `DriftStatistics`) at module top-level or inside the mechanism's
  per-tick function.
- No `if … then act` branches whose condition is a cross-subsystem metric.

The catalog of allowed dependencies per mechanism module is the rule's
load-bearing data; living-document, edited by the user when a mechanism
genuinely needs a new dependency (which is itself a reviewer-PASS event).

**Contract location.** **`src/energy_memory/architecture/`** is reserved
for this. Holds the lint catalog plus any future structural contracts that
earn their place. The naming signals that this is *project-architectural*,
not a phase mechanism — anything that lands here is non-negotiable.

**Anti-homunculus check on the lint rule itself.** The lint rule reads
*source-code shape*, not *mechanism state at runtime*. It does not arbitrate
between subsystems; it constrains what a single mechanism module can depend
on. It is allowed under the rule it enforces.

**What changes (later — not in this session).**
- Create `src/energy_memory/architecture/__init__.py` and
  `src/energy_memory/architecture/local_dynamics_catalog.py` holding the
  per-mechanism allowed-dependency list.
- Add a `pytest` collector or pre-commit hook that walks the catalog and
  AST-checks each registered mechanism module.
- The first entries: the five C.2.x dynamics, with their currently-permitted
  imports lifted verbatim from today's tree.

**ADR-shaped note.** Recorded so future architecture reviews don't re-suggest
"lift the anti-homunculus rule into a typed base class" — that path is
explicitly rejected on regress grounds, not on cost grounds.

## Card 2 — Cross-phase imports under `phase5/`

**Decision.** Fix **Leak B only** (the runtime import of an `experiments/`
script from `src/`). Leave the Phase 2 / Phase 4 source-code imports alone.

**Why the narrowing.** The original review card flagged two distinct things
under one heading; grilling separated them.

- **Leak A — `phase5/*.py` imports `phase2.encoding` and `phase4.trajectory`
  directly.** Not a leak. `phase2.encoding` (`encode_window`, position
  vectors, `decode`) is substrate-level reusable per `CLAUDE.md` and is
  intentionally cross-phase. `phase4.trajectory` is similar. Wrapping these
  in a "phase-integration adapter" would be a one-adapter seam — by the
  LANGUAGE.md rule (*one adapter = hypothetical seam, two adapters = real
  seam*), it's premature depth. Leave as-is.

- **Leak B — `phase5/bundle_first_scene_memory.py` does a runtime import of
  `experiments.44_phase5_prime_bundle_first`.** Sharp leak: a `src/` module
  is load-bearing on an `experiments/` script. If exp 44 is renamed,
  deleted, or moved (any of which is normal for a numbered experiment),
  `src/` breaks. The experiments tree is meant to be a scratchpad, not
  infrastructure.

**What changes (later — not in this session).**
- Identify whatever `bundle_first_scene_memory.py` (and any sibling) imports
  from `experiments.44_*` at runtime.
- Hoist that code into `src/energy_memory/phase5/` (most likely into the
  bundle-first scene-memory module itself, or a small new sibling). The
  hoisted code becomes the canonical home; experiments/44 may keep a thin
  re-export for back-compat or be updated to import from `src/`.
- Add a small grep test that fails if any `src/` file imports from
  `experiments.*` at runtime — cheap, prevents regression. Lives next to
  the Card 4 lint catalog under `src/energy_memory/architecture/`.

**Working-agreement check.** No supervisor; just a code-location fix. The
hoisted code keeps the same local geometry; only its import path moves.

**Phase 5 rename deferred.** Considered renaming `phase5/` to a concept
name (e.g. `scene_memory/`, `bundle_first/`) parallel to Card 5's
`online_consolidation/` rename. **Rejected for now**: Phase 5′ is paused
under Path C and may pivot; renaming would churn references in 100+
numbered reports for a phase whose shape may change. Revisit if/when
Phase 5′ reopens with a stable concept.

## Cards 1 + 3 — Experiment lifecycle runner + natural-source-protocol entry point

**Designed together** because they overlap on lifecycle ownership; treating
them as one decision avoids two parallel orchestrators with disputed
boundaries.

### Card 1 decisions

**Module.** New package **`src/energy_memory/experiment_step/`** holding one
module per step:

- `experiment_step/precommit.py`
- `experiment_step/preflight.py`
- `experiment_step/smoke.py`
- `experiment_step/gate.py`
- `experiment_step/parity.py`

Plus shared internals (likely `_envelope.py` for the JSON framing block,
`_sha.py` for the SHA-256 stamping, `_aggregate.py` for Wilson CI / LOO).
Names match the project's existing report vocabulary 1:1; nothing new to
learn at the seam.

**Granularity.** One runner call per step. Each step produces its own JSON
artifact with its own SHA-256. Reports already work this way (Report 102 is
preflight, 103 is smoke, 106 is bridge smoke, 107 is analysis); the runner
matches the existing rhythm rather than imposing a new one.

**Script minimum.** Each `experiments/NN_*.py` script provides only:

- A **per-trial closure** that takes a single condition + seed and returns a
  result dict.
- A **condition table** (list of dicts) declaring the conditions to run.

The runner owns: seed loop, scale loop, Wilson CI, LOO, JSON envelope,
SHA-256, framing block (active phase, headline metric citation, why this
experiment now — per CLAUDE.md's experiment preamble requirement).

**Pure-map runner.** The runner is a pure map: closure × conditions × seeds
→ result table → aggregated artifact. No early-exit, no result-driven
condition skipping, no adaptive seed counts. This is the **anti-homunculus
check**: a runner that reads results and routes would be a supervisor; this
runner doesn't. Any adaptive behaviour (e.g. early-exit on a clear null)
belongs in a separate `adaptive_runner` module that wraps the pure runner —
explicit, isolated, reviewed independently.

### Card 3 decisions

**Decision.** `natural_source_protocol` becomes a **caller** of
`experiment_step/`, not a peer. The protocol module gets one entry point
(e.g. `run_natural_source_step(step, ..., closure=...)`) that internally
calls the right `experiment_step.<step>.run(...)` with the protocol-specific
preflight/gate/parity closures wired in.

**Why caller, not plugin or replacement.** A plugin-shaped seam would let
the protocol customise step internals — that's both more surface area and
gives the protocol the ability to influence step behaviour invisibly. A
replacement-shaped path would duplicate lifecycle code across two
orchestrators. Caller-shaped is the cleanest: anyone running a natural-source
experiment goes through the protocol; anyone running a different family
calls the steps directly; nobody disputes who owns SHA-256.

### Joint working-agreement check

- **Anti-homunculus.** Pure-map runner; no result-driven routing. PASS.
- **No supervisor.** Condition selection is *data* (a table the script
  provides), not a decision the runner makes. PASS.
- **Don't make the LLM the source of persistence.** Runner owns JSON
  artifact writing; no LLM in the loop. PASS.
- **Don't trust a new mechanism until it survives a control condition.**
  This is *the runner itself* — its first commits should include a smoke
  that reproduces an existing report byte-identically through the new
  runner (Report 095-style parity), proving the runner doesn't change
  results.

### What changes (later — not in this session)

1. Scaffold `src/energy_memory/experiment_step/` with the five step modules
   and shared internals.
2. Migrate one Phase 5′ experiment (suggest the smallest — Report 102's
   preflight is a good candidate) end-to-end through the runner.
3. Run a byte-identical parity check against the original report's JSON
   (Report 095-style).
4. Migrate the natural-source-protocol caller next.
5. Migrate remaining Phase 5′ scripts opportunistically; older Phase 2 /
   Phase 3 scripts only as they're touched for other reasons.

**ADR-shaped note.** Future architecture reviews should not re-suggest a
"whole-lifecycle" or "result-routing" runner — both were considered and
rejected here on cohesion + anti-homunculus grounds.

## Card 6 — `phase2` metrics seam

**Decision.** **Fold into Card 1.** No separate refactor.

**Why.** Two of the three original wins from this card arrive automatically
through Card 1's `experiment_step/` runner:

- *9 experiments → one interface.* The runner calls Wilson CI / LOO
  internally via `experiment_step/_aggregate.py`; experiments stop
  importing `wilson_interval` / `cap_coverage` directly.
- *Tests stop importing `_`-prefixed helpers from `phase2`.* Tests hit
  the runner's seam instead.

**Residue not covered by Card 1.** One private import in
`tests/test_drift_replay_tension.py`:
`from energy_memory.phase2.encoding import encode_window as _ew`.
This is reaching into `phase2.encoding`, not `phase2.metrics`. Fix it
opportunistically the next time that test is touched, by either:

- making `encode_window` part of `phase2`'s public interface (it's
  already used as if it were), or
- routing the test through the Card 1 runner.

**No retrieval-aggregation module.** Considered building
`phase2/retrieval_aggregation.py` as a dedicated semantic home for
Wilson / cap-coverage / meta-stable rate. **Rejected**: the runner is
the right home for aggregation in this codebase (it's where the seeds
and conditions live); a parallel `retrieval_aggregation` module would
duplicate intent and force callers to choose.

**ADR-shaped note.** Future reviews should not re-suggest a
`retrieval_aggregation` module in `phase2/` — aggregation lives in
`experiment_step/_aggregate.py` per Cards 1+3.

---

## Sequencing (later — not in this session)

Recommended order for the implementation work:

1. **Card 5** (rename `phase34/` → `online_consolidation/`). Mechanical,
   no design risk, unblocks vocabulary in PR titles and reviews.
2. **Card 2** (hoist the `experiments.44_*` runtime import). Small,
   focused, removes a load-bearing dependency on a scratchpad file.
3. **Card 1+3** (build `experiment_step/`, migrate one report
   byte-identically, then migrate the natural-source-protocol caller).
   The biggest piece; do it after the noise from 1 and 2 has cleared.
4. **Card 4** (lint rule under `src/energy_memory/architecture/`). Best
   landed *after* Card 1+3 so the lint catalog has a stable surface to
   describe.
5. **Card 6** residue (one private import) — folded into routine
   maintenance; no standalone task.

**Phase context.** Path C (Phase 3 reopen) is the active research thread.
None of these refactors is on the Path C critical path; treat them as
opportunistic improvements between Path C work items, not blockers.
