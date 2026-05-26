# Neuro-AI Ground Truth Principles

This file is the compact project doctrine for future sessions. It does not
replace primary literature, numbered reports, phase specs, or `STATUS.md`; it is
the entrypoint that tells an agent how to use them without drifting into a
different project.

## Authority Order

1. `STATUS.md` is the live bookmark for active phase, blockers, and fences.
2. The active phase design/checklist defines graduation metrics and controls.
3. Numbered reports and raw artifacts define what has actually been run.
4. This ground-truth pack routes literature and principles into design work.
5. Source cards are summaries. Primary PDFs and URLs remain authoritative.

## Core Commitments

- **No controller.** Do not add a supervisor that decides what matters, which
  subsystem wins, or when to switch strategies.
- **Memory is the self.** Durable identity lives in the learned landscape,
  trajectories, and consolidation history, not in LLM weights.
- **Contextual completion over token prediction.** The native question is what
  the system is reminded of and what latent gap remains unresolved.
- **Continuous learning.** Live experience writes traces; replay consolidates,
  abstracts, and reshapes the landscape.
- **Latent reasoning.** Reasoning should happen in vectors, energy states, and
  settling trajectories before language is generated.
- **Local-first.** The research system should remain plausible on a MacBook Pro
  class machine.
- **Energy efficiency.** Prefer compact latent operations, sparse activation,
  replay, and accelerated batched math over large decoding loops.

## Anti-Homunculus Filter

Every proposed addition must either be a local geometric dynamic or be
expressible as a measurement of one.

Before accepting a mechanism, answer:

- What quantity moves locally?
- Where does the apparent decision live in energy, geometry, settling,
  tension, replay, or consolidation?
- What fixed dynamic replaces any tempting `if metric then action` rule?
- What control would show the mechanism is not hidden arbitration?

Failing shape:

- A metric reader triggers a response.
- A supervisor chooses between subsystems.
- A fallback path adaptively changes source families after seeing results.
- A router picks the best condition and reports it as the headline.

Passing shape:

- The same energy function always applies.
- Sampling, replay, and consolidation probabilities are fixed functions of
  local state.
- Slow-timescale variables modulate dynamics without inspecting outcomes.
- Controls can be precommitted before retrieval or training.

## Headline Vs Drill-Down Discipline

Each phase has one headline metric that defines viability. Other metrics are
drill-downs that explain the headline, not alternate success definitions.

Future work must state:

- the active phase;
- the design-spec headline metric and required controls;
- whether the proposed measurement is a headline run or a drill-down;
- the source IDs and principle checks motivating the work.

Multi-metric panels without a named headline are not acceptable evidence.

## Proposal Citation Rule

Every non-trivial mechanism proposal should cite:

1. at least one `source_id` from `source_manifest.jsonl`; and
2. at least one principle check from this file.

If a source is `link_only`, it can motivate a literature scan or precommit, but
it should not be treated as load-bearing implementation evidence until a full
source card has been written from the primary source.
