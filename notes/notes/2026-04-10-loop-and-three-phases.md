---
date: 2026-04-10
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - project/personal-ai
---

# The Loop, The Question, and The Three Phases

Working note from the 2026-04-10 session. Captures new architectural structure that emerged while working through LeCun's "Introduction to Latent Variable Energy-Based Models" paper (Dawid & LeCun 2023) and stress-testing the loop hypothesis against counterexamples. This note is exploratory scaffolding, not a spec — see the Posture section of the briefing.

## What changed from the 2026-04-09 briefing

The 2026-04-09 briefing claimed memory and exploration might be the same operation. This session showed that claim is incomplete. Retrieval is generative (call it A — blended recall from the landscape) and simulation is projective (call it B — projecting forward through a world model), and creative thought happens in a **loop between them**: A→B→A→B, where each step's output becomes the next step's input. Neither the previous collapse nor LeCun's separation of memory and world model into distinct modules captured this loop structure.

The world model earned its place in the architecture during this session, but not as a standalone module — as one half of a loop whose other half is the memory.

## The loop

```
question (held in workspace)
    │
    ├──▶ cue ──▶ memory retrieval (A) ──▶ blended candidate
    │                                         │
    │                                         ▼
    │                                    world model (B)
    │                                         │
    │                                         ▼
    │                                    projected consequence
    │                                         │
    │◀────────────────── new cue ◀────────────┘
    │
    ▼
(terminates when candidate fits question-shape, or
 neighborhood exhausted without fit)
```

What makes this loop creative rather than either pure recall or pure simulation is that A and B keep handing work off to each other. Pure A alone gets you blended recall (limited combinatorial creativity). Pure B alone gets you planning (useful but not surprising). A feeding B feeding A lets the system retrieve a blend it didn't store, run it forward to see what it implies, and use the implications as a new cue that reaches into parts of memory the original cue could never have touched. The creativity lives in the handoff.

## The question as shared currency

The loop needs something held steady across iterations, otherwise it would drift. That something is **the question** — and a question does two jobs at once:

1. It serves as the **cue** that drives retrieval (input to A).
2. It serves as the **criterion** that evaluates whatever comes back (the test that decides whether the loop terminates).

These two jobs are the same job from different angles. A question is simultaneously "what I'm looking for" and "how I'll know when I've found it." The cue and the criterion are the same object, used at different moments in the loop.

This collapses what was going to be a third module (an evaluator) into a single object held in the workspace. The workspace isn't just a translation layer between the LM and the memory anymore — it's where the active question lives while the loop runs against it.

This also sharpens the claim from 2026-04-09 that "the query should belong to the idea, not the token." A stronger version: **the query should belong to the question, not the idea.** The thing that drives retrieval isn't a static representation of what's currently in mind — it's an active, unresolved gap looking to be filled.

## Reasoning and creativity are the same mechanism

This session's cleanest claim, and the one LeCun's paper almost makes but doesn't quite:

**Reasoning and creativity are not two faculties. They are the same loop operating on differently-shaped questions.** Tight constraint-shape (what's 17 × 23?) → narrow reachable neighborhood → feels like reasoning. Loose constraint-shape (what would happen if memory and world model were the same thing?) → vast reachable neighborhood → feels like creativity. Same mechanism, different question topologies.

LeCun frames reasoning as energy minimization (System 2) and leaves creativity basically unaddressed. This frame covers both without needing a separate "creativity module."

## Stress test: what killed the single-loop hypothesis

The single-loop hypothesis as originally stated (all thought is loop-with-a-question) failed against shower-thoughts. The "I should call my brother" example was load-bearing: the thought arrived first, the question ("I wonder what he's up to") formed around it a fraction of a second later. If the loop hypothesis were complete, the question would have come first. It didn't. The thought led, the question followed.

What replaces the single-loop picture is a **two-mode structure**.

## The three phases of one process

After stress-testing the two-mode picture against an external-stimulus case (missing the bus), the two modes grew into three phases that are actually one continuous process at different stages of consolidation:

### Diffuse mode
The landscape in its resting state, shaped by everything that's ever been consolidated. No specific question held. Retrieval is driven by landscape topology alone — recent primings, lingering tensions, emotional weights, unresolved pressures. Things surface because the terrain made them reachable, not because anything was looking for them. This is the regime that produces shower-thoughts, associative drift, and the experience of "finding yourself thinking about something" without knowing how you got there.

Diffuse mode's job is to surface things salient enough to crystallize questions for focused mode to work on.

### Focused mode
A question has crystallized and is held steady in the workspace. The A→B→A→B loop runs against it. Candidates are tested against the question's shape. The loop terminates when a candidate fits (resolution) or when the reachable neighborhood is exhausted without a fit (dead-end — "I don't know"). This mode covers both reasoning and deliberate creativity, differentiated only by how tightly the question constrains the answer-shape.

### Reactive mode
What a focused-mode trajectory becomes after enough repetitions have carved the trajectory deep enough into the landscape that the stimulus alone is sufficient to trigger the whole trajectory without the loop needing to run. Well-worn patterns play back as a unit. Missing the bus the hundredth time doesn't re-derive the consequences — the whole cascade activates as a consolidated groove.

### The unification
These are not three separate mechanisms. They are **three phases of one consolidation-driven process**, differentiated only by how much the landscape has been shaped by prior work on a given kind of question:

- **No active question** → diffuse drift through existing grooves
- **Novel question** → focused-mode loop carving a fresh trajectory
- **Well-worn question** → reactive-mode schema playback along a pre-cut groove

Repetition is what moves a trajectory from focused mode into reactive mode. The mechanism that does this is the same consolidation process identified in the 2026-04-09 briefing (surprise + reinforcement), except with a richer target.

## Trajectories, not just content

The richer target: **what gets consolidated isn't just facts or patterns, it's entire loop trajectories.** The path that focused-mode took from one state to another gets written into the landscape as a single groove. Next time a similar stimulus hits, the groove is deep enough that the system slides down it without reconstructing the reasoning from scratch.

This upgrades the 2026-04-09 claim that "personality lives in the memory, not the model." Personality isn't just *what* the system remembers — it's **the paths its thoughts habitually take through what it remembers**. Two systems with identical stored facts but different consolidated trajectories would reason completely differently, because their landscapes would have different grooves carved into them. The self is not the content of memory but the topology of its well-worn paths.

## What this leaves open

- How literal is the distinction between the memory substrate and the world model? Are they two separate structures that happen to loop with each other, or the same substrate accessed through different operations? This is a real architectural fork that got glossed over in session.
- What does *writing* into the landscape look like when what's being written is a trajectory rather than a pattern? Trajectory-consolidation gives a new angle on the write operation question from 2026-04-09 but doesn't answer it.
- JEPA is still unaddressed. It was originally one of the two things for this session and we deferred it. Returning to it now that the loop structure is in place will make it land differently than it would have.
- The question-crystallization mechanism (what makes diffuse mode hand off to focused mode) is named but not specified. What exactly makes a surfaced item "salient enough" to form a question around it?

## Status

Exploratory scaffolding. Survived one stress test (shower-thoughts) and one potential counterexample (external stimuli → cascade). Has not been formalized. Should not be treated as load-bearing until further sessions poke at it. Consistent with the project posture: understanding first, building later.
