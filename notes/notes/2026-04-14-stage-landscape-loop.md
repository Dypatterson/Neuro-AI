---
date: 2026-04-14
project: personal-ai
tags:
  - notes
  - subject/personal-ai
  - subject/cognitive-architecture
  - project/personal-ai
---

# Stage / Landscape / Loop — Three-Layer Architecture

Working note from the 2026-04-14 session. Extends the 2026-04-10 writeup (`2026-04-10-loop-and-three-phases.md`) after the memory-vs-world-model fork resolved into a three-layer picture. This note is the architectural reference; the journal entry is the narrative.

## The starting point

Coming out of 2026-04-10, we had an A→B→A→B loop between "memory" and "world model," three phases (diffuse / focused / reactive), trajectory consolidation, reinforcement as a brake, and the question as shared currency (cue and criterion at once). We also had an explicit open fork: *is memory and world model one substrate accessed two ways, or two genuinely distinct structures looped together?*

The JEPA repos note from 2026-04-12 reframed that fork as: "does LeWM validate or falsify the single-substrate horn?" That was the entry point for this session.

## What changed

### The word "world model" was doing double duty

It was being used to mean both:

1. A **representational space** where current perception manifests (colored by mood, memory, context) — a stage for things to happen on.
2. A **predictive process** that runs forward on that space — "given this state, what comes next?"

Collapsing these two generated a fake timescale problem. If the "world model" is both the space and the predictor, and the predictor has to run fast while the landscape reshapes slow, the two timescales appear to be competing for the same surface. Split them, and the problem largely dissolves.

### The memory-vs-world-model fork, sharpened

Set up as a 2x2 on two axes — state-space shared? × dynamics shared?

|  | Dynamics shared | Dynamics separate |
|---|---|---|
| **State-space shared** | A. Fully collapsed (LeWM-like) | **B. Shared stage, different rules** ← live candidate |
| **State-space separate** | C. Same rules different worlds (dead) | D. Fully distinct modules |

Cell B chosen. Same terrain walked at different rates by different processes. Cleaner than A (avoids timescale collision), cleaner than D (preserves the shared-currency loop instead of routing everything through an interface), C doesn't instantiate.

## The three-layer picture

### Stage — fast

The representational space where the present manifests. Not raw perception; perception already tinted by memory, mood, prior context. Updates at perception-speed. This is what "world model" *really* meant in Dylan's intuition.

### Landscape — slow

The consolidated terrain. Shaped by long-timescale consolidation (surprise-driven + reinforcement-braked, per 2026-04-10). Where trajectories have grooved over time. Where personality lives. Where diffuse mode wanders at rest.

The landscape bleeds *into* the stage — memory colors perception — so these are not fully separate in practice. The stage is partly a readout of the landscape under current input. But they update at different rates and serve different functional roles, which is why the split is worth making.

### Loop — activity across both

The A→B→A→B loop from 2026-04-10, now understood as running *across* stage and landscape rather than between memory and world model. Reads the landscape, projects onto the stage, takes the stage's state as a new cue back into the landscape, repeats. Expands reachable memory-space with each pass because running-forward changes what a retrieved thing is as a subsequent cue (this is the substantive claim under "creativity lives in the handoff," keeping the claim and dropping the phrase).

### Prediction reclassified

Prediction is not a fourth piece. It is a **mode** of the loop, distinguished by the shape of the question currently driving it:

- Tight temporal question ("what's next?") → **prediction**
- Pastward question ("what happened?") → **recall**
- Counterfactual question ("what if?") → **imagination**
- Constraint-satisfying question ("what works here?") → **reasoning**
- Loose question ("what comes to mind?") → **creativity**

This extends the reasoning/creativity unification from 2026-04-10 to cover the whole family. One loop, one activity, many modes picked out by question shape.

## Anti-collapse as a candidate general principle

LeWM uses a Gaussian regularizer on its latents to prevent representation collapse under prediction pressure. The 2026-04-10 architecture uses user reinforcement as a brake against runaway consolidation. Both are anti-collapse pressures under a consolidation-driven loop. Different substrates, same structural job.

Filed as a working hypothesis: **any learning loop with a consolidation pressure may need an anti-collapse pressure as a counterweight.** Not proven. Worth stress-testing against other cases before promoting from hypothesis to principle.

## What LeWM actually says about the fork, after the reframe

LeWM puts its encoder and predictor in one latent space. But those latents are *ephemeral* — trajectory-local, live-and-die within a single rollout. That makes them structurally analogous to the **stage**, not the landscape.

So LeWM validates that one kind of memory (the present representational state) can share substrate with prediction. It says nothing about the consolidated landscape — there is no hippocampus in LeWM, no cortex, no long-timescale terrain. The single-substrate horn for *stage-memory* is addressed; the single-substrate horn for *landscape-memory* is untouched.

Useful to have pinned: LeWM is not an existence proof for what we actually care about. It is an existence proof for something narrower.

## Side-distinction established: substrate vs. dynamics

- **Substrate** = the medium where states live. Vector spaces, state spaces, languages, attractor landscapes. A place.
- **Dynamics** = the rules by which states change. Transition matrices, update rules, prediction functions. A how.
- **Pattern** = a property or shape that emerges from a substrate's configuration. Personality is a pattern, not a substrate.

Attractor landscapes blur substrate and dynamics (the shape of the landscape *is* the dynamics). That's a feature, not a confusion — but worth noticing when it happens.

## Open questions after this session

1. What does *writing* a trajectory into the landscape look like mechanically? (Carryover, now narrower — specifically into the slow structure, not the fast one.)
2. What makes a diffuse-mode item salient enough to crystallize into a focused-mode question? (Carryover. May now be a stage-landscape interaction rather than happening in one place.)
3. Does anti-collapse hold as a general principle across any loop with consolidation pressure? (New.)
4. Residual timescale question: when the loop reads the landscape while consolidation is writing to it, is there any interference? (Mostly dissolved but not fully closed.)
5. JEPA still has more to say once we're looking at it as a stage-only architecture. What does a JEPA-shaped landscape look like, if one were to exist?

## What this note is not

Not a build spec. Not load-bearing. If the next session's thinking contradicts something here, the new thinking wins. This is scaffolding for where the architecture sits at the end of 2026-04-14, not a commitment.
