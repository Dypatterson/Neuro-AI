---
date: 2026-05-02
project: personal-ai
tags:
  - notes
  - subject/personal-ai
  - subject/cognitive-architecture
  - project/personal-ai
---

# Trajectory Trace, Replay, and Consolidation of Discovered Relationships

Derived 2026-05-02 from analysis of BrainCog's KRR-GSNN and the question: what functional property of spiking networks enables emergent abstraction, and can the Hopfield substrate achieve it without switching to SNNs?

## The Problem

In the current architecture, when a query settles in the Hopfield landscape, two properties are readable: **resolution** (how cleanly the query converges to a single attractor) and **engagement** (how much total force the landscape exerts on the query). These are summary statistics of the settling process. They tell the system *how* settling went, but not *where the query traveled along the way*.

Specifically: if a query starts settling toward attractor A, gets partially pulled toward attractor D, then finally resolves to attractor B — the system registers "messy settling, high engagement" but loses the information that A and D were co-activated during the process. The **trajectory** — the sequence of attractor neighborhoods visited during settling — is discarded.

BrainCog's KRR-GSNN gets emergent abstraction (Biden/Putin → "person is president of country") precisely because spike timing preserves the *path* through the network, and STDP wires new connections along paths that fire in temporal sequence. The path carries information; the destination alone does not.

## The Insight

The difference between Hopfield settling and spiking propagation is not that one can do multi-step traversal and the other can't — Hopfield settling *is* multi-step. The difference is that the spiking system treats the path as first-class information, while the Hopfield system only reads the endpoint (plus summary statistics).

The fix is not to swap substrates. It is to **add a read mechanism that traces the settling trajectory** and makes the path available to the rest of the system.

## Three New Components

### 1. Trajectory Trace

A passive observer of the settling process. At each step of Hopfield settling, the trace records which stored patterns (attractor neighborhoods) are significantly activated — i.e., which attractors are exerting meaningful pull on the query at that moment.

Output: an ordered sequence of co-activation snapshots. Example: `[step 1: A↑ C↑ D↑] → [step 2: A↓ C↑ D↑ B↑] → [step 3: C↓ D↑ B↑] → [step 4: B commits]`. The trace captures which attractors appeared together, which competed, and which dropped out.

The trace does not interfere with settling — it only observes.

### 2. Fast Replay Store (Hippocampal Analog)

A temporary buffer that holds recent settling trajectories. **Not** all trajectories — specifically those with **high engagement + low resolution**. These are trajectories where the landscape exerted strong force but the query never cleanly committed. Multiple attractors were pulling hard without resolution.

High engagement + low resolution is the geometric signature of an unresolved trajectory. No controller decides what enters the replay store — the property itself is the gate.

During "downtime" (between active queries, or during dedicated replay cycles), the system **replays** stored trajectories through the current landscape. Crucially, the landscape may have changed since the original settling event — new experiences consolidated, new patterns written. The same trajectory may resolve differently on replay than it did originally.

**Replay behavior:**
- If replay resolves (high engagement → high resolution): the trajectory exits the fast store and its resolution products enter the consolidation pipeline (see below)
- If replay remains unresolved: the trajectory stays in the fast store for future replay
- Trajectories that never resolve after sufficient replay cycles decay out naturally — not everything the system chews on will produce insight

### 3. Second Consolidation Channel

The consolidation gate currently has one input: experiences from the perceptual stage (things that happened in the world). It now has a second input: **discovered relationships** — new patterns produced when previously unresolved trajectories resolve during replay.

When attractors A and D keep co-activating across many unrelated settling events but never resolve, and then during replay the landscape finally resolves their competition into a new combined pattern E (≈ A+D), that pattern E is a genuinely new structure. It was never experienced. It was *discovered* by the system through its own settling dynamics.

Pattern E gets written into the landscape through normal consolidation, subject to the same density-dependent gating as experiential consolidation. This is important: the landscape is now being written to from two sources, so collapse risk increases. Discovered relationships should likely consolidate at **lower initial strength** than direct experiences, requiring re-discovery across multiple replay events before they're written durably. Only robust co-activation patterns survive into the landscape.

## What This Means Architecturally

The landscape now has two qualitatively different sources of knowledge:

1. **Experiences** — things that happened in the world, entered through the perceptual stage
2. **Discoveries** — things the system learned about its own stored patterns through the act of settling through them

The system isn't just remembering anymore. It is **learning from the act of remembering.** The loop — which runs across the perceptual stage and the consolidated landscape — now has a concrete reason to run repeatedly. Every pass through the landscape generates trajectory information. Unresolved trajectories get replayed. Replay against an updated landscape can resolve previously unresolvable competition. Resolution produces new patterns. New patterns change the landscape. Changed landscape changes future trajectories.

This is the meta-loop. It was the highest-priority open question entering this session.

## The Anti-Homunculus Check

Who decides what gets traced? Nobody — the trace passively observes all settling.
Who decides what enters the replay store? Nobody — high engagement + low resolution is a geometric property, not a decision.
Who decides when to replay? Nobody — replay runs during downtime, driven by the energy still present in unresolved trajectories.
Who decides what the new pattern is? Nobody — it's whatever the landscape resolves to when the trajectory finally settles.
Who decides when to consolidate discoveries? Nobody — the same density-dependent gate that governs experiential consolidation.

Distributed into geometry. No new controllers introduced.

## Relationship to BrainCog

BrainCog's KRR-GSNN achieves emergent abstraction through spike timing and STDP — activation flows through shared synaptic pathways, and the temporal sequence of activations creates new connections. This architecture achieves the functionally equivalent result through trajectory tracing and replay — co-activation patterns during Hopfield settling are accumulated, replayed, and consolidated as new landscape structure.

Different substrates, same principle: **relationships discovered through dynamics become durable structure in the memory landscape.**

The Nature Neuroscience (2024) dynamic engrams paper provides biological validation: hippocampal memory engrams are not static after encoding; neurons drop in and out during consolidation; inhibitory plasticity is critical for engrams becoming selective. The replay store's behavior (trajectories that resolve exit; those that don't stay for re-processing; those that never resolve decay) mirrors this dynamic engram lifecycle.

## Open Questions

1. **What data structure holds the trajectory trace?** Sequence of co-activation vectors? Sequence of energy gradients? How much resolution does the trace need to be useful without being prohibitively expensive?
2. **How does the fast replay store interact with sleep/wake cycles?** Biological hippocampal replay happens predominantly during sleep. Does the system need dedicated offline replay periods, or can replay interleave with active use?
3. **How should discovered patterns (E ≈ A+D) be represented?** Are they literally a blend of the SONAR embeddings of A and D? Or something more complex — a new embedding that captures their *relationship* rather than their *average*?
4. **Consolidation strength for discoveries vs. experiences.** What's the right initial strength ratio? Should discoveries require N independent re-discoveries before durable consolidation? What value of N prevents noise while allowing genuine insight?
5. **Does this subsume the meta-loop question entirely, or is there still a separate meta-loop concern?** Tentatively: this *is* the meta-loop, at least at the architectural level. The meta-loop is the landscape thinking about its own contents through settling, replay, and re-consolidation. But this needs to be tested against the specific meta-loop questions from the 2026-04-29 session.
