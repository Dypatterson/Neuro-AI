---
date: 2026-05-04
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
---

# LLM Integration (Option A)

How the emergent codebook integrates with the LLM-as-voice via the Global Workspace. This is Phase 6 architecture; not relevant for Phases 1-5 but on the map from the start so the substrate interface is built compatibly.

## The decision: Option A

**BPE permanent for input segmentation, codebook permanent as internal meaning representation, LLM in the workspace pathway for language output.**

The codebook is purely the substrate of memory. The LLM stays as the language interface (input/output). The workspace bridges them — the same workspace already in the briefing, just with a clearer specification of what it does on each side.

## Why Option A over alternatives

**Option B** (codebook eventually subsumes BPE — emergent vocabulary) is more thesis-aligned in principle but adds significant complexity. The transition from "BPE input atoms" to "emergent input atoms" is itself a research problem and isn't necessary for a working v1.

**Option C** (tight LLM-codebook coupling — LLM token embeddings ARE the codebook) is most ambitious but most fragile. Coupling means anything that breaks the codebook breaks the LLM, and the consolidation dynamics weren't designed with LLM embedding stability in mind.

Option A is the cleanest and most modular. Components stay independent. The codebook can be redesigned without affecting the LLM; the LLM can be swapped without affecting the codebook. Modularity matters when both sides are still under development.

## The BPE-as-segmentation distinction

A subtle but important point: BPE doesn't give us permanent atoms — it gives us permanent input *segmentation*. Token IDs are stable (BPE consistently maps "the" → token 1234). Codebook entries for those token IDs drift over time. The segmentation layer is fixed; the meaning layer is plastic.

Biologically: phoneme parsing is stable across adulthood, word meanings shift with experience. Same architectural shape.

This means:

- The same input ("the cat sat") tokenizes to the same token IDs every time
- The codebook lookup for those token IDs returns the current atom hypervectors
- Atoms drift via consolidation, so the same input produces slightly different bundles over time
- Stored Hopfield/SDM patterns from earlier contain older atom versions (Phase 6 staleness issue, mitigated by replay-and-re-encode)

## Input pathway

```
text → LLM tokenizer (BPE) → token IDs → codebook lookup → bundle → Hopfield
```

The LLM tokenizer is the same one the LLM uses for its own inputs (consistent vocabulary). Token IDs map to atom hypervectors via codebook lookup. Sequences become bundled hypervectors via the Phase 1 encoding scheme (`bind(position, token_atom)` for each position, then bundle).

The Hopfield receives the bundle and settles. Settled state is what the workspace reads.

## Output pathway

```
Hopfield settled state → workspace decoder → context for LLM → LLM generation → text
```

The workspace decoder is the new component for Phase 6. It takes settled HD states and produces something the LLM can consume.

### MVP workspace decoder: top-K retrieval

Simplest version. Given a settled bundle:

1. Unbind by relevant positions to extract candidate atom hypervectors.
2. For each candidate, find the K nearest atoms in the current codebook (by cosine similarity).
3. Convert to token IDs, then to text via tokenizer.
4. Return the top-K retrieved tokens or short sequences as "what the system is thinking about."

This produces a list of relevant tokens/phrases that the system has surfaced from memory in response to the current input. Not natural language, just retrieved content.

The LLM then uses this list as context for generating a fluent response. The LLM's prompt structure: user input + retrieved memory context + instruction to respond fluently.

### Why this works for v1

The LLM does what it's good at (fluent text generation). The codebook does what it's good at (structured memory retrieval). The workspace decoder is a thin interface, not a sophisticated component. Each component stays in its lane.

### Future: more sophisticated decoders

Eventually, the workspace decoder could be:

- A learned HD-to-text generator (small transformer that decodes HD states directly to language)
- A cross-attention bridge where the LLM attends directly to settled HD states
- A hybrid where the LLM conditions on both retrieved tokens and the raw HD vector

These are post-Phase-6 directions. Don't build them in v1; they may not be needed.

## What the workspace does

The workspace is more than just the decoder — it's also the input bridge and the place where "ideas" exist between modules. From the briefing:

> The workspace is where "an idea" exists in a form separable from any particular module. Queries into the long-term memory come from the workspace, not from the LM directly. Results return to the workspace where the LM and other modules can read them.

For the codebook architecture, this means:

- The workspace receives input bundles from the encoding pathway
- The workspace queries the Hopfield with cues derived from current activity (not raw token bundles, but the *idea* the system is currently holding)
- The workspace receives settled states from the Hopfield
- The workspace runs the decoder to produce LLM-consumable context

The "query should belong to the idea, not the token" principle from the briefing carries through. The workspace produces queries that aren't necessarily token-shaped; they can be richer combinations of recent context, surfaced memories, and emerging intentions. The codebook architecture supports this naturally because cues are HD vectors, and arbitrary HD vectors can serve as cues.

## Interaction with consolidation

Live use produces only Hebbian adjustments — small reinforcement of patterns that successfully retrieve. Significant codebook learning happens during dedicated training passes (over corpus or replay buffer) where masked-token training provides the error-driven signal.

This means:

- During conversation: light reinforcement, no major codebook changes
- During replay/sleep cycles: error-driven updates, larger codebook adjustments
- Stored patterns periodically refreshed via replay-and-re-encode

The companion's experience of the user shapes consolidation through what gets stored and replayed — the user's actual conversations are the training data, but learning happens in dedicated passes, not online. This matches the existing project architecture's wake/sleep distinction.

## Phase 6 design questions

Open questions to resolve when Phase 6 begins:

1. **Replay-and-re-encode frequency and prioritization.** How often to refresh stored patterns? Which patterns get refreshed first when refresh capacity is limited? Recency-weighted? Importance-weighted? Drift-magnitude-weighted (patterns whose constituent atoms have drifted most)?

2. **Cue construction in the workspace.** What's the exact mechanism for turning "current activity in the workspace" into a Hopfield cue? Bundle of recent inputs? Weighted bundle of recent and currently-attended content? This is itself a research question.

3. **Decoder fidelity.** Is top-K retrieval enough for fluent LLM responses, or does the decoder need to provide more structured context (relationships between retrieved items, temporal context, etc.)?

4. **Workspace query frequency.** Does the workspace query the Hopfield once per user turn, or continuously throughout the conversation? Continuous querying would let the system surface relevant memories mid-thought; per-turn is simpler.

5. **LLM model choice for v1.** Small local model (Llama 3.2 1B-3B, Qwen 2.5 3B, Liquid Foundation Model)? The briefing originally listed candidates; the choice doesn't depend on the codebook architecture but does affect deployment.

These don't need answers for Phase 1. They need to be on the map so the substrate interface is built compatibly.
