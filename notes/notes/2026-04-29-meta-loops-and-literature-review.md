---
date: 2026-04-29
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
---

# Meta-Loop Mechanics and Literature Review

## Entry Point

Seventh working session. Before diving into meta-loop mechanics (highest-priority open question from session 6), Dylan surfaced three external sources found between sessions. Two proved highly relevant to the architecture.

## Literature Review

### Ng — "LLM Neuroanatomy III: The Language-Agnostic Middle" (March 2026)

Source: https://dnhkng.github.io/posts/sapir-whorf/

Core finding: transformer LLMs have a three-phase internal anatomy — early layers decode surface form (language, script, code syntax) into a shared semantic space, middle layers reason in that format-agnostic space, late layers encode back to surface tokens. Replicated across five architecturally different frontier models (Qwen3.5-27B, MiniMax M2.5, GLM-4.7, Gemma-4 31B, GPT-OSS-120B). Middle layers cluster representations by meaning, not language — photosynthesis in Hindi is closer to photosynthesis in Japanese than to cooking in Hindi.

Extended to code and LaTeX with single-letter variables: cross-modal convergence in the mid-stack, not just cross-lingual. Connected to the Platonic Representation Hypothesis (Huh et al. 2024) and the Semantic Hub Hypothesis (Wu et al. 2024).

**Relevance to our architecture:**

- Empirical validation that the kind of semantic space SONAR is designed to be — language-agnostic, meaning-organized — actually forms spontaneously inside large enough transformers. Strengthens the case for SONAR as substrate.
- The encode/reason/decode structure maps onto what happens when a query enters the landscape: translation from linguistic input into workspace representation → reasoning in the landscape → LM turning a result back into words.
- Cross-modal convergence (code, LaTeX, natural language → same mid-stack region) supports SONAR embeddings as a genuine lingua franca for the system, not just a language encoder.
- Key tension: Ng's finding is about what emerges *inside* a pretrained transformer ephemerally. Our architecture externalizes and persists that middle space as the Hopfield landscape. We're engineering permanence for something transformers build and throw away each forward pass.

### Hao et al. — "Coconut: Chain of Continuous Thought" (arXiv 2412.06769, FAIR/Meta)

Source: https://arxiv.org/html/2412.06769v3

Core idea: instead of forcing LLM reasoning through word tokens (chain-of-thought), feed the last hidden state directly back as the next input embedding — bypassing the language bottleneck. The model switches between "language mode" (normal token prediction) and "latent mode" (hidden states cycling without decoding). These hidden states are called "continuous thoughts."

Key finding: when reasoning happens in continuous space, the model spontaneously develops BFS-like behavior. A single continuous thought can encode multiple alternative next steps simultaneously, whereas a language token commits to exactly one. The model explores multiple paths in parallel and prunes bad ones — something chain-of-thought structurally cannot do.

**Relevance to our architecture:**

- Closest published work to our "query belongs to the idea, not the token" principle. Coconut literally decouples reasoning from the token stream.
- BFS emergence connects to temperature-as-loop-mode: a continuous thought encoding multiple alternatives is functionally similar to a high-temperature query hitting the Hopfield landscape where multiple attractors contribute simultaneously.
- Structural parallel: Coconut's "latent mode" is what happens inside a single forward pass when the system thinks without speaking. Our architecture externalizes this — the Hopfield landscape is a persistent version of that latent reasoning space, and the workspace is where ideas exist without being linguistic.
- Coconut demonstrates that removing the language constraint changes the *kind* of reasoning that emerges, even within a standard transformer. Our design does this by construction.
- Training note: Coconut uses progressive internalization — language CoT gradually replaced by continuous thoughts. The principle (start explicit, let the system learn to compress) might inform how the meta-loop develops capability over time.

### Jia et al. — "The AI Hippocampus" survey (arXiv 2601.09113)

Assessed as low-medium relevance. Broad taxonomy of memory in LLMs: implicit (in weights), explicit (external stores), agentic (persistent in autonomous agents). Confirms our architecture doesn't fit cleanly into any existing category — we blur the implicit/explicit boundary. Filed as reference for later, not actionable now.

## Meta-Loop Mechanics

### The Core Move — Uncertainty as the Signal

Dylan's initial proposal: the meta-loop is not a separate process — it's uncertainty. The self-check between a snag catching attention and full engagement isn't an inspection; it's the landscape failing to resolve cleanly.

This follows the anti-homunculus principle: the settling process itself produces the signal. No module stands outside watching the dynamics — the dynamics watching themselves IS the meta-loop.

### Resolution and Engagement — Two Readable Properties

A single quantity (uncertainty / failure to settle) doesn't distinguish between productive uncertainty and noise. Two scenarios illustrate the gap:

- **Productive uncertainty:** reading a claim that contradicts a prior understanding. Multiple attractors pull on the query. The landscape is active but unresolved. Feels like a snag worth engaging with.
- **Noise:** receiving meaningless input ("synergize quantum blockchain wellness optimization"). Nothing in the landscape responds. The input "goes in one ear and out the other." No deliberate dismissal — just absence of engagement.

Both produce low resolution. But they differ on a second axis: engagement — how much total force the landscape exerts on the query. This yields a 2×2:

| Resolution | Engagement | Experience |
|---|---|---|
| High | High | Confident recognition — "I know this" |
| Low | High | Productive uncertainty — the snag, the contradiction, sustained attention |
| High | Low | Faint familiarity — déjà vu without content |
| Low | Low | Nothing grips — passes through without registering |

Noise self-filters by failing to engage the landscape. No rejection gate needed.

### Dissolution of the Three Roles

The three previously identified "roles" of the meta-loop all cash out as responses to resolution and engagement:

1. **Snag self-check** — not a gate. It's the settling dynamics taking time. High engagement + low resolution = the thing stays active. Low engagement = falls through. The "check" is just the landscape doing what landscapes do.

2. **Query shaping / temperature** — the meta-loop doesn't shape the first query. The landscape's response profile shapes what happens next. Failure to settle feeds back as a modified query. Temperature isn't set before retrieval — it evolves through iterative settling. Queries that keep failing to resolve stay diffuse (high temperature). Queries that converge sharpen (temperature drops). The loop IS the temperature controller, but it's not controlling.

3. **Consolidation oversight** — engagement tells whether the landscape has relevant structure in this region. Low engagement = sparse = gate threshold low (density-dependent gating from session 6). High engagement + low resolution + high prediction error = strong write signal. Low engagement = weak write signal regardless of prediction error.

The three roles were never three functions of one mechanism — they were three descriptions of the same dynamics from different angles.

### Engagement-Dependent Convergence Rate (Coconut Connection)

Standard Hopfield dynamics want to converge — each iteration moves closer to an attractor. But there's value in sustaining superposition when the landscape is contested (Coconut's BFS emergence demonstrates this in transformers).

Mechanical proposal: when engagement is high and resolution is low, the settling dynamics slow down — the effective temperature stays elevated rather than cooling toward convergence. The system carries multiple live hypotheses forward rather than committing early. This isn't a new module — it's a property of the settling dynamics themselves.

High engagement + low resolution = slow convergence = sustained superposition (deliberation, creative thinking).
High engagement + high resolution = fast convergence (recognition).
Low engagement = query passes through regardless.

This follows the anti-homunculus principle: nobody decides to slow down. The dynamics ARE slower when the landscape is actively contested.

Neuroscience parallel: neural populations maintain distributed activity across competing representations before committing (drift-diffusion models, Shadlen & Newsome). Hippocampal retrieval co-activates multiple memories before pattern completion resolves to one — the co-activation period is where associative connections between memories become accessible. Collapse too fast → nearest memory. Let competition breathe → relationships between memories → insight vs. mere recall.

### Consequence for SONAR Query-Sharpness Concern

The open concern from session 6 (does SONAR natively carry query-sharpness distinctions?) softens. The first query arrives however SONAR encodes it. The settling behavior determines effective temperature, not the query's initial geometry. A query in a contested region stays diffuse regardless of arrival shape. A query in a deep basin converges fast regardless. Local topology does the work we were worried the query had to do up front.

Doesn't fully eliminate the concern — edge cases may exist where different question types land with identical geometry but need different retrieval behavior. But removes most of the pressure.

### Formal Definition

**Meta-loop:** Not a module or separate process. The meta-loop is the name for the landscape's settling dynamics as read from the system's own perspective. When a query hits the Hopfield landscape, the settling process produces two independently readable properties — **resolution** (how cleanly the query converges to a single attractor) and **engagement** (how much total force the landscape exerts on the query). The combination of these two signals determines downstream behavior: whether attention is sustained, whether the consolidation gate opens, and how quickly the system commits to a retrieval result. The settling rate is itself engagement-dependent — high engagement with low resolution produces slower convergence, sustaining superposition across competing attractors. The term "meta-loop" persists for historical continuity across sessions, but it refers to the dynamics of the primary loop, not a loop running above or outside it.

## What This Session Resolves

- **Meta-loop mechanics** (highest-priority open question from session 6): dissolved into landscape settling dynamics. Two readable properties — resolution and engagement — plus engagement-dependent convergence rate. No module, no controller.
- **Three "roles" of the meta-loop**: unified as three perspectives on the same dynamics, not three functions.
- **SONAR query-sharpness concern** (open question #2 from session 6): substantially softened — settling dynamics, not initial query geometry, determine effective temperature.
- **Noise vs. productive uncertainty**: distinguished by engagement, not resolution. No rejection gate needed — noise self-filters.

## What Remains Open

- Edge cases where SONAR encodes genuinely different question types with identical geometry (softened but not fully eliminated).
- Per-source priors formation (#2 from session 4) — still no concrete mechanism for how source identity modulates engagement/resolution during retrieval.
- Where raw snags come from (#3 from session 4) — untouched this session.
- Whether engagement-dependent convergence rate needs to be explicitly engineered into Hopfield dynamics or emerges from the energy landscape's natural properties at the right scale.
