---
date: 2026-05-01
project: personal-ai
tags:
  - notes
  - subject/personal-ai
  - subject/cognitive-architecture
  - project/personal-ai
---

# Gemini Architecture Review and Google Ecosystem Survey

## Entry Point

Ninth working session. Dylan surveyed Google's AI and developer platforms for tools that complement the Claude-based workflow, compared frontier model capabilities (Gemini 3.1 Pro vs Claude Opus 4.6 vs GPT-5.4), then ran the personal-ai architecture through Gemini 3.1 Pro as a structured second-opinion reviewer. Claude prepared a five-prompt review kit; Dylan ran it and brought back the raw results for triage.

## Google Ecosystem — Tools Worth Using

Five tools identified as genuinely useful alongside Claude, out of ~150 Google offers:

1. **Google AI Studio** — Free Gemini API key, no credit card. Useful for routing cheap/fast tasks (batch processing, multimodal analysis, quick classification) to Gemini while reserving Claude for reasoning-heavy work.
2. **NotebookLM** — Already in use for immunology. Best-in-class "upload sources and talk to them" tool.
3. **Google Colab** — Free Jupyter notebooks with GPU/TPU. Cloud scratch pad for when experiments exceed M5 Pro capacity.
4. **Gemini CLI** — Open source terminal agent. Second opinion tool alongside Claude Code.
5. **$300 Google Cloud trial** — Activate when needed. Covers GPU instances, TPU access, storage. Pair with TPU Research Cloud application for free TPU quota.

Skip: Firebase Studio (sunset), Workspace Studio (enterprise), Antigravity (unstable), Gemini Code Assist (redundant with Copilot + Claude Code), anything with "Enterprise" in the name.

## Frontier Model Comparison (April/May 2026)

LMArena Text Leaderboard: Claude Opus 4.6 #1 (1504 Elo), Gemini 3.1 Pro #2 (1500 Elo), GPT-5.4 ~#5-6 (1484 Elo). Gemini is competitive at the top and dramatically underestimated by the general public.

| Dimension | Leader | Notes |
|-----------|--------|-------|
| Abstract reasoning (ARC-AGI-2) | Gemini (77.1%) | Claude 68.8%, GPT 52.9% — decisive Gemini lead |
| Scientific reasoning (GPQA Diamond) | Gemini (94.3%) | Graduate-level physics/bio/chem |
| Multimodal (Video-MME) | Gemini (78.2%) | Largest gap in any category; native audio+video |
| Writing quality (human pref) | Claude (47% preferred) | GPT 29%, Gemini 24% |
| Coding (Arena Elo) | Claude (1549) | Anthropic holds top 4 spots |
| Price per M tokens | Gemini ($2/$12) | Claude $5/$25, GPT $2.50/$20 |

Key takeaway: model selection should be a routing problem, not a loyalty problem. Different models for different tasks.

## Google DeepMind

The research lab behind Gemini. Founded 2010, acquired by Google 2014, merged with Google Brain in 2023. Led by Nobel laureate Demis Hassabis. ~6,000 employees. Behind AlphaGo, AlphaFold, AlphaZero. Announced AI co-clinician initiative (April 30, 2026) — relevant to Dylan's pre-med trajectory. Research DNA in RL, neuroscience-inspired architectures, and scientific problem-solving explains Gemini's strengths in reasoning and multimodal.

## Gemini Architecture Review — Method

Dylan reported that using GPT for open-ended brainstorming on the personal-ai project was a poor experience. Strategy for Gemini: use as a **structured reviewer**, not a thinking partner. Claude prepared a condensed context block (full architecture summary) and five targeted prompts designed for analytical passes, not conversational back-and-forth. Results brought back to Claude for triage.

## Gemini Review — Triage Results

### Genuine Gaps Identified (Action Required)

**1. SONAR anisotropy (CRITICAL)**
SONAR vectors may be clustered in a narrow cone where random unrelated sentences have cosine similarity ~0.7-0.8. If true, Hopfield energy landscape operates on a substrate where "nearby" doesn't cleanly mean "semantically related." Dot products between query and stored patterns may differ only at 4th decimal place, making floating-point noise potentially deterministic. Requires empirical testing: encode 100 random unrelated sentences, compute pairwise cosine similarities. If confirmed, evaluate **Nomic Embed** as alternative — specifically engineered for geometric uniformity, supports Matryoshka representation learning (truncatable dimensions).

**2. SDM/continuous-vector mismatch**
Kanerva's SDM designed around binary Hamming spaces. Plugging dense continuous SONAR vectors in requires non-trivial adaptation. **Kanerva Machine** (Wu et al., DeepMind, 2018) directly addresses this — reformulates SDM for continuous spaces using variational inference. Must read before any SDM implementation.

**3. Macro-attractor collapse (long-term risk)**
Repeated return to core themes causes attractor merging when local density exceeds threshold relative to beta. Density-dependent gating helps but doesn't fully solve. Two references:
- **Hopfield unlearning** (Hopfield, Feinstein, Palmer, 1983) — periodic "sleep" phases where network settles from random noise and applies negative learning rates to flatten spurious attractors.
- **Benna-Fusi complex synapses** (Nature Neuroscience, 2016) — maintaining landscape over years requires cascading hidden variables at exponentially slower timescales, not single scalar attractor depths.

**4. BCM Theory upgrades density-dependent gating**
Naive density-dependent thresholds cause noise amplification in empty regions of 1024D space. **BCM Theory** (Bienenstock, Cooper, Munro, 1982) provides mathematical fix: sliding threshold as nonlinear function of *temporal average* activity, not just spatial density. Threshold drops to allow learning, snaps back up after learning to prevent noise capture. Direct improvement to incorporate.

**5. SONAR negation collapse and pragmatic stripping**
"I love my job" and "I hate my job" land in same basin — translation-aligned spaces strip valence. Questions and statements geometrically identical. Candidate fix: **propositional extraction** — LM preprocesses raw input into atomic propositions before embedding. Breaks "raw experience" fast-store rule but solves negation collapse, illocutionary stripping, and granularity mismatch simultaneously. Hold as candidate, don't commit yet.

**6. Readout mechanism (closes open question #5)**
Engagement = **Free Energy** (readable from Hopfield update equation's denominator). Resolution = **Shannon entropy of softmax attention distribution** over stored memories. Low entropy = sharp convergence (one attractor wins); high entropy = sustained superposition. Math exists in Liu et al. (NeurIPS 2020) and falls out of Ramsauer update rule. No observer module needed. This is potentially a clean closure of open question #5.

### Concrete Scale Numbers (Implementation Reference)

| Parameter | Value |
|-----------|-------|
| Hopfield capacity (14GB budget) | ~1.5-2M patterns before OOM |
| At 500 consolidations/day | 8+ years of memory — capacity not a bottleneck |
| SDM safe capacity | 1,800-9,000 patterns before noise wash |
| Hopfield update step (1M patterns) | ~26ms (memory-bandwidth bound, not compute) |
| Fast convergence (3-5 iterations) | 100-150ms — snappy interactive |
| Deliberation (30-50 iterations) | ~1.3s — noticeable but acceptable "thinking" pause |
| Precision requirement | Must use FP32 (FP16 overflows on exp(100)) |
| Must use LogSumExp trick | Naive softmax will produce NaN at scale |

### Mischaracterizations (Noted, Not Actionable)

- **Anti-homunculus "contradiction"**: Gemini calls the consolidation gate a homunculus. It's a reactive threshold, not a deliberative controller. A thermostat is not a homunculus.
- **Saddle point freeze**: True saddle points are measure-zero in continuous spaces. Any perturbation breaks symmetry. What actually happens is very slow convergence — which is the intended behavior (engagement-dependent convergence rate). A practical timeout is reasonable engineering, not a fundamental flaw.
- **LM domination**: Real tension but applies equally to every system that uses a pre-trained LM, including Gemini's own alternative architecture. Empirical question, not architectural flaw.
- **Sycophantic collapse**: Misframes the meta-learning signal as minimizing divergence. The system should adjust *categories* of surprise, never suppress the surprise signal itself. Implementation constraint, not architecture flaw.
- **Temporal binding**: Partially valid — individual vectors don't encode sequences. But landscape topology from related consolidations does encode relational structure. Whether explicit sequential binding is needed remains genuinely open.

### Alternative Architecture (Synaptic Dreaming) — Contrast Value Only

Gemini proposed a LoRA-based approach where personality lives in weight modifications and consolidation happens via nightly fine-tuning with generative replay. Elegantly solves translation gap and temporal binding. But violates core design principles: memory becomes uninspectable weight changes, requires fine-tuning, personality is a black box. Valuable as negative example — clarifies what the project is NOT. Highlights translation gap and temporal binding as problems that must be solved explicitly rather than gotten for free.

## New References to Read

| Reference | Why |
|-----------|-----|
| Wu et al. (2018) — "The Kanerva Machine" (DeepMind) | Continuous-space SDM via variational inference. Must read before SDM implementation. |
| Bienenstock, Cooper, Munro (1982) — BCM Theory | Sliding-threshold plasticity. Direct upgrade to density-dependent gating. |
| Hopfield, Feinstein, Palmer (1983) — Unlearning | Active maintenance of energy landscape via sleep-phase negative learning. |
| Benna & Fusi (2016) — Complex synapses (Nature Neuroscience) | Multi-timescale memory maintenance. Prevents catastrophic interference over years. |
| Liu et al. (NeurIPS 2020) — Energy-based OOD Detection | Free Energy as engagement readout. May close open question #5. |
| Abraham & Bear (1996) — Metaplasticity | Plasticity of synaptic plasticity. Companion to BCM. |
| Nomic Embed documentation | Alternative embedding space if SONAR anisotropy confirmed. |

## What Remains Open

Previous open questions (sessions 7-8) plus refinements from this review:

1. Per-source priors formation — untouched.
2. Where raw snags originate — untouched.
3. Whether engagement-dependent convergence rate needs explicit engineering — untouched but strengthened by CTM validation.
4. SONAR geometry edge cases — now upgraded to **SONAR anisotropy as critical risk** requiring empirical testing.
5. Readout mechanism for resolution/engagement — **candidate solution identified** (Free Energy + softmax entropy). Needs validation against Hopfield dynamics.
6. **NEW: Temporal/sequential binding** — whether landscape topology is sufficient or explicit chaining mechanism needed.
7. **NEW: SDM adaptation** — Kanerva Machine paper required before implementation.
8. **NEW: Long-term landscape maintenance** — unlearning/sleep phase and Benna-Fusi cascading timescales.

## Priority for Next Session

1. **SONAR anisotropy empirical test** — encode random sentences, compute pairwise similarities, determine if cone effect is real.
2. **Free Energy + Softmax Entropy readout** — work through the math against the Ramsauer update rule to confirm this closes the readout question.
3. **BCM sliding threshold** — formalize upgraded density-dependent gating.
