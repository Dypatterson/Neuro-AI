---
date: 2026-04-22
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
---

# Substrate Resolution — SONAR + Hopfield + SDM

## Entry Point

Dylan stumbled on two papers from Meta FAIR:

1. **Large Concept Models** (LCM team, Dec 2024, arXiv 2412.08821) — autoregressive prediction of sentence-level embeddings in SONAR space, not tokens. Explored MSE regression, diffusion-based generation, and quantized SONAR. Diffusion variant scaled to 7B params. Key framing: "concepts are language- and modality-agnostic and represent a higher level idea or action in a flow."
2. **Unified Vision–Language Modeling via Concept Space Alignment** (Qiu et al., ICLR 2026, arXiv 2603.01096) — extends SONAR to image and video via post-hoc alignment of a vision encoder (V-SONAR). Striking result: LCM trained only on English text can zero-shot process visual embeddings encoded by V-SONAR.

## What These Papers Validate

Several of Dylan's instincts from sessions 1–4 are now empirically demonstrated:

- **"Query belongs to the idea, not the token"** (session 1) — LCM literally operates at concept/sentence granularity, not token level. Same dissatisfaction with token-level processing that Dylan articulated.
- **Modality-agnostic shared space** — V-SONAR's post-hoc alignment shows you can build the shared representational surface incrementally by aligning new modalities into an existing space. Closer to "stage where perception manifests" than joint-training multimodal architectures.
- **The JEPA connection** — LCM paper explicitly notes: unlike JEPA (which emphasizes learning a representation space), LCM focuses on prediction within an existing space. Maps onto stage/landscape split: JEPA builds the landscape; LCM runs the loop on a pre-built stage.

## What These Papers Don't Address

Everything Dylan actually cares about: persistent memory, consolidation, personality emergence, dual-store dynamics, meta-loops, per-source priors. LCM is a concept-predicting machine rather than a token-predicting machine — the granularity changed, the fundamental operation didn't. The embedding space is traversed but never written to by experience. The landscape is frozen.

## The Three-Piece Substrate

The session's central move: SONAR, Hopfield, and SDM are not competing for the same slot. They solve three different sub-problems.

| Piece | Role | What it does |
|-------|------|-------------|
| **SONAR** | Encoding | The language everything is written in. Defines geometry of the space — what vectors look like, what "nearby" means, how to get back to text/images for output. Every stored and retrieved item is a SONAR vector. |
| **Hopfield** | Retrieval dynamics | Energy-based settling. Given a query, the system lands on an answer via attractor dynamics. Results can be blends of stored patterns (creativity-through-retrieval). The energy landscape IS the slow-consolidated terrain. Shape of this surface = personality. |
| **SDM** | Storage mechanics | How patterns are physically organized and written. Fixed hard locations in high-dimensional space. Writes activate a neighborhood (not just one slot). Reads return a superposition of everything stored nearby. Naturally sparse, noise-tolerant, graceful degradation. |

## Mapping to Stage / Landscape / Fast Store

| Architecture Layer | Substrate Piece | Behavior |
|-------------------|----------------|----------|
| **Stage** | SONAR encoding | Current perception encoded into SONAR. Transient. Overwritten every moment. |
| **Landscape** | Hopfield energy surface over SONAR embeddings | Stored patterns are SONAR embeddings that survived consolidation. Retrieval = settling into attractor that may blend multiple stored experiences. Shape carved by consolidation history = personality. |
| **Fast ephemeral store** | SDM buffer of SONAR embeddings | High-bandwidth writes of everything that happens. No filtering. Raw snags live here before meta-loop gating. SDM's sparse activation prevents corruption of existing patterns during fast writes. |

**Consolidation** = transfer from SDM to Hopfield network. Dual signal (surprise + reinforcement) gates what moves. When something survives, it becomes a new stored pattern in the Hopfield network, reshaping the energy surface, changing how all future retrieval works. Experience alters personality by adding attractors that warp the terrain.

## The "Cells" Insight

Dylan's grounding move: the human brain is made of cells. Same electrochemical substrate everywhere. What makes hippocampus different from cortex isn't different material — it's connectivity patterns, local circuit architecture, timescales of plasticity, gating rules.

Consequence: sharing SONAR as the encoding across all three layers doesn't flatten the distinction between them. The stage is transient because nothing holds it in place. The landscape is slow because the Hopfield surface only changes when something survives the gates. The SDM buffer is fast because it writes everything with no filter. Three different temporal dynamics, three different structural roles, one shared representational material.

Practical payoff: meta-loops can operate across all three layers without translation. Self-check ("did I misread, or is it genuinely incoherent?") compares stage encoding against landscape attractors — same kind of object, so the comparison is pure geometry (distance, angle, overlap). No adapter layer needed.

## What This Resolves

- **Open question #4 from the briefing** ("How exotic does the substrate need to be?") — answered. Not a single exotic substrate, but a composition of three well-understood pieces, each doing what it's best at.
- **Substrate homogeneity concern** — dissolved by the cells insight. Shared material, differentiated dynamics.
- **Distance to something touchable** — significantly reduced. SONAR is open-source with encoders and decoders, runs on Dylan's hardware. SDM and Hopfield are implementable from the literature. The novel contribution is the combination + gating + consolidation dynamics, not the substrate pieces.

## What Remains Open

The substrate is resolved. The gating rules, meta-loop mechanics, how per-source priors form and get read, and anti-collapse dynamics remain genuinely open — and these are the questions that should stay open because they're what makes this a companion rather than a retrieval engine.

Specific next question with real teeth: **what does writing into the Hopfield landscape actually look like when the substrate is SONAR?** A SONAR embedding survives the gates and needs to move from SDM to Hopfield. Does it become a new attractor? Deform an existing one? Does proximity in SONAR space determine merging vs. new valley?

## References

- LCM team et al. (2024). *Large Concept Models: Language Modeling in a Sentence Representation Space.* arXiv:2412.08821. Code: github.com/facebookresearch/large_concept_model
- Qiu et al. (2026). *Unified Vision–Language Modeling via Concept Space Alignment.* ICLR 2026. arXiv:2603.01096.
- Ramsauer et al. (2020). *Hopfield Networks is All You Need.* (already on reading list)
- Kanerva (1988). *Sparse Distributed Memory.* (already on reading list)
