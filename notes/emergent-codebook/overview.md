---
date: 2026-05-04
project: personal-ai
tags:
  - overview
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
---

# Emergent Codebook — Overview

A thesis-preserving embedding architecture for the personal-ai project. This folder contains the design and experimental plan for an embedding substrate that supports semantic similarity, structural composition, and Hopfield-compatible retrieval without compromising the bottom-up emergence thesis.

## Why this exists

The personal-ai thesis is that rules emerge bottom-up from consolidation of experience — the landscape geometry IS the rule, grown from what's been seen, not specified externally. Adding structural reasoning to the system was identified as a needed capability (see `notes/2026-05-03-hdc-kona-embedding-space.md` for the requirements analysis). The naive path — using off-the-shelf VSA tooling with imposed role schemata — would compromise the thesis by introducing externally-specified structure.

The emergent-codebook approach resolves this. Substrate operations (binding, bundling, similarity) are universal — like physical laws of the representation space. But codebook contents emerge through experience. Roles, fillers, and stable patterns are discovered via consolidation, not declared.

## Architectural framing

The system is a **contextual-completion architecture, not a sequence-prediction architecture**. Modern Hopfield networks are associative memories — pattern completion is what they do natively. The codebook's job is to support retrieval: given a partial cue (current context), surface relevant content from accumulated experience. "What does this remind me of?" not "what comes next?"

This framing is the cleanest statement of what the personal-ai thesis has wanted from the start. The briefing's claim that "true thinking requires the ability to remember and reflect on past instances" is a contextual-completion claim, not a sequence-prediction claim. The label was wrong; the architecture was right. Masked-token training (pattern completion) is the natural prediction objective; retrieval relevance is the natural evaluation metric.

## Navigation

- `literature-and-principles.md` — What's close in the literature and the five design principles
- `mvp-architecture.md` — Concrete MVP design, multimodal extension considerations, substrate notes (qFHRR)
- `experimental-progression.md` — Six-phase build plan with success criteria and cross-phase failure modes
- `phase-3-deep-dive.md` — Detailed mechanics for Phase 3 (codebook growth), where the most design risk lives
- `llm-integration.md` — How the codebook integrates with the LLM-as-voice via the workspace, Phase 6 architecture

## Current status

Design phase complete. No code written. Phase 3 has been examined in depth as the highest-risk phase. Key Phase 3 design decisions resolved:

- **Update rule:** two-pathway hybrid — Hebbian reinforcement on retrieval success, error-driven update on retrieval failure, gated per-experience by retrieval quality. The two pathways compute structurally different things, not the same operation at different magnitudes.
- **Context bag:** clean bundle of co-occurring atoms (no position bindings), used in the Hebbian pathway to drift atoms toward their successful-retrieval neighborhoods.
- **Error-driven update:** atoms in failed cues drift toward the corresponding atoms in correct retrieval targets, flowing the error signal from the bundle level down to constituent atoms.
- **Bimodality tracking:** persistent bimodal distribution of an atom's context bags signals polysemy; tracked in Phase 3, used in Phase 5 for atom splitting. Bimodality (not raw variance) is the right signal — function words have high variance but unimodal distributions; polysemous words have bimodal distributions.
- **Phase 2 baseline:** both next-token and masked-token prediction tested as a design comparison; the difference informs Phase 3's primary objective.

Ready to begin Phase 1 (substrate validation) when Dylan signals readiness to build.

## Key architectural commitments

- Substrate is universal; content emerges. Binding, bundling, similarity, permutation are fixed up front and never change.
- Codebooks grow from experience. Atoms are allocated, refined, and decay through consolidation dynamics, not direct specification.
- The bind-versus-bundle distinction is itself learned from data, not declared.
- Substrate operations are kept behind a swappable interface so that improvements in the underlying VSA implementation (e.g., qFHRR) can drop in later without touching downstream code.
- All five principles (substrate-content separation, growing codebooks, compositional binding from co-occurrence, hierarchical compression, prediction as universal training signal) operate in the same system on the same vectors.
- BPE provides permanent input *segmentation*; the codebook provides *meaning* that drifts. Segmentation layer is fixed; meaning layer is plastic. (Biological analogy: phoneme parsing is stable across adulthood, word meanings shift with experience.)

## Relationship to existing architecture

The emergent codebook is a substrate replacement / augmentation. Hopfield settling, SDM long-term storage, consolidation, replay, density-dependent gating, and the dual consolidation signal all continue to work — they operate on the new vectors instead of SONAR vectors. The existing components are not replaced; they are given a richer substrate to operate on.

The LLM-as-voice role from the briefing is preserved via Option A integration. The workspace becomes the bridge between the codebook (memory) and the LLM (language interface). See `llm-integration.md`.

## Why this is novel

No published system combines emergent codebooks with VSA-substrate operations and Hopfield-compatible retrieval at the scale this project operates. Three close cousins (Eugenio's Self-Organizing Language, Hyperseed, predictive coding networks) each capture some pieces. The integration is the contribution, and the alignment with bottom-up emergence is what makes this version distinct from existing VSA work that imposes codebooks externally.
