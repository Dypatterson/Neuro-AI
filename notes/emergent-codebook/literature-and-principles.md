---
date: 2026-05-04
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
---

# Literature Landscape and Design Principles

## What's close in the literature

No published work meets the full bar — semantic preservation, structural composition, Hopfield-compatibility, and codebook emergence from experience. Three clusters come closest.

### Self-Organizing Language (Eugenio, June 2025)

Hierarchical Hopfield memory chains acting as compositional short-term memory and dynamic retokenizer. Builds projection tensors that glue symbol sequences into multi-scale tokens via local event-driven learning, with no predefined tokens and no supervised training. Tokens, grammar, and reasoning arise as compressed memory traces within an unsupervised Hopfield hierarchy.

The closest single-paper alignment to the thesis. Untested at scale; oriented toward language specifically rather than the general substrate this project would need. Worth reading carefully.

### Emergent VSA codebooks (Hyperseed, SOM-VSA spatiotemporal work)

Use VSA operations (binding, bundling) as substrate and learn the codebook from data via unsupervised competition. Patterns stabilize through repeated co-occurrence; stable patterns become atoms; codebook grows.

Right shape for "atoms emerge from experience" but published systems are small-scale demonstrations on synthetic or benchmark data. Proves the principle works; doesn't prove it scales to the representational richness this project needs.

### Predictive coding networks

Hierarchical PCNs trained with purely unsupervised prediction objectives develop emergent factor-like representations. Connects directly to Free Energy Principle framing.

Don't natively give VSA-style binding operations, but combining hierarchical predictive coding with VSA substrate is a real research direction (Taniguchi's *Collective Predictive Coding* explicitly bridges symbol emergence and free energy minimization). The most theoretically grounded route; cleanest fit to the existing intellectual frame.

### What none provide

A tested, deployable embedding model that takes arbitrary inputs and produces structured hypervectors with emergent codebooks, semantic preservation, and compatibility with modern Hopfield settling at the scale this project operates. The work is in the integration.

## The five design principles

### 1. Substrate-content separation

The vector algebra (binding, bundling, permutation, similarity) is fixed up front — these are like physical laws of the representation space. They are not "imposed structure" any more than the geometry of Euclidean space is imposed structure. Pick FHRR or similar, commit to the operations, never change them.

### 2. Growing codebooks

Atoms enter the system through experience, not by declaration. Simplest mechanism: when an experience produces residual prediction error that current atoms can't explain, a new atom is allocated and bound to whatever context it appeared in. Stable atoms accumulate; unstable ones decay. This is consolidation applied to the codebook itself rather than just to memories — same dynamic, one level lower.

### 3. Compositional binding from co-occurrence

Whether two atoms get bound (as role-filler) versus bundled (as superposition) is a learned distinction, not a declared one. Patterns that consistently appear together with consistent relative roles get bound; patterns that appear as alternatives in similar contexts get bundled. The system discovers binding-versus-bundling through statistical structure of experience.

This is the hardest piece to design well; most of the research effort lives here.

### 4. Hierarchical compression

Multi-scale structure emerges from chained Hopfield layers where each layer's attractors become the next layer's atoms. Lower layers represent fine-grained patterns; higher layers represent compressed regularities over those patterns. This is how the rule-vs-example distinction appears naturally — rules are what consolidates at higher hierarchical levels, examples at lower.

### 5. Prediction as the universal training signal

The system learns by trying to predict — next state, masked content, future observations. There's no labeled training data, no supervised structure. The encoder, the codebook, the binding decisions, all of it gets shaped by how well it supports prediction. Connects directly to active inference / Free Energy Principle.

## Why this preserves the bottom-up thesis

The original concern: imposing structural primitives (role schemata, codebook atoms) compromises the "rules emerge from experience" thesis.

The resolution: substrate operations and content atoms are different layers. Binding-as-substrate is no more a concession than Hopfield-settling-as-substrate — both are dynamics primitives. Specific role atoms, however, must be allowed to emerge. Fixed templates would be the concession; consolidation-grown codebooks preserve the thesis at every layer that matters.

The biological analogy actually supports rather than weakens the thesis. Humans aren't blank slates — we have evolved priors about parsing the world (agent/patient distinction, object/property distinction, spatial relations) that exist before any specific learning. That doesn't make human cognition less "bottom-up emergence"; it means the substrate has structure that emergence operates on. Substrate priors are not the same as rule specification.

## Reading queue

Priority order for understanding the literature this design draws from:

1. Eugenio (June 2025) — *Objective-Free Local Learning and Emergent Language Structure in Thinking Machines*. Closest single-paper match.
2. Osipov et al. — *Hyperseed: Unsupervised Learning with Vector Symbolic Architectures*. Emergent VSA codebooks.
3. Taniguchi — *Collective Predictive Coding hypothesis*. Bridge between symbol emergence and free energy.
4. Yeung et al. (2024) — Modern Hopfield resonator with log-sum-exp energy. Relevant for Phase 4-5 hierarchical work.
5. Snyder et al. (April 2026) — *qFHRR*. Bookmark for post-MVP optimization, especially Pi 5 deployment.
