---
date: 2026-05-03
project: personal-ai
tags:
  - notes
  - subject/personal-ai
  - subject/cognitive-architecture
  - project/personal-ai
---

# HDC, Kona EBMs, and the Embedding Space Question

Session 11 architectural note. Browsing/literature session — no new components derived, but a significant reframing of the architecture's most important open design decision emerged from the intersection of three topics.

## Hyperdimensional Computing (HDC / Vector Symbolic Architectures)

HDC is a computational framework built on the statistical properties of high-dimensional vector spaces. In sufficiently high dimensions (thousands+), random vectors are all nearly orthogonal — creating an astronomically large space of non-interfering representations.

Three operations build everything:

- **Bundling (addition):** Superimpose multiple vectors. Result is similar to all components. Represents sets. "Is A in this bundle?" = check similarity.
- **Binding (element-wise multiplication / circular convolution):** Combine two vectors into something dissimilar to either. Represents association/structure. bind(role, filler) encodes "this role is filled by this filler." Reversible — unbind to recover filler given role.
- **Permutation (dimension shuffle):** Creates dissimilar but recoverable vector. Encodes order/sequence.

From these three, you can build representations of sequences, sets, graphs, analogies, and hierarchical structures — all as single fixed-width vectors with queryable internal structure.

### Relationship to SDM

Kanerva is a founding figure of both SDM and HDC. They're siblings:

- **SDM** = the memory architecture (how you store and retrieve by approximate address matching)
- **HDC** = the representational algebra (how you build structured, decomposable representations that can live in that memory)

SDM doesn't care what's inside the vectors it stores. HDC gives you a way to make those vectors structured and internally queryable. The storage substrate and the representation algebra solve complementary halves of the same problem.

### VaCoAl Paper

Araki (2026), arXiv:2604.11665. Proposes replacing random projection in SDM/HDC with Galois-field diffusion — deterministic algebraic operations (XOR gates, shift registers) that achieve the same quasi-orthogonality properties. Enables practical hardware implementation on standard SRAM at million-dimensional scale with O(1) similarity search.

**Relevance to personal-ai:** Primarily a hardware implementation paper — solves the "address decoder wall" for building SDM at scale on cheap silicon. Not our bottleneck at software scale on the M5. The HDC operations themselves (binding, bundling, permutation) are the more immediately useful concept, independent of VaCoAl's specific implementation trick. The collision tolerance design (hash collisions treated as acceptable noise, robust patterns survive, fragile ones decay) resonates with anti-collapse philosophy.

## Logical Intelligence / Kona 1.0

First commercial Energy-Based Reasoning Model (EBRM). Announced January 2026. Yann LeCun as founding chair of technical research board, Fields Medalist Michael Freedman as Chief of Mathematics.

### Architecture (as publicly disclosed)

1. **Thinking stage:** Problem encoded into continuous "soft thought" tensor
2. **Reasoning stage (Langevin dynamics):** Energy function E(z) measures logical consistency. Gradient descent slides the thought tensor toward low-energy state (valid solution)
3. **Generation stage:** Only when energy is minimized does the thought get decoded into text or actions

Three core properties distinguish it from LLM reasoning:
- **Non-autoregressive at trace level:** Generates complete reasoning traces simultaneously, can revise any part
- **Globally scored:** Energy evaluates end-to-end trace quality, not local next-token prediction
- **Continuous latent space:** Dense vector tokens, not discrete tokens — enables gradient-based local edits

### Convergence with personal-ai

The structural split is identical: LLM handles language interface, energy landscape handles the actual cognitive work. Same deep bet — reasoning lives in energy landscapes, not token sequences.

| | Kona | Personal-AI |
|---|---|---|
| Energy landscape origin | Trained from labeled data, frozen at deployment | Grown from experience through consolidation, continuously updated |
| Energy function meaning | "Does this state satisfy formal constraints?" | "What does this situation resemble in my experience?" |
| Settling mechanism | Langevin dynamics (gradient descent + controlled noise) | Hopfield settling (energy minimization through attractor dynamics) |
| Scope | Per-task-type constraint satisfaction | Domain-agnostic, shaped by whatever the person brings |
| Memory | None — doesn't grow with use | Core feature — personality emerges from accumulated experience |
| Target | Formal correctness for critical systems | Personal companionship and experiential reasoning |

Same mathematical object (energy landscape), different origin stories, different applications.

### Useful to take from Kona

- **Langevin dynamics as contrast to Hopfield settling:** Both are energy minimization, but Langevin adds controlled noise during descent — enables escaping shallow local minima. Potentially relevant for replay store: Langevin-style noise during re-settling could help unresolved trajectories resolve in ways pure deterministic Hopfield settling wouldn't.
- **Enso (GitHub: MVPandey/Enso):** Open-source replication of Kona using JEPA approach + Langevin dynamics. Concrete, runnable implementation. Worth examining when ready to build.
- **Validation:** Well-funded company with LeCun on the board building a commercial product on the same foundational thesis we derived independently.

## The Embedding Space Insight

The most significant outcome of the session. Emerged from this question chain:

1. Dylan asked whether the system could achieve Kona-like reasoning abilities
2. Claude initially overclaimed it couldn't — the architecture knows experiences but not rules
3. Dylan pushed back: "why can't something that knows experiences also learn rules and logic structures?"
4. Claude corrected: rules ARE what consistent experience looks like once consolidated into landscape geometry. The distinction is how rules enter (top-down from specification vs. bottom-up from experience), not whether they can exist in the landscape
5. Dylan then asked: if I explicitly told it the rules of Sudoku, would it solve puzzles the way Kona does?
6. Analysis revealed the real bottleneck: SONAR encodes semantic meaning but collapses structural/compositional information. The concept "rows can't repeat" and a specific board state violating that constraint exist in different parts of SONAR space with no gradient connecting them

### The core reframing

The gap between "knowing concepts" and "applying rules" is not a limitation of energy landscapes. It's a limitation of SONAR's semantic-only geometry.

If the embedding space preserved both:
- **Semantic similarity** (what things mean, how they relate conceptually)
- **Structural/compositional relationships** (how things behave, how parts relate to wholes, role-filler bindings)

...then the same Hopfield settling dynamics could handle experiential reasoning AND rule-based reasoning. No architectural changes needed. All existing components (Hopfield dynamics, consolidation, trajectory trace, replay, density-dependent gating, dual consolidation signal, second consolidation channel) are substrate-agnostic — they operate on whatever geometry the embedding space provides.

### The filing cabinet analogy

The landscape is a filing cabinet organized by similarity. Some items in it are photographs (flat semantic vectors — meaning without internal structure). Some are recipes (structured, decomposable vectors — binding-encoded relationships you can query and unpack). The filing cabinet doesn't know the difference. Its organizing principle (similarity/energy) works the same on both. But when you pull something out, what you can DO with it differs — you look at photographs, you follow recipes.

This is not "two layers" of the landscape. It's one landscape where the patterns stored in its valleys carry different amounts of internal structure. From the settling dynamics' perspective, they're all just attractors.

### HDC as candidate mechanism

HDC binding operations could provide the structural encoding. bind(row_3, position_4, digit_7) is a single vector — same dimensionality as any semantic embedding — but internally decomposable. The landscape stores it as just another attractor. But on retrieval, the system can unbind to ask structured queries.

Open question: can SONAR-like semantic vectors and HDC-like algebraic vectors coexist in the same space? They typically differ in dimensionality and number type (SONAR = 1024D continuous; HDC = thousands of dimensions, often binary). This compatibility question is now the most important technical question in the project.

### What this means for the architecture

**SONAR was always a placeholder.** The architecture needs an embedding space that:
1. Encodes semantic similarity (what things mean)
2. Preserves structural/compositional relationships (how things behave, role-filler structure)
3. Supports the geometric operations Hopfield settling requires (distance, blending, energy minimization)
4. Is rich enough that rules consolidated from experience create gradients the settling dynamics can follow

The choice of embedding space determines the ceiling on what the landscape can represent and what settling dynamics can achieve. Everything else is machinery that operates on whatever the embedding space gives it.

## Updated priority for next session

1. **Survey embedding spaces** that handle both semantic and structural information
2. **Investigate HDC + Hopfield compatibility** — can binding algebra and continuous energy dynamics coexist?
3. **Enso codebase** as concrete reference for energy-based reasoning implementation
4. SONAR anisotropy empirical test still relevant but now part of the larger embedding space question

## Reading queue additions

- Chen et al. (Nov 2025) — "Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought"
- MVPandey/Enso — open-source JEPA-based Kona replication
- Plate (1995) — Holographic Reduced Representations
- Gayler (2003) — Multiply-Add-Permute VSA
