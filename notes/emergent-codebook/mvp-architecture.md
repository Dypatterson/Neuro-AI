---
date: 2026-05-05
project: personal-ai
tags:
  - notes
  - subject/cognitive-architecture
  - subject/personal-ai
  - project/personal-ai
---

# MVP Architecture (Text-Only)

The smallest version that exercises all five principles. Concrete choices, top to bottom. Designed so multimodal extension is straightforward later.

## Architectural framing

The system is a **contextual-completion architecture**, not a sequence-prediction architecture. Modern Hopfield networks are associative memories — pattern completion is their native operation. The MVP is built around this commitment: masked-token prediction as the training objective, retrieval relevance as the evaluation metric.

## Component choices

### Substrate

FHRR (Fourier Holographic Reduced Representations) at 4096 complex dimensions. 8192 real-valued degrees of freedom per vector — enough capacity for clean binding behavior, small enough to fit codebooks of reasonable size in M5 Pro RAM.

- Binding: element-wise complex multiplication
- Unbinding: multiplication by complex conjugate (algebraically exact at float32 — verified empirically)
- Bundling: per-dimension normalized element-wise sum (NOT torchhd's default — see below)
- Similarity: cosine

`torchhd` provides binding, unbinding, similarity, and random hypervector generation correctly. **Do not use `torchhd.bundle` directly** — it returns an unnormalized element-wise sum that drifts away from FHRR's unit-modulus constraint. After enough bundling operations, vectors accumulate magnitude and similarity calculations become meaningless. The substrate's `bundle` method must wrap torchhd with per-dimension renormalization: `z_j = sum_k(v_{k,j}) / |sum_k(v_{k,j})|`.

This is a hard requirement, not a stylistic preference. Discovered during the May 5 substrate validation spike — see `notes/2026-05-05-substrate-validation-spike.md` for empirical results and the bug demonstration.

**Package name:** PyPI publishes torchhd as `torch-hd` (with hyphen). Import name remains `torchhd`. `pyproject.toml` should specify `torch-hd`, not `torchhd`.

**Empirical baselines from the spike** (D=4096, M5 Pro, float32):
- Pairwise orthogonality: mean |sim| = 0.0088, std = 0.011 — matches theory (1/√(2D))
- Round-trip fidelity: exactly 1.0 across 1000 trials, no float drift
- Bundling capacity: 100% recovery at K=50 with 11× noise floor margin
- MPS speedup: 1.6× on bind, 3.6× on cosine_matrix, 4.3× on bundle (relative to CPU)

**Context bag size guidance:** The spike demonstrated 100% recovery at K=50, but signal strength scales as 1/√K (0.40 at K=5, 0.13 at K=50). Phase 3's Hebbian context bags should stay small (K≤10) by design even though the substrate supports much larger bundles. The architectural decision is "what's the right context window for consolidation," not "what's the maximum the substrate handles." Larger context bags have weaker per-atom signal for the same recovery rate.

### Input layer

Text tokenizer — start with an existing BPE tokenizer (GPT-2's is fine) so segmentation isn't being solved alongside everything else. Each token type maps to a hypervector. Initial vectors are random unit-modulus complex vectors; they get refined over time through consolidation.

**Important distinction:** BPE provides permanent input *segmentation*; the codebook provides *meaning* that drifts. Token IDs are stable (BPE consistently maps "the" → token 1234); codebook entries for those token IDs change over time. The segmentation layer is fixed; the meaning layer is plastic. Same shape biologically — phoneme parsing is stable across adulthood, word meanings shift with experience.

### Codebook

Starts empty. New token types get random hypervectors on first encounter. Each atom carries:

- usage_count
- last_used_step
- stability_score (moving variance over recent updates)
- utility_score (moving avg of contribution to correct retrievals)
- context_bag_history (rolling window for bimodality tracking)

Atoms below thresholds decay; atoms above thresholds consolidate. Soft cap on total codebook size with weakest-decay enforcement. Atoms refine through the two-pathway hybrid update rule (see `phase-3-deep-dive.md`).

### Sequence encoding

A sequence becomes a bundle of `bind(position_i, token_i)` for each position. Position vectors are themselves hypervectors — start with a fixed permutation-based encoding (each position is a permutation of a base vector) so positions are systematically related. The bundled result is a single hypervector representing the whole sequence.

### Hopfield layer

Modern Hopfield with log-sum-exp energy, single layer for MVP. Stored patterns are bundled sequences from training. Settling on a query pulls toward the most similar stored bundle.

### Prediction objective (masked-token)

Given a sequence with a masked position:

1. Encode the sequence (with the masked position represented as a special "mask" hypervector or simply omitted from the bundle).
2. The bundled hypervector is the cue.
3. Hopfield settles, retrieves the closest stored pattern.
4. Unbind by the masked position to extract the predicted hypervector for that slot.
5. Score retrieval quality by cosine similarity between predicted and actual masked-token hypervector.

The retrieval quality score gates the consolidation pathway routing — see `phase-3-deep-dive.md` for the mechanics.

Note: next-token prediction is also tested as a Phase 2 baseline comparison, but masked-token is the primary training objective from Phase 3 onward.

### Decoding (for evaluation)

For evaluation, retrieve the codebook atom most similar to the predicted hypervector. Goes through similarity in HD space rather than discrete classification.

## Deliberately left out of MVP

- No bind-versus-bundle discovery yet — positional binding is fixed (use position, bundle for co-occurrence). Phase 5 work.
- No multi-layer hierarchy yet — single Hopfield layer. Phase 4 work.
- No atom splitting yet — bimodality is tracked in Phase 3, splitting fires in Phase 5.
- No cross-modal anything. Post-Phase-6 work.
- No SDM integration yet — keep MVP standalone. Phase 6 work.
- No LLM integration yet — codebook operates standalone for Phases 1-5. See `llm-integration.md` for Phase 6 architecture.

Each gets added in later phases once the core is validated. See `experimental-progression.md`.

## Substrate-as-swappable design discipline

The substrate operations (bind, unbind, bundle, similarity, normalize) live behind an interface with one concrete implementation (torchhd's complex FHRR, with the bundle wrapper). When better substrates appear, they can drop in without touching downstream code. This is not premature abstraction — it's keeping the door open for changes that are likely to happen.

The torchhd.bundle issue immediately validates this discipline: the FHRR implementation already needs a wrapper around torchhd's bundle, and any future substrate (qFHRR, etc.) will need its own correct bundle implementation behind the same interface.

### qFHRR (Snyder et al., April 2026)

Quantized phase FHRR — encodes each dimension as a discrete phase index, enabling integer-only implementations. Reduces 64-bit complex to 3-4 bits per dimension, ~16-32x storage savings, faster on resource-constrained hardware. Preserves algebraic properties of FHRR.

Not for MVP. Three reasons:

- Three weeks old at time of writing. No follow-up work, no community vetting, no torchhd integration.
- Storage and compute savings don't matter during M5 Pro validation. The MVP fits in 24GB comfortably.
- If substrate stays swappable, qFHRR can drop in later for free. No reason to build it now.

Bookmark for Phase 6 or post-MVP optimization, especially if Pi 5 deployment becomes load-bearing.

## Multimodal extension

Adding modalities is significantly less work than the initial build, *if* the architecture is built modality-agnostic from the start.

### What changes per modality

- Input encoder (text tokenizer vs vision patcher vs audio framer)
- Prediction head (masked-token for text, masked-patch for vision, masked-frame for audio)
- Modality-specific atoms that emerge in their own codebooks

### What stays the same

- Substrate operations
- Codebook dynamics
- Binding discovery
- Hierarchy
- Hopfield settling
- SDM
- Consolidation
- Replay

### Cross-modal alignment is natively easy

When text and image co-occur (caption + photo), both modalities' encodings are present in the same experience. Binding the text codes with the image codes during shared experience produces cross-modal associations algebraically — no CLIP-style contrastive training required. The same consolidation dynamics that grow within-modality structure grow cross-modality structure when modalities co-occur.

### Discipline required upfront

To preserve modality-agnosticism through the text-only build:

- Substrate must not assume token-like atoms.
- Codebook management must handle multiple parallel codebooks rather than one global one.
- Hopfield layers must operate on hypervectors regardless of origin.
- Prediction objectives must be plugin heads, not core logic.

If maintained through text-only build, adding vision later is weeks of work (new input encoder + prediction head). If text-specificity creeps into the core, it becomes months of refactoring.

## Hardware and resource notes

Build on M5 Pro (24GB unified memory). 4096 complex hypervectors at float32 = 32KB per vector. Codebook of 50K tokens = ~1.6GB. Plus Hopfield store, plus working memory. Tight but workable. MPS backend is fully functional for FHRR operations (verified in spike); use MPS throughout for cosine_matrix-heavy operations like Hopfield settling.

Pi 5 (8GB) deployment is possible but not for MVP. Smaller substrate (2048 complex dim) or qFHRR substrate would make Pi-native operation comfortable. Defer until MVP validates.
