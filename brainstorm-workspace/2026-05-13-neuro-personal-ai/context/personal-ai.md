# Personal AI Project Context Summary

**Date:** 2026-05-13  
**Phase:** Transitioning from Phase 1 (complete) → Phase 2 (probes complete, full driver pending)  
**Repository:** `/Users/dypatterson/projects/personal-ai`

---

## What This Is

Personal AI is a memory-first AI companion architecture where personality emerges from consolidation of lived experience, not from the base model. The thesis: the same LLM paired with different users' memories should become genuinely different companions through differential consolidation alone. The system uses an **emergent codebook** — a Vector Symbolic Architecture (FHRR) substrate with universally-fixed operations (binding, bundling, similarity, permutation) and contents that grow from experience through Hopfield-mediated retrieval and consolidation dynamics. It's fundamentally a **contextual-completion architecture** (pattern completion via associative memory), not a sequence-prediction architecture.

---

## Key Themes and Concepts

### Architectural Philosophy
- **Substrate-Content Separation:** Vector algebra (FHRR operations) is fixed and universal — like physical laws of the representation space. Codebook atoms, roles, patterns, and schemas emerge bottom-up through consolidation, never declared top-down.
- **Swappable Substrate:** All operations live behind an abstract `Substrate` interface so future implementations (qFHRR, learned projections) can drop in without touching downstream code.
- **Modality-Agnostic Core:** MVP is text-only, but substrate/codebook/retrieval layers make no token-specific assumptions. Multimodal is a future phase.
- **Contextual Completion, Not Sequence Prediction:** Modern Hopfield networks are associative memories. The system retrieves "what does this remind me of?" not "what comes next?" — masked-token training is the natural objective.

### Technical Foundation
- **FHRR (Fourier Holographic Reduced Representation):** Complex-valued vectors at D=4096 (8192 real DOF). Implemented via `torchhd` library.
- **Vector Operations:**
  - `bind(a, b)`: element-wise complex multiplication (associative operation)
  - `unbind(bound, b)`: inverse of bind (recovers original)
  - `bundle(vectors)`: element-wise sum followed by **per-dimension renormalization to unit modulus** (critical: `torchhd.bundle` is incorrect, has custom wrapper)
  - `permute(v, shifts)`: cyclic shift of dimensions for position encoding
  - `similarity(a, b)`: cosine similarity (via real interpretation of complex vectors)
  - `cosine_matrix(a, b)`: pairwise similarity matrix (bottleneck for retrieval)

- **Hopfield Retrieval:** Modern continuous Hopfield update rule (`ξ_new = X @ softmax(β * X^T @ ξ)`). Settles cues toward stored patterns via iterative updates. Inverse temperature β controls soft-blending vs sharp-retrieval regimes.

### Consolidation Mechanics (Phase 3+)
- **Dual-pathway update:** Hebbian reinforcement on retrieval success + error-driven update on failure, gated by retrieval quality
- **Context bags:** Clean bundle of co-occurring atoms (no position bindings), used in Hebbian pathway
- **Bimodal detection:** Tracks polysemy via context-bag bimodality, feeds Phase 5 atom splitting

---

## Current State

### Completed: Phase 1 (Substrate Validation)
**Deliverable:** Passing test suite + markdown report with empirical baselines

✅ **Substrate Interface** (`substrate/base.py`): Abstract `Substrate` ABC with methods for all VSA primitives
✅ **FHRR Implementation** (`substrate/fhrr.py`): Wraps `torchhd` with custom `bundle` normalization
✅ **Encoding Module** (`encoding/sequence.py`): Position encoding, sequence bundling, decoding primitives
✅ **Test Suite** (5 tests in `tests/`):
  1. Round-trip fidelity (bind/unbind recovery): **mean 1.0000, min 1.0000** (algebraically exact)
  2. Quasi-orthogonality (1000 random vectors): **mean sim ≈ 0.00002, std 0.011** (theory: 1/√8192 ≈ 0.011)
  3. Bundling capacity (K=2 to 50): **100% recovery up to K=50, signal ~1/√K**
  4. Structured retrieval (5-node graph): **All 5 queries correct, SNR ≈36×**
  5. Computational characteristics: **MPS speedup 1.6–4.3× over CPU** (cosine_matrix bottleneck: 2.7ms on MPS)

✅ **Validation Report** (`experiments/01_phase1_validation.md`): Empirical baselines captured; Phase 1 gate cleared

**Key Finding:** FHRR substrate is sound. Fidelity is algebraically exact (not approximate). Per-dimension bundling normalization is essential and must be explicit — this is why a custom wrapper exists.

### In Progress: Phase 2 (Hopfield Retrieval Baseline)
**Goal:** Measure masked-token vs next-token prediction on FHRR-encoded sequences; determine which retrieval objective is architecturally stronger

**Completed Probes:**
- **02a (β-sweep, May 7):** Identified disjoint regime structure at D=4096
  - Soft-blend regime (β ≤ 0.001): K-ambiguous cues show per-dim modulus proportional to K, but retrieval accuracy = 0% (softmax too diffuse)
  - Sharp-retrieval regime (β ≥ 0.01): Retrieval accuracy = 100% on memorization, but all cues commit to single patterns (K-signal erased)
  - **Recommendation:** Run finer β sweep in [0.001, 0.01] to find crossover where both retrieval and K-graded modulus coexist

- **02b (K-ambiguous trajectories, May 8):** Synthetic landscape with K-ambiguous cues at K ∈ {2, 5, 20}
  - K=2 at any β: 2/3 groups commit to single patterns, 1/3 stays near symmetric blend (float32 rounding in softmax sum breaks symmetry)
  - K≥5: All groups commit across all β (larger K amplifies weight asymmetry)
  - **Conclusion:** Settled-state modulus is NOT a reliable ambiguity signal. Ambiguity lives in retrieval **distribution** (softmax entropy, top-2 margin), not in vector magnitude.

**Full Driver Status:** Matrix of 540 experimental conditions (asymmetric: 486 masked-token, 54 next-token) is specified but not yet run. Awaiting Phase 2 Session 3.

**Empirical Baselines (Phase 2 Context):**
- Vocabulary: ~10,000 words (target cutoff from WikiText-2 frequency distribution; exact TBD)
- Codebook: Random unit-modulus FHRR vectors, one per vocabulary entry
- Landscape sizes: Small (~500), Medium (~5,000), Large (~50,000) non-overlapping window sequences
- Window sizes: 8, 16, 32 tokens (balance between encoding SNR and bundle capacity)
- Masking patterns: 1, 2, or 3 contiguous masks at edge/center/end-of-window (masked-token only)
- β values: {0.01, 0.1, 1.0} after recalibration (original {1, 10, 100} was in wrong regime)
- Retrieval conditions: Memorization (cues from stored landscape), Generalization (cues from held-out split)

**Per-Retrieval Metrics (02a/02b findings feed into 02 full driver):**
- Trajectory: softmax entropy, top-2 margin, per-dim modulus mean at each iteration
- Convergence: iteration count, energy, max stored-pattern cosine, modulus std
- Decode: nearest codebook atom, top-2 codebook cosine gap
- Outcome: binary correctness (1 if retrieved correct token, 0 otherwise)

### Deferred: Phase 3+ (Codebook Growth, Consolidation)
Phase 3 is highest-risk and has been deep-dived. Key design decisions made:
- Codebook atom allocation and drift mechanics via dual-pathway consolidation
- Bimodal distribution tracking for polysemy detection (Phase 5)
- Context-bag construction and window size (K≤10 by design, not substrate-limited)

Out of scope for Phase 2: codebook learning, Hebbian updates, atom splitting, consolidation.

---

## Open Questions and Tensions

### Empirical (Pending Phase 2 Full Driver)
1. **Which objective wins?** Does masked-token consistently outperform next-token across the experimental matrix, or only under specific conditions (window size, landscape size, β)?
2. **β regime optimization:** Can both soft-blending K-ambiguity signal AND retrieval success coexist at an intermediate β, or are they fundamentally disjoint at D=4096?
3. **Landscape size capacity:** Is there a smooth scaling to 50,000 stored sequences, or a capacity cliff? How does generalization degrade with landscape size?
4. **Window size sweet spot:** Where does the SNR-vs-capacity tradeoff balance? Shorter windows = stronger SNR (1/√K) but less context; longer windows = more signal to retrieve from but noisier cues.
5. **Memorization vs. Generalization gap:** How much does performance drop when test sequences weren't stored? This predicts Phase 3's operating conditions.

### Architectural (Settled but worth noting)
1. **Omission-style masking:** Phase 2 commits to `<MASK>`-substitution (mask position filled with MASK token, arity preserved). Omission-style (mask position dropped, arity reduced) is a deferred ablation — revisit if small-W retrieval is noise-limited.
2. **Temperature-as-geometry:** Phase 2 treats β as explicit parameter; Phase 3+ will embed it as a property of the query geometry. This spec knows the transition is coming.
3. **BPE vs word-level:** Phase 2 uses word-level (simpler, better SNR at codebook-random stage). BPE is a future refinement.

### Design Risk (Phase 3 Specific)
- **Bimodal detection robustness:** Is bimodal distribution tracking a reliable signal for polysemy in a learning system, or will it be noisy in practice? Validated via Phase 3 early experiments.
- **Consolidation convergence:** The dual-pathway update (Hebbian + error-driven) is mathematically sound but untested at scale. Phase 3 will surface this.

---

## Interesting Threads

### Emerging from Phase 2 Probes
1. **Float32 rounding as a feature:** The K=2 symmetric-blend instability (where 2/3 groups commit, 1/3 don't) is driven by float32 sum rounding. Is this noise, or does it have functional value for breaking ties in ambiguous retrievals?
2. **Entropy as the right ambiguity signal:** Session 2b found that settled-state modulus does NOT correlate with ambiguity, but softmax entropy and top-2 margin do. This suggests the Hopfield landscape geometry itself (distribution of patterns in state space) is what carries the ambiguity signal, not the magnitude of the settled state. Interesting for Phase 3 design.
3. **Regime gap at D=4096:** The soft-blend/sharp-retrieval gap at β ∈ (0.001, 0.01) is a D-specific phenomenon. Is there a dimension choice (D=2048? D=8192?) where these regimes overlap and both design goals (K-graded modulus + successful retrieval) can coexist? Or is this fundamental to Hopfield settling?

### Future Architecture Threads
1. **Multimodal extension:** Substrate is ready; Phase 2 onward must not foreclose it. All tests should conceptually work with image/audio vectors as atoms.
2. **Hierarchical compression (Phase 4):** How do you compress a large codebook while preserving retrieval semantics? VSA allows bundle-of-atoms, which hints at compositional compression.
3. **Replay and consolidation interplay:** The personal-ai thesis hinges on consolidation driving differentiation. Phase 6 will integrate replay buffer + consolidation dynamics. The interaction is untested at scale.
4. **LLM-as-voice integration:** Workspace is the bridge between codebook (memory) and LLM (language). Phase 6 design is sketched; implementation order TBD.

---

## Technical Stack and Tools

### Dependencies
- **PyTorch** 2.10+: Core tensor operations
- **torch-hd** 5.8+: FHRR primitives (note: PyPI name is `torch-hd`, import is `torchhd`)
- **NumPy** 2.0+: Diagnostics and numerical analysis
- **Matplotlib** 3.8+: Report visualization (histograms, curves, heatmaps)
- **pytest** 8.0+: Test runner
- **datasets** 2.0+: HuggingFace datasets for WikiText-2 (Phase 2+)
- **scipy** 1.10+: Confidence intervals (Phase 2+)
- **ruff** 0.8+: Formatting and linting

### Hardware
- **Primary:** Apple M5 Pro, 24GB unified memory, MPS-accelerated PyTorch
- **Operations on MPS:** All FHRR primitives work natively (no CPU fallback); cosine_matrix is 3.6× faster

### Code Structure
```
personal-ai/
├── src/pai/
│   ├── substrate/
│   │   ├── base.py           # Abstract Substrate ABC
│   │   └── fhrr.py           # FHRR implementation with custom bundle
│   ├── encoding/
│   │   └── sequence.py       # Position encoding, bundling, decoding
│   ├── hopfield/
│   │   └── retrieval.py      # Modern Hopfield retrieval module (Phase 2+)
│   ├── data/
│   │   ├── wikitext.py       # WikiText-2 loading, word-tokenization (Phase 2+)
│   │   └── windows.py        # Window sampling utilities (Phase 2+)
│   └── codebook.py           # Codebook initialization (Phase 2+)
├── tests/
│   ├── test_substrate_interface.py
│   ├── test_substrate_fhrr.py
│   ├── test_encoding.py
│   ├── test_hopfield.py     # Phase 2+
│   ├── test_data_pipeline.py # Phase 2+
│   ├── conftest.py
│   └── timing_utils.py
└── experiments/
    ├── 01_phase1_validation.py          # Phase 1 test runner
    ├── 01_phase1_validation.md          # Phase 1 report (complete)
    ├── 02a_renorm_probe.py              # Phase 2 β-sweep
    ├── 02b_dynamics_spike.py            # Phase 2 K-ambiguous trajectories
    └── 02_phase2_retrieval_baseline.py  # Full driver (pending Phase 2 Session 3)
```

### Conventions
- **Python 3.11+** with modern syntax (union types, `list[int]` not `List[int]`)
- **Type hints required** on all function signatures
- **Google-style docstrings**
- **Line length: 100** (not 88; better for numerical code)
- **No abbreviations except domain-standard** (dim, vec, idx)
- **Tests are mandatory:** Phase 1 deliverable was "passing test suite + report," not "code that works"

### Key Implementation Rules (from CLAUDE.md)
1. Define abstract interfaces before concrete implementations
2. Wrap `torchhd.bundle` with per-dimension renormalization (substrate invariant)
3. Persist random state for reproducibility (seeds saved, reloaded on re-runs)
4. All substrate operations through abstract interface, never direct `torchhd` imports
5. No fabricated empirical claims — all numbers must come from running tests
6. Treat Dylan's pushback on design as data; don't re-litigate settled architectural commitments

---

## Architectural Commitments (Non-Negotiable)

These are decisions that have been made deliberately and should not be revisited mid-session without explicit request:

1. **Substrate-content separation:** Vector algebra is fixed; codebook contents emerge
2. **Substrate-as-swappable interface:** Downstream code never imports from `substrate/fhrr.py` directly
3. **Modality-agnostic core:** MVP is text-only; multimodal must not be foreclosed
4. **Contextual-completion, not sequence-prediction:** Masked-token objective, retrieval relevance metric
5. **Per-dimension bundle normalization:** Required for FHRR correctness; custom wrapper is non-negotiable

---

## Recent Context (Last 2 Weeks)

- **May 5:** Substrate validation spike completed; FHRR confirmed sound; torchhd.bundle bug discovered
- **May 6:** Phase 1 test suite completed and passing; validation report written
- **May 7:** Phase 2 02a β-sweep probe completed; regime structure mapped
- **May 8:** Phase 2 02b K-ambiguous trajectory probe completed; entropy identified as ambiguity signal
- **May 9:** Phase 2 spec updated with Session 2 probe findings (cap-coverage, meta-stable-state diagnostics)
- **May 13:** Awaiting Phase 2 Session 3 to run full 540-condition driver

---

## What's Ready to Brainstorm

1. **Phase 3 design refinement:** Dual-pathway consolidation is sketched; Phase 2 results will refine it. What emerges from Phase 2 that changes Phase 3 assumptions?
2. **Capacity and scaling:** Phase 2 will characterize capacity ceiling. What architectural implications? Do landscape sizes of ~50K stored sequences suggest changes to Phase 3 consolidation dynamics?
3. **Multimodal extension plan:** Substrate is ready; what order makes sense for adding image/audio atoms? How does embedding space change?
4. **Replay and learning rate:** Phase 6 integrates replay buffer. What learning rates, consolidation cadence, and memory capacity feel right given Phase 2/3/4/5 findings?
5. **LLM integration pathway:** Workspace bridges codebook and LLM. What does the workspace look like? How does the LLM "see" the codebook at retrieval time?

