---
date: 2026-05-11
project: personal-ai
tags:
  - source
  - subject/personal-ai
  - subject/cognitive-architecture
  - project/personal-ai
---

# Predictive Associative Memory — Dury (Feb 2026, Mar 2026)

Two related papers from Jason Dury (Independent Researcher, Eridos AI) added to the reading queue. Discovered via a stray reference in an off-vault Claude conversation about Kohonen 1982; the reference was initially suspected to be fabricated and turned out to be real. Both papers operate on temporal co-occurrence as the training signal but in different regimes — the first targets faithful episodic recall, the second targets concept formation under compression.

## Paper 1 — Predictive Associative Memory: Retrieval Beyond Similarity Through Temporal Co-occurrence

- **Citation:** Dury, J. (2026). *Predictive Associative Memory: Retrieval Beyond Similarity Through Temporal Co-occurrence.* arXiv:2602.11322v1, 11 Feb 2026.
- **Link:** https://arxiv.org/abs/2602.11322
- **Code:** https://github.com/EridosAI/PAM-Benchmark

### Core claim

Useful memories are linked by temporal co-occurrence, not similarity. Stairs and a slip share no representational features but reliably evoke each other because they were experienced within the same temporal window. Standard similarity-based retrieval (RAG, dense passage retrieval, modern Hopfield networks) cannot recover these cross-modal, cross-context associations when the memories are distant in embedding space.

### Architecture

Dual JEPA. Outward predictor (standard JEPA) predicts the next sensory state from the current one — forward prediction, semantic similarity, "world model" knowledge. Inward predictor (new) predicts which past states are associatively reachable from the current state — lateral prediction, temporal association, episodic memory. Both operate over a shared embedding space but predict different targets.

The Inward predictor is trained on temporal co-occurrence pairs: for each state s(t), the training targets are states s(t′) within a temporal window τ. Trained with InfoNCE contrastive loss against in-batch negatives. A 4-layer MLP (2.36M params) with residual connections and L2 normalization on output.

Multi-hop retrieval is iterated prediction. Creative recombination across separate trajectories requires entity persistence across episodes — the synthetic benchmark in this paper doesn't provide it, so creative bridging remains a theoretical prediction.

Section 3.10 ("Similarity × Association = Specificity") frames episodic specificity as the intersection of the two channels: similarity alone returns all drills, association alone returns everything from Tuesday, the intersection returns *that drill on Tuesday*.

### Key empirical results

Synthetic world: 20 rooms in 128-dimensional embedding space, 50 objects, 500 trajectories of 100 timesteps, 242,264 temporal associations.

- Association Precision@1 = 0.970 (cosine baseline: 0.000)
- Cross-Boundary Recall@20 = 0.421 (cosine: 0.000) — cross-room pairs where similarity provides zero signal
- Discrimination AUC overall = 0.916, cross-room = 0.849 (cosine cross-room: 0.503, chance)
- Similarity-matched negatives AUC = 0.848 vs cosine 0.732 — predictor discriminates true associates from similar-but-not-associated distractors
- All metrics stable across training seeds (SD < 0.006) and query resamples (SD ≤ 0.012)

### The key control

**Temporal shuffle ablation.** Randomly permute temporal ordering within each trajectory while preserving all state embeddings. This destroys temporal co-occurrence structure while keeping embedding geometry intact. Cross-boundary recall collapses by 90% (0.421 → 0.044). Confirms the predictor learned genuine temporal structure, not artifacts of embedding geometry.

This is the cleanest published demonstration that experiential co-occurrence carries information content geometry doesn't.

### Other findings

- **Held-out query-state evaluation:** queries never used as training anchors score zero cross-boundary recall, while train-anchor queries score 0.508. Predictor learned anchor-specific mappings — episodic recall is perspective-bound. Framed as a feature, not a bug.
- **Generalisation stress test:** held-out associations score 0.023 vs train associations 0.578. Faithful recall paradigm, not retrieval generalisation. Memorisation is the correct behaviour for episodic memory.
- **Creative bridging:** synthetic world has no entity persistence → no cross-trajectory edges → oracle reaches target 0% of the time. Identified as boundary condition requiring entity persistence; flagged as ongoing work.
- **Fixed pairs outperform online sampling** (loss 0.409 vs 2.315, MRR 0.635 vs 0.423). Repeated exposure to the same association strengthens recall, mirroring complementary learning systems.

### Stated limitations

- Synthetic benchmark, geometrically clean cluster boundaries — gap may narrow on continuous noisy embeddings.
- Single-point prediction (not distribution) — inadequate for one-to-many associations as the creative bridging test showed.
- Temporal co-occurrence isolated — affect, arousal, causal inference not modeled (flagged as extension via additional input channels).
- No joint encoder training — encoder is fixed, Inward predictor is augmentation.
- Three seeds only; larger sweep would strengthen confidence.
- No comparison to temporal graph methods (PageRank, random walks, SR, temporal GNN).
- Association precision dilutes with depth (AP@1 = 0.970, AP@20 = 0.216) — predicted region broader than tight association neighbourhood.

## Paper 2 — From Topic to Transition Structure: Unsupervised Concept Discovery at Corpus Scale via Predictive Associative Memory

- **Citation:** Dury, J. (2026). *From Topic to Transition Structure: Unsupervised Concept Discovery at Corpus Scale via Predictive Associative Memory.* arXiv:2603.18420v1, 19 Mar 2026.
- **Link:** https://arxiv.org/abs/2603.18420
- **Demo:** https://eridos.ai/concept-discovery
- **Code:** https://github.com/EridosAI/PAM-Concept-Discovery

### Core claim

Embedding models group text by semantic content — *what text is about*. A different training signal — temporal co-occurrence within texts — discovers a different kind of structure: recurrent transition-structure concepts — *what text does*. Under capacity constraint, contrastive training on temporal co-occurrence extracts cross-author, cross-genre patterns that transfer to unseen texts.

### Architecture

Same family as PAM. 29.4M-parameter 4-layer MLP with GELU activations and a learned residual scalar α (converged to 0.756), operating on BGE-large-en-v1.5 1024-dim embeddings. Symmetric InfoNCE with in-batch negatives, temperature τ = 0.05, 150 total epochs. k-means clustering at six granularities (k = 50, 100, 250, 500, 1000, 2000) over association-space embeddings.

The key architectural commitment vs paper 1: training reaches only **42.75% accuracy** at epoch 150, well below capacity. The model cannot memorise 373M co-occurrence pairs into 29.4M parameters and is forced to compress across recurring patterns. Compression-under-bottleneck is positioned as analogous to hippocampal sleep replay consolidating episodes into stable neocortical representations.

### Corpus

9,766 English-language Project Gutenberg texts (10,000 requested, 234 excluded during chunking). Spans 16th to early 20th century, fiction, non-fiction, essays, drama, poetry, religious texts. Chunked into 50-token passages with 15-token overlap → 24,964,565 passages. Temporal co-occurrence pairs within ±15-chunk window → 373,296,555 unique pairs, all within-book.

### Key empirical results

**Discovered concepts span at least five categories:** narrative functions (confrontation, departure, revelation), discourse registers (cynical worldly wisdom, lyrical landscape meditation, sailor dialect), literary traditions (Russian psychological realism, American Western), scene templates (deathbed/medical crisis, formal negotiation), and subject-matter conventions (cats, witchcraft, Darwin-Huxley correspondence).

At k=100, mean cluster contains passages from 4,508 distinct books (46% of corpus), mean single-book dominance 4.0%. Confirms cross-author structure rather than author-specific artefacts.

**Direct comparison with raw BGE clustering** (similarity-based, computed on 2K-novel subset due to compute): BGE clusters group by topic (all fear passages together, all money passages together); PAM clusters group by transition structure (the moment of confrontation across Oscar Wilde, Jane Austen, Anne Brontë, Charles Dickens).

**Unseen-novel evaluation** on five canonical novels not in training corpus (Pride and Prejudice, Dracula, Frankenstein, Alice in Wonderland, War of the Worlds): PAM concentrates each novel into a selective subset of clusters (Alice: 51/100, top-5 holds 77.6%); BGE saturates nearly all clusters (Alice: 87/100, top-5 holds 32.2%). PAM tracks structural role and shifts at mode boundaries; BGE tracks topic and scatters with every vocabulary change.

**Authorial pacing signatures:** War and Peace shows long sustained blocks at every resolution; Ulysses shows episode-level style shifts at coarse resolution with sustained technique-blocks at fine resolution; Dr Jekyll and Mr Hyde shows rapid mode-switching within scenes. Different distributions of structural modes per author, visible simultaneously at multiple resolutions.

### Controls

- **Temporal shuffle (on 2K pilot, same architecture):** −95.2% cross-boundary recall. Confirms genuine co-occurrence structure.
- **Context-enriched baseline** (non-learned ±15 chunk averaging): inflates intra-cluster cosine to 0.861, collapses book diversity to 726, increases dominance to 8.1%. Symmetric averaging produces book-specific clusters, not cross-book concepts. The learned contrastive transformation extracts something simple smoothing cannot.
- **Random MLP baseline:** disperses passages near-uniformly (8,553/9,766 books per cluster). Architecture itself imposes no meaningful structure.
- **Position-in-book / token-count / book-concentration confounds:** 0/100, 2/100, and 10/100 flagged respectively. Most flagged clusters retain >1,300 distinct books.

### Theoretical claim

PAM/AAR/this work share training signal and objective but operate in different compression regimes with different emergent properties. PAM (paper 1) memorises specific associations — episodic recall — inductive transfer fails. AAR (intermediate paper, multi-hop QA retrieval, Zenodo preprint) reaches ~97% training accuracy on contingent passage-to-passage links — inductive transfer fails. This paper at 42.75% accuracy on recurring transition-structure patterns — inductive transfer succeeds because the patterns are recurrent regularities rather than contingent co-occurrences.

Compression is the key variable. Same architecture, same training signal, qualitatively different behaviour depending on the compression regime.

### Stated limitations

- Single training run, no multi-seed evaluation.
- Temporal shuffle on 2K pilot, not full 10K corpus.
- BGE baseline on 2K subset due to compute.
- Cluster labels generated post-hoc by LLM examining sample passages, not method outputs.
- No systematic compression-ratio sweep.
- English-language Project Gutenberg only, ascending-ID selection bias toward earlier-digitised works.
- Chunk size not varied.
- No formal human evaluation of cluster coherence.
- No downstream task evaluation.

## What these papers establish

1. **The temporal-co-occurrence gap is real and measurable.** Cosine on cross-boundary pairs = 0.503 (chance) vs predictor 0.849. Temporal shuffle ablation collapses 90% of the gain. This is the empirical answer to "is there a flavor of relatedness content geometry doesn't capture" — yes, and the gap is not subtle.

2. **Two-channel architecture for episodic specificity.** Similarity × Association = Specificity is a structural prediction with the same shape as decompositions implicit in the personal-ai project. PAM's specific instantiation (two distinct neural predictors over a frozen shared encoder) is not transferable to the FHRR + Hopfield substrate, but the structural insight may be.

3. **Compression regime determines inductive transfer behaviour.** Under-fitted training extracts regularities that transfer; over-fitted training memorises specifics that don't. Relevant to "generalization emerges from retrieval dynamics, not write-time baking" — though Dury's compression happens in training, not retrieval.

4. **Temporal shuffle ablation is the cleanest control.** The shape of the experiment translates to any substrate: shuffle temporal order, preserve content, measure the gap. Worth keeping in mind as a pre-committed Phase 5+ experiment.

## Where these papers don't apply

1. **Mechanism does not transfer.** Separately trained MLP with InfoNCE on explicit co-occurrence pairs assumes a frozen encoder and a predictor-on-top design. Personal-ai's substrate has no separately trained predictor — retrieval *is* the dynamics of the landscape. The information that temporal co-occurrence matters is useful evidence; the machinery for capturing it is not a candidate.

2. **Not memory-first in the project's sense.** Encoders are fixed (paper 1) or pre-trained BGE (paper 2). Both papers test associative augmentation of a fixed semantic space, not a substrate where personality lives in the consolidated memory. The differential-personality-emergence thesis is not tested or claimed.

3. **Temporal window is a fixed hyperparameter.** τ = 5 timesteps (paper 1) or 15 chunks (paper 2), both dialled. Personal-ai's standing commitment is that timescales should be dynamic properties of the substrate.

4. **Anti-homunculus filter would flag several components.** Predictor modules, retrieval thresholds ε, similarity-matched negatives, adaptive decay tied to "association density" as a separately tracked quantity, λ multiplier on negative-pair distance. May be convenient labels for things that could be re-derived as geometric dynamics, or may be genuine controllers — open question.

5. **Creative bridging not demonstrated.** Paper 1 explicitly acknowledges the synthetic world lacks entity persistence so cross-trajectory associations cannot form; transitive recombination remains a theoretical prediction of the framework.

## Open architectural question this opens for personal-ai

Where does temporal co-occurrence enter the FHRR + Hopfield substrate, given that the project rejects the separately-trained-predictor design? Candidate from the 2026-05-03 note's framing (HDC binding scope extended to temporal context): consolidation binds embeddings with a temporal-context bundle at write time, so the landscape's attractor geometry carries both flavors of relatedness in a single structure, and one settling operation reads both. Whether this passes the anti-homunculus filter and what specifically goes into the temporal_context bundle remains to be worked out — likely a Phase 5/6 architectural decision, not a Phase 2/3 blocker.

## Reading queue priority

Both papers are worth careful reading before Phase 5/6 design work begins, but neither is a Phase 2/3 prerequisite. Paper 2 is the more relevant for personal-ai because the compression-produces-concepts story is closer to where consolidation is supposed to land architecturally; paper 1 is the load-bearing empirical evidence for the gap claim itself. The AAR Zenodo preprint referenced in paper 2 (multi-hop passage retrieval, +8.6 R@5 on HotpotQA, inductive transfer fails) is the missing middle term and may be worth adding to the queue later.
