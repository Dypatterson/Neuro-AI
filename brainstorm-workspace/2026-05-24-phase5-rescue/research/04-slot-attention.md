# Research brief 04 — Slot attention / object-centric competitive grouping as the source of role-distinguished branches

## Angle

Phase 5's failure mode is **branch collapse**: K parallel HAM-settling branches seeded with different priors all flow to the same basin, so role-prior and content-prior conditions become indistinguishable (ΔE ≈ 0 unless a log-prior spike reshapes the retrieval logits, which is not structural retrieval). The thesis here is that **Phase 5 lacks a mechanism that makes branches push each other apart during settling**. Slot Attention (Locatello et al. NeurIPS 2020) is exactly such a mechanism: K slots are forced to specialize by a *softmax over slots* (not over keys), creating zero-sum competition for the input. No supervisor assigns slots to inputs — the competition *is* the differentiation. This is anti-homunculus-compatible by construction: every "decision" is a local normalization, not an arbitration. The brief is to investigate whether slot-style cross-K normalization can be grafted onto the existing FHRR + Modern Hopfield + emergent-codebook stack to give K branches a *reason to differ*.

## Key findings

1. **The single mechanistic change in slot attention is the normalization axis.** Slot attention is a dot-product attention layer applied iteratively (3 steps typically), with a GRU update and an MLP. The one non-cosmetic difference from standard cross-attention is that the softmax is taken across the **slot axis** for each input token, so all slots' attention weights on a given input pixel sum to 1. This is the entire source of competition. Replacing it with softmax-over-keys (standard attention) destroys specialization completely ([Locatello et al. 2020](https://arxiv.org/abs/2006.15055); [dissection](https://medium.com/@yusufshihata2006/dissecting-slot-attention-how-to-force-transformers-to-think-in-concepts-3c2ba9f60706)).

2. **Modern Hopfield retrieval uses the *other* normalization axis.** MHN softmax is over stored patterns (one row of attention per query) — that is, *softmax-over-keys*. So the project's current K-branch design has every branch independently compute softmax-over-patterns and flow to its argmax; **branches do not see each other**. This is precisely the structure slot attention deliberately avoids. The collapse observed in reports 041–061 is the *predicted* outcome of running K independent softmax-over-keys retrievals from priors that are similar in similarity-space.

3. **Slot attention is already a fixed-point operator.** [Chang, Griffiths, et al. 2022](https://arxiv.org/abs/2207.00787) ("Object Representations as Fixed Points") formalize slot attention as a fixed-point iteration and use implicit differentiation. This is structurally identical to HAM settling — both are iterated, contractive, fixed-point maps. The graft is mechanically natural.

4. **There is an emerging unification.** [Energy Transformer (Hoover, Liang, Krotov et al. NeurIPS 2023)](https://arxiv.org/abs/2302.07253) interprets attention as gradient descent on a Hopfield-style energy, with attention and the MLP applied in parallel as a single energy minimization. [Hopfield-Fenchel-Young Networks (Martins et al. 2024)](https://arxiv.org/abs/2411.08590) generalize this further. None of these have explicitly fused slot-style cross-slot competition with Hopfield settling — **this is an open seam in the literature** and the project is well-placed to push it.

5. **Slot merging via optimal transport.** [Zhang et al. ICML 2023 "Unlocking Slot Attention by Changing Optimal Transport Costs"](https://arxiv.org/abs/2301.13197) reinterprets slot attention as one-step Sinkhorn and shows that more iterations more fully resolve competition. SA-MESH (Minimize Entropy of Sinkhorn) gives crisper tie-breaking. This means the competition mechanism is *tunable* by changing how aggressive the cross-slot normalization is — a hyperparameter the project could sweep.

6. **SysBinder (Singh et al. ICLR 2023, ["Neural Systematic Binder"](https://arxiv.org/abs/2211.01177)) does *factor binding* inside slots.** It alternates spatial slot binding with **factor binding within a slot**, producing block-structured slot vectors where blocks correspond to abstract factors (color, position, texture) — emergent role specialization without supervision. This is the closest existing analogue to what Phase 5 wants: roles emerging from competitive normalization, not from a homunculus.

7. **Anti-homunculus check on slot attention itself.** Nothing in slot attention picks which slot gets which object. Slots are random Gaussians at init; the softmax-over-slots normalization plus the iterative GRU update is the entire causal chain. The "selection" is the gradient flow of a competitive softmax, which is a local geometric dynamic. **Passes the filter.** Note that even slot *initialization* is unsupervised — slot attention with shared init is permutation-equivariant, and tie-breaking emerges from numerical noise. (For this reason, slot attention has a known "tie-breaking weakness" that Sinkhorn variants address, which is itself relevant — see idea 4 below.)

## Promising leads

- **Smoothing slot attention iterations** ([2025, arxiv 2508.05417](https://arxiv.org/abs/2508.05417)) — addresses the cold-start query problem by pre-heating queries with input information. Directly analogous to Phase 5's "what should the prior be?" question.
- **When Slots Compete: Slot Merging** (2026, arxiv 2603.11246) — slots that should be one entity merge via Soft-IoU; conversely, slots that should be distinct repel via the merge gradient. The repulsion mechanism is a candidate transplant.
- **Neural Production Systems** ([Didolkar, Goyal et al. NeurIPS 2021](https://arxiv.org/abs/2103.01937)) — competitive production rules selected by attention, with hard sparse routing. The competition is over *rules*, which are role-like.
- **RIMs** ([Goyal et al. ICLR 2021](https://arxiv.org/abs/1909.10893)) — K modules that compete via attention bottleneck to be updated at a given timestep. The "only top-k get updated" structure could map onto "only the branch whose flow lowers energy most contributes to the codebook update".
- **LARS-VSA** ([2405.14436](https://arxiv.org/abs/2405.14436)) — explicit fusion of VSA binding with attention; one of the only papers that puts these primitives in the same architecture.
- **"Attention as Binding" (2025, arxiv 2512.14709)** — argues self-attention *is* approximate VSA: queries/keys are role spaces, values are fillers, attention is soft unbinding. Provides the formal bridge between slot attention's softmax-over-slots and FHRR role-filler structure.

## Concrete ideas for the project

### Idea 1 — Cross-branch softmax during HAM settling ("slot-Hopfield settling")

At each settling step *t*, after every branch computes its per-pattern logits ℓ_k(p) = β · ⟨s_k^t, p⟩, **normalize across branches** rather than (or in addition to) across patterns:
  α_k(p) = softmax_k( ℓ_k(p) )    [softmax over the K-branch axis, for each pattern p]
Then each branch's update mixes patterns weighted by α_k(p) · softmax_p(ℓ_k(p)). The cross-branch softmax means a pattern that branch j is already "claiming" gets de-weighted in branch k's update. Branches are pushed into different basins by the same input substrate.

**Anti-homunculus check.** Nothing picks which branch wins which pattern; α_k(p) is computed pointwise from inner products. The differentiation is the local geometry of the softmax-over-K, identical in shape to slot attention. Passes.

**Headline-metric link.** ΔE = E_content − E_role becomes nonzero *because the role-prior branch is repelled from content-prior basins by the cross-K normalization*, not because a prior bias reshaped logits.

### Idea 2 — FHRR-binding-aware competition (the role-specific transplant)

Standard slot attention treats slots as unstructured vectors. The project's substrate has FHRR with explicit role/filler binding. **Make the cross-branch competition operate on the *unbound role residual*, not the raw state.** At each settling step, for branch k with prior p_k:
  r_k^t = unbind(s_k^t, ρ_role)    [FHRR unbinding with the role hypervector]
  α_k(p) = softmax_k( β · ⟨r_k^t, p⟩ )
Branches now compete over *role-residual similarity*, so the role-prior branch gets first claim on patterns that have meaningful role-residual structure, while content-prior branches are pushed toward patterns with strong content-residual structure. This is the SysBinder factor-binding insight grafted onto MHN settling.

**Anti-homunculus check.** Unbinding is a deterministic algebraic operation (complex conjugate elementwise mult). The competition is again a softmax-over-K. No `if role then X` logic. Passes.

### Idea 3 — Sinkhorn-iterated branch assignment

Replace the cross-branch softmax with one-step Sinkhorn (Zhang 2023). After T settling steps, run S Sinkhorn iterations on the K×N pattern-claim matrix to *fully resolve* branch-pattern competition. This gives crisper differentiation than a single softmax and addresses the well-known tie-breaking weakness of slot attention.

**Anti-homunculus check.** Sinkhorn is a fixed-point algebraic iteration on a cost matrix. No decision module. Passes.

**Predicted effect on Phase 5 metrics.** `random_lowest` should drop below 0.33 because role-prior branches are *systematically pushed out* of content basins, not merely biased away from them.

### Idea 4 — Asymmetric init from prior (slot attention's tie-break grafted)

Slot attention tie-breaks via Gaussian init noise. In Phase 5, **the prior is the init**. So the prior choice (role vs content) already breaks the symmetry — but the *settling* doesn't preserve the asymmetry because each branch independently flows to its argmax. Combine with Idea 1: cross-branch normalization **amplifies** the prior-induced asymmetry instead of erasing it. The prior is then a real causal lever rather than a soft bias that gets washed out by the MHN softmax.

### Idea 5 — Block-slot (SysBinder) decomposition of FHRR state

Decompose each branch's state into B blocks along the FHRR dimension (e.g., 8 blocks of 512 dims each). Cross-branch competition happens **per block**, so different blocks of different branches can specialize to different factors (role, content, position, schema). This is exactly SysBinder's factor binding, ported to FHRR + MHN. Drill-down: per-block ΔE reveals *which* factor dimension role-prior branches actually find lower-energy in — possibly the role block has ΔE >> magnitude floor even while the global ΔE is borderline.

**Anti-homunculus check.** Block partition is fixed at architecture time; competition within each block is the slot-attention softmax. Passes.

## Surprises

1. **The project's K-branch retrieval is the *exact ablation* slot attention warns against.** Locatello shows that softmax-over-keys (which is what MHN does) gives slots no way to differentiate. The project ran this ablation and observed the predicted collapse. Slot attention's whole reason to exist is to fix this failure mode. The literature has been carrying the answer the entire time.

2. **Slot attention's cold-start problem maps directly onto Phase 5's prior problem.** The "Smoothing Slot Attention" 2025 work and re-init/self-distillation 2025 work both address a problem the project already has — what to seed K branches with so they actually differentiate. Recent 2025 papers (RandSF.Q, SlotPi, STATM) are all variations on this theme.

3. **No paper found that explicitly fuses slot-style cross-K normalization with Hopfield settling.** Energy Transformer (2023) is the closest neighbor — it gives Hopfield-attention a unified energy — but it does *not* do softmax-over-slots. This is a small, well-defined open problem; the project could plausibly write the first paper on it.

4. **SysBinder's factor binding is the closest existing analogue to Phase 5's role-vs-content distinction.** And it works in the unsupervised setting the project needs. Worth a careful read of the Singh ICLR 2023 paper before designing Path B'.

5. **The optimal-transport perspective gives a knob.** "Resolve competition more fully" is literally "run more Sinkhorn iterations". If the project's branches are *almost* differentiating but flow-collapsing late in settling, this is a one-line fix.

## Sources

- [Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020](https://arxiv.org/abs/2006.15055)
- [Greff, van Steenkiste, Schmidhuber, "On the Binding Problem in Artificial Neural Networks", 2020](https://arxiv.org/abs/2012.05208)
- [Goyal et al., "Recurrent Independent Mechanisms", ICLR 2021](https://arxiv.org/abs/1909.10893)
- [Didolkar, Goyal et al., "Neural Production Systems", NeurIPS 2021](https://arxiv.org/abs/2103.01937)
- [Chang, Griffiths et al., "Object Representations as Fixed Points: Implicit Differentiation", NeurIPS 2022](https://arxiv.org/abs/2207.00787)
- [Singh et al., "Neural Systematic Binder" (SysBinder), ICLR 2023](https://arxiv.org/abs/2211.01177)
- [Zhang et al., "Unlocking Slot Attention by Changing Optimal Transport Costs" (SA-MESH), ICML 2023](https://arxiv.org/abs/2301.13197)
- [Hoover, Liang, Krotov et al., "Energy Transformer", NeurIPS 2023](https://arxiv.org/abs/2302.07253)
- [Martins et al., "Hopfield-Fenchel-Young Networks: A Unified Framework for Associative Memory Retrieval", 2024](https://arxiv.org/abs/2411.08590)
- [LARS-VSA: A Vector Symbolic Architecture For Learning with Abstract Rules, 2024](https://arxiv.org/abs/2405.14436)
- [Smoothing Slot Attention Iterations and Recurrences, 2025](https://arxiv.org/abs/2508.05417)
- [Slot Attention with Re-Initialization and Self-Distillation, 2025](https://arxiv.org/abs/2507.23755)
- [When Slots Compete: Slot Merging in Object-Centric Learning, 2026](https://arxiv.org/abs/2603.11246)
- ["Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning", 2025](https://arxiv.org/abs/2512.14709)
- [Conditional Object-Centric Learning from Video (SAVi) project page](https://slot-attention-video.github.io/)
- [Aditya Mehrotra, "An Introduction to Slot Attention" (explainer)](https://adityamehrotra.ca/blog/Slot-Attention/)
- [Aniket Didolkar's homepage](https://aniket-didolkar.github.io/)
