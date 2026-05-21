# Brainstorm: Phase 5 graduation after the A+B → A1 → A1' → β chain

> Generated 2026-05-20 from context in:
> - `notes/`, `docs/`
> - `reports/` (040–050)
> - `src/energy_memory/` (substrate, memory, phase4 consolidation, phase5 modules)
> - `research/` and `tmp/pdf_text/` (21 bookmarked papers)
> - `experiments/`, `scripts/`

## Project Understanding

The project is a neuroscience-inspired cognitive substrate built from FHRR + Modern Hopfield + emergent codebook + replay/consolidation. It's been disciplined by the **anti-homunculus rule** from the 2026-05-09 paper synthesis: every proposed addition must be a local geometric dynamic or measurement of one — never an arbitration over them. The architectural target is contextual completion, not sequence prediction, and the user has been working through a phase plan (Phase 4 graduated on D1 meta-stable rate, Phase 5 wraps a structural-retrieval test around the post-consolidation substrate).

The Phase 5 graduation has been blocked by a **four-step debugging journey**: A+B (continuous coverage-weighted reinforcement + −α·log(d_eff) repulsion) → A1 (substrate-derived r_ema init) → A1' (max-over-others reduction operator) → β (continuous role-fidelity-weighted prior). Each step correctly closed its layer's failure mode (substrate-construction, measurement, selector). The β smoke test (report 050) surfaces a deeper problem: all 1064 atoms have **identical** role-fidelity f_i = 0.9858 at D=4096 because the FHRR crosstalk noise floor 1/√D ≈ 0.0156 sets `f_i = mean(1 − |G_jk|) ≈ 0.984` for every pattern regardless of content. The K-branch state_divergence headline is looking for a property the substrate's encoding doesn't expose at this dimension.

By every substrate-level measure A+B+A1+A1' was designed to satisfy, the substrate is the right substrate: d_eff = 35.23 preserved, discovery atoms throttled to noise floor (max effective_strength 12.11 → 0.029 across the chain), top-1 atom is now an *original* Phase-3 atom (idx 713). The K-branch divergence test was derived from the *binary-death* pre-death substrate's small-N geometry (6–12 surviving atoms with d_eff ~ 5). On the continuous-death substrate (1064 atoms, d_eff 35), the diagnostic shape doesn't fit. **The brainstorm question is therefore not "how do we patch β" but "what does this substrate actually do, and what's the right way to measure and use it?"** Five context summaries (notes+docs, reports, implementation, literature, experiments) and five research briefs (PAM/JEPA, Modern Hopfield SNR, bounded fidelity metrics, the four unexplored diagnostic-actuator pairs, cue/binding/dimension reformulations) converge on a small set of high-leverage moves.

---

## Ideas and Approaches

The ideas below are organized into three tiers by cost-to-information ratio. Each idea names the research brief that motivates it, gives a concrete formulation (no hand-waving), and flags the anti-homunculus shape. Tier 1 = hours of work; Tier 2 = ~1 day each; Tier 3 = the real architectural moves, ~2-5 days each.

### Tier 1 — Cheap, fast, high information

#### Idea 1: Cue-regime sweep (research E §3.1)

**What.** The project has fixed `binding_noise_std=0.05`, `content_distortion=0.6` from the outset and never swept them. A 36-cell grid over `binding_noise_std ∈ {0.01, 0.05, 0.10, 0.20}` × `content_distortion ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}` against the existing seed-17 A1' snapshot. Look at whether `std(f_i)` ever rises above ~0.02 in any cell.

**Why it's relevant.** Three of the four 2026-05-20 design notes are substrate-side fixes; none questions the cue regime. The β failure could be substrate-side (the substrate has no f_i variance to expose) OR operating-point-side (the cue regime is at a corner where binding-decomposition collapses). The sweep tells us which.

**How to explore it.** Add `--cue-noise-std` and `--cue-content-distortion` CLI flags to `experiments/40_phase5_branching.py --mode smoke`, write a small grid driver, run sequentially on the existing snapshot. Wall time ~20 minutes per cell × 36 cells ≈ 12 hours but trivially parallelizable; with cue n=20 it's minutes total. **Recommended first experiment regardless of which longer-term lever is picked**, because it's the cheapest possible disambiguator between substrate-side and operating-point-side failure attribution.

**Sources.** [research/E-cue-binding-dimension.md §3.1](research/E-cue-binding-dimension.md)

**Anti-homunculus shape.** Test methodology, not runtime architecture. PASS by construction.

#### Idea 2: Settling-based f_i (research B idea B2, research C #5, research E §3.3 — three angles converging on the same move)

**What.** Compute fidelity against the *retrieved* (post-settling) state, not the raw unbind. Specifically:

```
For each atom i:
  Set query = cue_synthesized_from(s_i)
  Run Hopfield settling → q_settled_i
  Compute f_i = some-measure(q_settled_i, s_i, cue)
```

Candidate measures: cosine `Re(<q_settled_i, s_i>)`, basin-membership flag (1 if argmax retrieval index = i else 0), or settling trajectory length (steps to convergence).

**Why it's relevant.** The unbind sum collapses many sources of pattern variation into a single noisy vector; settling drives that noise out and converges to discrete basin membership. **The settled state has discrete support; the unbind doesn't.** Three research briefs independently flag this as the cleanest cheap diagnostic on the existing substrate.

**How to explore it.** Wrap `Hopfield.retrieve_with_trace` over the substrate's own atoms (use each atom as a probe). The `TrajectoryTrace` infrastructure already captures top-k, entropy, energy, final state — nothing new needed. ~50 lines, half day.

**Sources.** [research/B-modern-hopfield-snr.md idea B2](research/B-modern-hopfield-snr.md); [research/C-bounded-fidelity-metrics.md #5](research/C-bounded-fidelity-metrics.md); [research/E-cue-binding-dimension.md §3.3](research/E-cue-binding-dimension.md)

**Anti-homunculus shape.** Per-atom measurement using existing settling dynamics. PASS by inheritance from A1' audit.

#### Idea 3: Ganesan Jp/Jn discriminability-gap fidelity (research C #1)

**What.** Replace pairwise-distance with the per-atom discriminability gap from Ganesan 2021 (already bookmarked):

```
f_i^margin = mean_k(
    cos(r_k* ⊗ s_i, a_{i,k})              # role-k filler decodes to its target
  - mean_{j≠k} cos(r_k* ⊗ s_i, a_{i,j})   # … minus mean alignment to other roles
)
```

Bounded in `[-2, 2]`. Per-atom variance is governed by binding count `W`, **not by dimension `D`**. The author's own Figure 1 explicitly shows this gap does NOT close with binding count where pairwise variance does.

**Why it's relevant.** This is a "one-paragraph change to the project's evaluation" (literature-scan agent's phrasing) that may dissolve the noise-floor issue without any architecture change. The Ganesan paper is already in `tmp/pdf_text/4720_learning_with_holographic_redu.txt` with the exact formulas at lines 197-368.

**How to explore it.** ~25 lines drop-in next to `src/energy_memory/phase5/role_fidelity.py:40` (alongside the current `compute_role_fidelity`). Re-run the smoke test against the same A1' snapshot. If `std(f_i^margin)` is non-zero across the 1064 atoms, β is rescued.

**Sources.** [research/C-bounded-fidelity-metrics.md #1](research/C-bounded-fidelity-metrics.md), Ganesan 2021 NeurIPS workshop paper.

**Anti-homunculus shape.** Per-atom measurement of substrate geometry. Same shape as the current `compute_role_fidelity` (also a per-atom row reduction). PASS by inheritance.

---

### Tier 2 — Real architectural moves (~1 day each)

#### Idea 4: PAM-style predictor-distance fidelity (research A — the most surprising find)

**What.** Train a small MLP `g_φ : ℝ^D → ℝ^D` (the Dury 2026 PAM recipe: 2.36M params, 128→1024→1024→1024→128) on the substrate's own atom co-occurrence — paired (atom_i, atom_j) where j appears in i's W-window neighborhood — with InfoNCE loss. Then **define the per-atom fidelity as**:

```
f_i^PAM = exp( -||g_φ(cue_i) − s_i||² / σ² )
```

where cue_i is built from the same cue-synthesis pipeline. Score retrieval as `-||g_φ(query) − s_target||`, not by cosine.

**Why it's relevant.** PAM was explicitly written to address "cosine cannot discriminate states that should be associatively distinct" — Dury reports cosine baseline AUC 0.789 → PAM 0.916 on the equivalent test. **This is the project's exact failure shape, with a published fix.** The metric has per-atom variance by construction because each atom has its own neighborhood of true associates, regardless of D. The PAM framework also dissolves the "replay vs consolidation" split into a single knob: predictor capacity vs corpus.

**How to explore it.** Build a small training loop on the existing substrate: extract co-occurring atom pairs from the trace store, train a 3-layer MLP for ~10 epochs on a few thousand pairs, evaluate. Library scaffolding already partly exists (PyTorch + the existing TrajectoryTrace co-occurrence data). ~1 day to first signal.

**Sources.** [research/A-pam-jepa.md](research/A-pam-jepa.md), [Dury 2026a (PAM, arXiv 2602.11322)](https://arxiv.org/abs/2602.11322), [Dury 2026b (capacity-constrained, arXiv 2603.18420)](https://arxiv.org/abs/2603.18420), [LLM-JEPA (ICLR 2026)](https://arxiv.org/html/2509.14252v2)

**Anti-homunculus shape.** `g_φ` is a fixed-once-trained substrate measurement, not a controller. PASS. The training is done once before evaluation; no online retuning. The PAM framework's "association ≠ similarity" framing is anti-homunculus-clean in the same way the FHRR substrate's bind/unbind operations are clean.

#### Idea 5: Phase 5 headline pivot to pair #4 — metastability ~ replay-prioritization (research D — strongest "stop fighting" move)

**What.** Switch Phase 5's headline metric from K-branch state_divergence to **Δ meta_stable_rate at W=3** under continuous replay-prioritization. The substrate already computes meta_stable_rate every retrieve() call (it's in `phase2/metrics.py` and traced through `TrajectoryTrace`). Add a per-atom EMA `m_i` of metastability, multiply it into the existing `ReplayStore` priority that already composes `gate × tag × suppression`. Atoms whose retrievals are metastable get more replay; atoms whose retrievals are sharp don't.

**Why it's relevant.** Four of five diagnostic-actuator pairs from the 2026-05-09 synthesis have no design notes. Pair #4 is the **cleanest pivot** because (a) the substrate already computes the diagnostic every retrieve call and discards it, (b) the actuator is a Benna-Fusi/Saighi-shape per-atom EMA — same template as A's r_ema, already implemented in `consolidation.py`, (c) the headline metric is the same substrate-pure quantity (Δ meta_stable_rate) that just graduated Phase 4 on D1.

This is the "K-branch was the wrong test; meta_stable_rate is the right one" move. Phase 5 graduates on the same metric class as Phase 4, applied to a mechanism the architecture has already ~90% built.

**How to explore it.** ~½ day plumbing: add `m_i` per-atom EMA in `ConsolidationState`, plumb through to `ReplayStore.add` (multiplicative priority modifier), add the standard tests, run Colab n=10. The MIR (Aljundi 2019) and HEN (Kashyap 2024) papers ground the literature side; both publish the equivalent mechanism.

**Sources.** [research/D-unexplored-actuator-pairs.md](research/D-unexplored-actuator-pairs.md); MIR; HEN; Kashyap 2024.

**Anti-homunculus shape.** Per-atom EMA of a per-atom measurement, integrated into existing replay-prioritization. PASS by inheritance from A's r_ema audit.

#### Idea 6: Permutation-binding as role operator (research E §2.1 — the algebraic fix)

**What.** Replace FHRR convolution-binding with Kanerva/Plate **permutation-binding** for role-vector encoding. The substrate already has `permute()` in `torch_fhrr.py:86-97` — exactly invertible, composes additively, never used as a binding operator. Each role position gets a fixed random permutation π_r; bind(filler, π_r) = π_r(filler); unbind(pattern, π_r) = π_r^{-1}(pattern).

**Why it's relevant.** The FHRR phasor-crosstalk `1 − 1/√D` floor is the *algebraic* source of f_i uniformity. Permutation binding replaces it with a `(W−1)/D` *bundle-only* crosstalk term that has per-atom variance by construction. Recchia et al. 2015 (PMC4405220) empirically show random-permutation binding **beats** circular convolution on paired-associate retrieval (M=457 vs 381, p=0.001) at D=2048. The project has the primitive in the codebase but has never used it as a binding operator.

**How to explore it.** New module `src/energy_memory/substrate/torch_perm_binding.py` (or extend `torch_fhrr.py`). Alternative pipeline: encode windows via `bundle(permute(filler_r, π_r) for r)` instead of `bundle(bind(filler_r, pos_r) for r)`. Retrain Phase 4 → measure f_i on the new substrate. 2-3 days.

**Sources.** [research/E-cue-binding-dimension.md §2.1](research/E-cue-binding-dimension.md); Recchia et al. 2015 PMC4405220; Plate "Permutation as a binding operator"; Kanerva MAP architecture.

**Anti-homunculus shape.** Same shape as FHRR bind/unbind — fixed algebraic operations, no controllers. PASS by inheritance.

#### Idea 7: Multi-scale substrate D=512 ‖ D=4096 (research E §1.2)

**What.** Keep the existing D=4096 substrate for capacity, **add** a synchronized D=512 substrate whose only job is to provide f_i for β. The two substrates encode the same windows at different dimensions; the D=512 substrate has crosstalk noise floor 1/√512 ≈ 0.044 (3× the D=4096 SNR for fidelity measurement) but lower capacity. f_i is computed on the small substrate; retrieval uses the big one.

**Why it's relevant.** The simplest possible answer to "D=4096 noise floor is structural." Don't *replace* the substrate; *add* a measurement-only one at a lower D. Preserves Phase 4 D1 graduation. The two substrates can be trained from the same cue stream in parallel.

**How to explore it.** Modify exp 19 to maintain a parallel D=512 substrate alongside the main D=4096; both consume the same cue stream and Hebbian updates. At Phase 5 evaluation time, compute β's f_i on the D=512 substrate and use the result to weight β's prior over the D=4096 substrate's atoms. ~1 day.

**Sources.** [research/E-cue-binding-dimension.md §1.2](research/E-cue-binding-dimension.md)

**Anti-homunculus shape.** Two substrates running their own local dynamics in parallel; f_i is a measurement on the small one, used as a weight on the big one. No controllers. PASS.

---

### Tier 3 — Bigger architectural reshapes (2-5 days)

#### Idea 8: Sparsemax / SparseMAP cleanup with Hopfield-Fenchel-Young (research B B1)

**What.** Replace softmax in Hopfield settling with **sparsemax** (or α-entmax, or SparseMAP) per Martins et al. 2024. Sparsemax gives **exact retrieval with finite margin** — readouts have *discrete support* (some atoms get exactly zero weight), so a retrieval-side diagnostic gains natural per-atom variance.

**Why it's relevant.** Even if the encoding-side crosstalk floor is structural, the *cleanup-side* output of sparsemax has discrete-zero values for off-support atoms. SparseMAP is specifically designed for retrieving pattern *associations* (vs single patterns) — directly relevant to role-binding. Hopfield-Fenchel-Young is the most directly relevant Modern Hopfield variant the research surfaced.

**How to explore it.** Swap softmax → sparsemax in `torch_hopfield.retrieve_with_trace`. Run smoke test. If post-settling retrieved-state distribution has discrete support, β's f_i computed post-settling (Idea 2) gains discriminative variance.

**Sources.** [research/B-modern-hopfield-snr.md B1](research/B-modern-hopfield-snr.md); Martins et al. 2024 ["Hopfield-Fenchel-Young Networks"](https://arxiv.org/abs/2411.08590).

**Anti-homunculus shape.** Just a different settling kernel. PASS by inheritance.

**Honest uncertainty.** Medium confidence sparsemax will actually produce variance at D=4096 — needs a 1-day prototype to verify before committing.

#### Idea 9: Resonator Networks diagnostic (research B B3)

**What.** Run the substrate's role-binding decomposition through a **Resonator Network** (Frady & Sommer 2020). A resonator iteratively decodes a bound pattern into its factor codebook entries; per-factor convergence trajectories are heterogeneous by construction.

**Why it's relevant.** Resonators *directly solve the factorization problem* the role-fidelity metric is trying to measure indirectly. Per-factor convergence speed (or final cosine to a codebook entry) is a per-atom fidelity signal that's structurally heterogeneous (different atoms have different basin geometries; resonators expose this).

**How to explore it.** Standalone diagnostic on a sample of 100 atoms; not in the runtime architecture. ~2 days to implement the resonator + test.

**Sources.** [research/B-modern-hopfield-snr.md B3](research/B-modern-hopfield-snr.md); Frady & Sommer 2020 resonator network papers.

**Anti-homunculus shape.** Pure measurement, off the runtime path. PASS.

#### Idea 10: Drift / replay-pressure (pair #2 — research D)

**What.** A per-atom drift accumulator `δ_i` (Saighi-shape) coupled to replay weight: high recent drift relative to per-atom estimate → more replay pressure. `codebook_drift()` already exists globally; per-atom δ_i is a trivial reduction.

**Why it's relevant.** Same template as Idea 5 (Saighi-shape EMA into existing replay store) but for the drift axis. If Phase 5 graduates on pair #4, pair #2 becomes the natural Phase 5.5 follow-up.

**How to explore it.** ~1 day. Mirrors Idea 5's plumbing.

**Sources.** [research/D-unexplored-actuator-pairs.md](research/D-unexplored-actuator-pairs.md); MIR + InfoRS literature ground.

**Anti-homunculus shape.** PASS by inheritance.

#### Idea 11: Capacity-constrained replay (research A idea 5)

**What.** Per Dury 2026b: "capacity constraint" is not a loss term but natural under-fitting that emerges when predictor capacity is much smaller than the corpus. **Reframe the project's death dynamics as a capacity-vs-corpus tradeoff**: instead of A+B+A1+A1' continuous throttling, set predictor capacity (e.g., a small projection module on top of substrate) and let consolidation happen by forced compression.

**Why it's relevant.** Dissolves the "replay vs consolidation" split (currently two separate mechanisms) into a single capacity knob. The under-fitting *is* the consolidation. Dury reports 29.4M params on 373M pairs → 42.75% accuracy, which is the actual mechanism by which abstraction emerges.

**How to explore it.** Bigger architectural reshape than other ideas. Probably Phase 6 territory unless the cheaper ideas all fail.

**Sources.** [research/A-pam-jepa.md](research/A-pam-jepa.md); [Dury 2026b](https://arxiv.org/abs/2603.18420).

**Anti-homunculus shape.** Capacity is a fixed-once meta-parameter at construction. PASS by inheritance from A+B's α-fixed-pre-retrain discipline.

#### Idea 12: Cap-coverage as restructuring pressure (pair #5 — research D)

**What.** A substrate-energy term that penalizes low cap-coverage, integrated as continuous gradient flow on patterns (same shape as B's `-α log(d_eff)` repulsion). Atoms in regions of low coverage feel restructuring pressure; the splitting question (pair #3) folds in as restructuring-by-creation.

**Why it's relevant.** Deepest architectural payoff — would unify pair #3 (splitting) and pair #5 (restructuring) into one mechanism. But requires a new substrate-energy term with autograd, the θ′(β) calibration spike (still open), and discovery-channel gating composition.

**How to explore it.** **Phase 6 territory**, not Phase 5. Research D explicitly recommends sequencing pair #5 after pair #4 graduates.

**Sources.** [research/D-unexplored-actuator-pairs.md](research/D-unexplored-actuator-pairs.md)

**Anti-homunculus shape.** Substrate-energy term + autograd gradient flow — same shape as B. PASS by inheritance.

---

## Cross-Cutting Themes

### Theme 1: The substrate works; the diagnostic is misshaped (4 of 5 briefs converge)

Research A, C, D, and E all independently arrive at the same meta-conclusion: A+B+A1+A1' has built the substrate the architecture wants. **The K-branch state_divergence under role-prior was the wrong operationalization of "structural retrieval."** It was derived from the binary-death pre-death substrate's small-N geometry (6-12 atoms, d_eff ~5) where state_divergence was naturally non-zero because the K branches had to occupy distinct corners of a tiny subspace. On the continuous-death substrate (1064 atoms, d_eff 35) there's no small-N geometry to exploit; the test shape doesn't fit.

This is consistent with multiple anomalies already in the empirical record:
- B2 has fired (K=1 ≥ K=4 on post-death substrates) — the K-branching premise itself is empirically weak
- Pre-death seeds 1 and 2 have lowest state_divergence (0.018, 0.008) despite 1024 atoms — pre-death substrate quality is itself bimodal
- Softmax entropy = ln(K) to FP precision across K ∈ {1, 2, 3, 4, 6, 8} — K settled branch energies are *numerically identical*
- Report 050: β q=0 already slightly worse than content_K1 — the continuous-weighted-sum *shape* is energetically inferior to picking the sharpest single atom

The convergent recommendation is: stop fighting K-branch state_divergence; pick a metric that the substrate actually has variance in.

### Theme 2: Three different metrics, all anti-homunculus PASS, all with per-atom variance by construction

Three of the five briefs (A, B, C) converge on the same architectural pattern: **a per-atom fidelity that derives from prediction or settling dynamics, not from pairwise unbind distance.**

- **Research A (PAM):** `f_i = exp(-||g_φ(cue_i) − s_i||²/σ²)` — predictor-distance
- **Research B (settling):** `f_i = cos(q_settled_i, s_i)` or basin-membership flag — post-settling
- **Research C (Ganesan):** `f_i = mean_k(cos(r_k* ⊗ s_i, a_{i,k}) - mean_{j≠k} cos(...))` — discriminability gap

All three share the structural property: **per-atom variance is intrinsic to the metric, not dependent on D.** Each measures a substrate-relationship-to-something (predictor neighborhood / settled basin / discriminative-margin) that has per-atom variance for geometric reasons unrelated to crosstalk.

This is a strong signal: the *category* of "non-pairwise-distance, geometry-grounded, per-atom fidelity" is the right architectural shape. The question is which specific metric to commit to first. Implementation cost ordering: Ganesan (~25 lines, ½ day) < Settling-based (~50 lines, ½ day) < PAM (~1 day, training loop).

### Theme 3: The "untouched axes" beyond the four design notes

The five context summaries identified four implicit assumptions never questioned by the four 2026-05-20 design notes:
1. The substrate is the right object to fix
2. D=4096 is fixed
3. FHRR convolution-binding is the right algebra
4. Cue construction is fixed

The research briefs map these to specific moves:
- Assumption 1 → Theme 1 (the substrate may be fine; the diagnostic is wrong)
- Assumption 2 → Idea 7 (multi-scale D=512 ‖ D=4096)
- Assumption 3 → Idea 6 (permutation-binding via the already-available `permute()`)
- Assumption 4 → Idea 1 (cue-regime sweep)

**All four assumptions can be questioned cheaply.** None requires a full Phase 5 redesign.

### Theme 4: The 2026-05-09 pair structure is a hidden roadmap

Research D's finding that **only 1 of 5 diagnostic-actuator pairs has a design note** is structurally important. The architecture's progression isn't really "Phase 4 → Phase 5 → Phase 6" but "close pair #1 → close pair #2 → ..." The K-branch state_divergence headline implicitly tried to close pair #3 (bimodality/splitting) without an explicit design note for that pair.

Pair #4 (metastability/replay-prioritization) and pair #2 (drift/replay-pressure) both:
- Have their diagnostic side already computed by the substrate
- Have an actuator template (Saighi-shape EMA into ReplayStore) that's already implemented
- Can graduate on a substrate-pure metric that the project has already used (Δ meta_stable_rate)

This is a **roadmap reframing**: Phase 5 isn't "validate structural retrieval"; it's "close pair #4 (metastability)." Pair #5 (restructuring) becomes Phase 6, with pair #3 (splitting) folded in. Pair #2 (drift) is the natural Phase 5.5 follow-up.

---

## Challenges and Counterarguments

### Counterargument 1: Choosing a "metric the substrate has variance in" is a stealth retune

If we replace f_i with whichever of Ganesan/settling/PAM produces non-zero variance, are we **fitting the metric to the substrate** rather than measuring what the architecture claims? The discipline against "tune parameters until metric crosses threshold" applies here too.

**Counter to the counter:** The three candidate metrics (Ganesan margin, settling-based, PAM predictor-distance) were all pre-specified in literature *before* the project's substrate existed. They are not metrics-engineered-for-this-substrate. They're independently-published measures of "binding fidelity" or "associative retrieval quality" that happen to not have the noise-floor pathology FHRR pairwise-distance has. Choosing among them is metric selection, not metric tuning.

But the discipline gate to honor: **pre-commit a falsification criterion for the new metric before observing the result**, just like A+B+A1+A1' did. E.g., for the settling-based f_i: "if cross-seed median |std(f_i)| < 0.01 at n=5, the substrate has no variance in this metric class either; report and stop."

### Counterargument 2: Pivoting to pair #4 means Phase 5 doesn't validate "structural retrieval"

The Phase 5 design says it's about **structural** retrieval (role-prior over content-prior on the consolidated substrate). Pair #4 (metastability/replay-prioritization) tests a different property — that the substrate's replay machinery responds to per-atom metastability, not whether the substrate decomposes role-binding correctly.

**Counter:** "Structural retrieval" was the headline aspiration; the K-branch state_divergence was the operationalization. The empirical data suggests this operationalization wasn't testing what was claimed (substrate quality varies across pre-death seeds; K=1 ≥ K=4; β's f_i has zero variance). If the operationalization is wrong, switching to a different validated property of the substrate is the disciplined move, not a regression of ambition.

That said: pair #4 is **substrate-machinery validation**, not **structural-retrieval validation**. Worth being honest about this in any walk-back.

### Counterargument 3: PAM and settling-based metrics have hidden hyperparameters

PAM has predictor size, training epochs, InfoNCE temperature, etc. Settling-based has cue-synthesis parameters, β temperature, max_iter. These are *new* parameters the four 2026-05-20 design notes don't have. Adding them risks reopening the parameter-tuning question H4 closes.

**Counter:** The discipline binds — set these once from theoretical defaults (PAM's published recipe; settling's existing β=10, max_iter=12), don't tune to land variance in a target range. If they fail at default, that's a falsification, not an invitation to retune.

### Counterargument 4: Permutation-binding is a substrate rewrite, not a patch

Idea 6 (permutation-binding) changes how patterns are encoded. Every existing snapshot has to be retrained from scratch. Phase 4 D1 graduation has to be re-validated. The 2-3 day estimate may be optimistic.

**Counter:** Honest assessment. Permutation-binding is the **algebraic** fix; everything else is workarounds. If the cheaper alternatives all fail or only partially succeed, permutation-binding is the principled long-term answer. But don't open this until Tier 1 (cue sweep, settling-based, Ganesan) and Tier 2 (one of PAM/pair #4/multi-scale) have been exhausted.

### Counterargument 5: The cue regime might not be the lever the literature suggests

Idea 1 (cue-regime sweep) is the cheapest experiment, but the theoretical argument for why D=4096 is structurally noise-dominated is independent of cue regime. Sweeping cue params might just confirm the noise-floor; the binding pathway from cue to substrate isn't where the f_i collapse happens.

**Counter:** True. But the sweep is 20 minutes wall-clock. If it shows zero variance across all 36 cells, that hard-confirms the substrate-side attribution and removes any residual doubt about whether the cue is the lever. If it shows variance in any cell, that's a surprising result that would reframe everything.

---

## Rabbit Holes Worth Following

### Rabbit hole 1: The Vangara-Gopinath cap-coverage-of-unbind-cluster

Research C found this is verified at `tmp/pdf_text/Geometry of Consolidation.txt:24-32`. The project uses Vangara-Gopinath for **substrate-level** d_eff (cap-coverage of pattern set). The same Theorem 1 applies to the **unbind-cluster** — the set of unbound fillers from a single schema. Per-schema cap-coverage of unbind-cluster could be a fidelity metric. Research C #2 has the formula. Not the recommended first move, but worth investigating if Ganesan margin disappoints.

### Rabbit hole 2: Resonator Networks beyond diagnostic use

Research B B3 frames resonators as a measurement tool. But resonators are *also* a retrieval architecture — they could replace the iterative softmax-settling loop entirely. Frady & Sommer's "neural compositional VSA" line is the project's most natural architectural cousin and has been bookmarked but never deeply mined. If pair #4 pivot succeeds and Phase 5 ships, Phase 6 could reasonably be "resonator-based retrieval."

### Rabbit hole 3: LLM-JEPA's SVD-spectrum diagnostic

Research A surfaced LLM-JEPA's per-atom fidelity living "in the rank structure of prediction residuals." This is a different shape from any of the proposed metrics — it's a **spectral** measure, not a scalar. Could be a powerful diagnostic if combined with PAM-style predictor training. Worth a one-day exploration after PAM is wired up.

### Rabbit hole 4: Capacity-constrained replay as the unification

Idea 11 sketches Dury 2026b's "capacity is the consolidation mechanism" framing. This is the deepest possible reframing — it would replace A+B+A1+A1' with a single capacity knob. **Phase 6 or later**, but worth flagging now so the project doesn't over-invest in A+B's specific shape.

### Rabbit hole 5: Sparse VSA / Sparse Distributed Memory comparison

Research E flagged sparse VSA (Schlegel, Eliasmith, Frady) as a parallel substrate option. Sparse representations have different noise-floor scaling than dense FHRR. Not in the top 6 ideas but worth a one-day literature scan if multi-scale and permutation-binding both disappoint.

### Rabbit hole 6: The Hopfield-Fenchel-Young family beyond sparsemax

Idea 8 mentions Martins et al. 2024 specifically. The Fenchel-Young family includes α-entmax (continuous between softmax and sparsemax), Tsallis entropy, etc. There's a temperature/sparsity parameter that could be tuned. The literature is thin on which member of the family is best for VSA-style retrieval — a small spike could pin this down.

---

## Sources

### Bookmarked papers (already in `research/` and `tmp/pdf_text/`)
- Ganesan et al. 2021, "Learning with Holographic Reduced Representations" — `tmp/pdf_text/4720_learning_with_holographic_redu.txt`
- Vangara & Gopinath, "Geometry of Consolidation" — `tmp/pdf_text/Geometry of Consolidation.txt`
- Krotov & Hopfield 2016, "Dense Associative Memory for Pattern Recognition"
- Ramsauer et al. 2020, "Hopfield Networks is All You Need" — https://arxiv.org/abs/2008.02217
- Papyan et al., Neural Collapse (NC2 simplex-ETF)
- Dury 2026a, "Predictive Associative Memory (PAM)" — https://arxiv.org/abs/2602.11322
- Dury 2026b, "Capacity-Constrained PAM" — https://arxiv.org/abs/2603.18420
- LLM-JEPA (ICLR 2026) — https://arxiv.org/html/2509.14252v2
- Kashyap 2024, HEN — Heterogeneous Energy Network
- Saighi et al., self-inhibition replay
- Plate, Holographic Reduced Representations (foundational)
- Kanerva, MAP / Sparse Distributed Memory
- Sahlgren, Random Indexing
- Schlegel et al., Sparse VSA
- Eliasmith, Neural Engineering Framework / Spaun
- Frady & Sommer, Resonator Networks

### Web-discovered papers from research
- Martins et al. 2024, "Hopfield-Fenchel-Young Networks" — https://arxiv.org/abs/2411.08590
- Hoover et al. 2023, "Energy Transformer" — NeurIPS 2023
- Millidge et al., "Universal Hopfield Networks"
- Aljundi 2019, "Maximally Interfered Retrieval (MIR)"
- Benna & Fusi 2016, "Computational principles of synaptic memory consolidation"
- Recchia et al. 2015, "Encoding sequential information in semantic space models" — PMC4405220
- Fang et al., "Neural learning rules for SR" — eLife
- "Modern Methods in Associative Memory" — arXiv 2507.06211 (survey)

### Internal documents
- `notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md` (foundational synthesis)
- `notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md` (A+B)
- `notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md` (A1)
- `notes/notes/2026-05-20-r-inst-measure-dynamic-form.md` (A1')
- `notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md` (β)
- `reports/038–050` — the empirical journey
- `STATUS.md` — bookmark

### Context summaries (in this brainstorm-workspace)
- `context/notes-docs.md`
- `context/reports.md`
- `context/implementation.md`
- `context/literature.md`
- `context/experiments-scripts.md`

### Research briefs (in this brainstorm-workspace)
- `research/A-pam-jepa.md`
- `research/B-modern-hopfield-snr.md`
- `research/C-bounded-fidelity-metrics.md`
- `research/D-unexplored-actuator-pairs.md`
- `research/E-cue-binding-dimension.md`
