# 06 — Energy-based training / score matching / replay-as-EBM for role-target basins

## Angle

The Phase 5 diagnosis is that the substrate **has no role-target basins**.
Reports 041–061 show `hit_role ≈ 0.003`, `rank_role ≈ 443/489`, and naive
K-branch retrieval collapses across priors. This is exactly the failure
mode you'd expect from a substrate trained *only* by Hebbian co-occurrence
during retrieval: Hebbian reinforcement makes whatever atom won the last
softmax slightly more retrievable. It is a positive-phase-only rule. There
is no negative phase, so the energy landscape has never been told
*not* to put a basin at a role-mismatched configuration.

EBM theory says the way to *place a basin where you want one and remove
one where you don't* is contrastive: pull energy down on positives, push
energy up on negatives. Modern Hopfield is *already an EBM*
(log-sum-exp energy; gradient is the softmax retrieval step). So if we
add an **offline-batch contrastive pass** — Hebbian for runtime stays
intact — and construct positives/negatives so that role-aligned
configurations are positives and role-mismatched configurations are
negatives, we'd be training the substrate to put role-target basins where
the design has been *assuming* they emerge for free. They don't.

This angle is sanctioned by the project's rule ("error-driven only in
batch offline passes"), uses the existing replay buffer as its sampling
mechanism, and respects FHRR + MHN as the substrate. The mechanism that
distinguishes it from "supervisor module that picks which subsystem wins"
is that all the work is **energy-landscape sculpting at consolidation
time**; at runtime the same energy-only K-branch settling remains the
sole arbiter.

## Key findings

1. **Modern Hopfield is trivially trainable as an EBM.** Ramsauer's
   log-sum-exp energy `E(s) = -lse(β · P s)` is differentiable in the
   patterns `P`. The retrieval step is one gradient descent step on this
   energy. Hopfield-Fenchel-Young (Santos et al. 2024, JMLR 2025;
   arXiv:2411.08590) generalises this to a *family* of energies
   parameterised by Fenchel-Young losses, and explicitly proves the
   connection between **loss margin, sparsity, and exact retrieval of a
   single pattern**. The "margin" object is exactly what's missing in
   Phase 5 — role vs content patterns currently have no margin separating
   their basins, so MHN softmax flow erases the distinction during
   settling. The Fenchel-Young training objective gives a principled
   loss to optimise that margin offline.

2. **Equilibrium propagation (Scellier & Bengio 2017; Laborieux 2021;
   Bal & Sengupta 2023; Lin/Bal/Sengupta 2024; Hopfield-Resnet 2024) is
   the most-local error-driven training rule that exists for energy
   networks.** Two-phase: (1) **free phase**, network settles under the
   cue alone; (2) **nudged phase**, network settles with a small force
   pulling the output toward the target. The weight update is
   `Δw ∝ (1/ε)[⟨s_i s_j⟩_nudged − ⟨s_i s_j⟩_free]`. This is
   Hebbian minus anti-Hebbian — *Movellan-style contrastive Hebbian*
   (1991), proved equivalent to BPTT in the limit. **Crucially, both
   phases are pure settling dynamics on the existing substrate.** No
   backprop graph, no auxiliary network. This is the rule that most
   cleanly survives the anti-homunculus filter.

3. **Persistent CD with replay buffer (Tieleman 2008; Du & Mordatch
   2019; modern PCD with diffusion-CD 2023–2024).** Negative samples are
   drawn from MCMC chains that are *initialised from the replay buffer*
   and persist across training steps. The project *already has a
   consolidation replay buffer.* It is essentially a PCD buffer in
   disguise. Du & Mordatch's stability tricks (L2 regularisation on
   energies, spectral norm, KL-to-buffer) are the difference between
   training that converges and training that diverges.

4. **Score matching / denoising score matching (Song & Ermon 2019;
   Salimans & Ho 2021) is partition-function-free.** Instead of
   contrasting positives against samples, it matches ∇log p(s) (the
   score). For Hopfield, `∇E(s)` is the retrieval residual, which is
   already computed every step. **Denoising score matching** says: take
   a clean role-bound exemplar, perturb it with FHRR-appropriate noise
   (random phase jitter at each complex element), and train the energy
   so that ∇E points back to the clean state. No negative sampling at
   all. This is the cheapest pass that still introduces a *gradient
   signal* into atom updates.

5. **Forward-forward (Hinton 2022) and Mono-Forward (2025) are local
   layerwise contrastive Hebbian.** Each layer/atom has a local
   goodness function (sum-of-squares of activations) trained to be high
   on positives, low on negatives. The relevance here is that FF gives
   a recipe for **per-atom local updates with no global error signal** —
   exactly what the emergent codebook needs if you want each atom to be
   trained independently against its own positive/negative pairs.
   Hinton explicitly proposed FF as a model of cortical learning and as
   compatible with low-power analog substrates; the project's
   "Hebbian for runtime" rule is closer to FF than to backprop.

6. **In-context denoising = MHN trained as a denoising score model
   (Smart 2025, ICML).** Shows that a single-layer transformer trained
   on a denoising prompt task *learns* a context-aware DAM energy
   landscape, and the trained attention layer performs one gradient
   descent step on that energy. This is an existence proof that
   *training MHN with a score-matching-like objective places the basins
   you train for.*

7. **EBMs for relational/role structure exist already.** Du et al.
   "Learning to Compose Visual Relations" (NeurIPS 2021) trains a
   separate EBM per relation, and a composite scene's energy is the
   sum. Role-binding in FHRR is the binding analog: role*filler. An EBM
   trained on role-bound positives vs. role-shuffled negatives is the
   direct analog of their factorised-relation training.

## Promising leads

### Lead A — Equilibrium propagation on the existing MHN substrate

This is the single highest-leverage idea in this brief.

- Free phase: cue is a role binding `r⊗f`. Substrate settles via the
  existing K-branch HAM dynamics. Take fixed point `s_free`.
- Nudged phase: same cue, but add a small β-scaled term that pulls `s`
  toward the *true filler* `f*` for that role: `E_nudge = E(s) + λ
  ||s − f*||²`. Re-settle. Take fixed point `s_nudge`.
- Weight (atom) update: `Δp_k ∝ (1/λ)[(p_k · s_nudge) s_nudge −
  (p_k · s_free) s_free]`.

This is Movellan-style CHL on the continuous Hopfield model — already
proved to perform gradient descent on the squared error. It uses **only
the substrate's own settling dynamics**, no backprop, no auxiliary
module. Phase 4's frequency-weighted consolidation can stay; this
addition runs in the same offline pass, sampling cues from the replay
buffer.

**Anti-homunculus check.** Who picks `f*`? In the offline batch the
*ground truth* role-filler pair is what generated the replay sample
in the first place. No runtime arbitration — the "decision" lives in
the perturbation force λ during the nudged phase, which is a local
geometric force on the state vector. Selection of the perturbation
target is data-driven (replay sample), not metric-driven.

### Lead B — Persistent CD with the existing replay buffer

- Positives: role-bound exemplars `r⊗f` drawn from replay buffer.
- Negatives: states drawn from PCD chains initialised from a buffer of
  *past negatives*; each chain runs K Langevin steps on `E` between
  training updates.
- Update: `Δp_k ∝ E_{x_+}[∂E/∂p_k] − E_{x_−}[∂E/∂p_k]` evaluated by
  the standard MHN energy gradient (a softmax-weighted outer product).

Du & Mordatch's `L_reg = α(E(x_+)² + E(x_−)²)` is essential — without
it the energies drift.

**Anti-homunculus check.** Negative selection is the danger. If
negatives are "compute role similarity, pick lowest-similarity atoms,"
that's arbitration-shape. PCD avoids this: negatives are *samples from
the model's own current distribution*, which is a pure dynamical
process. Langevin steps on `E` are local geometric flow. The buffer is
just memory; it doesn't *decide* anything.

### Lead C — Denoising score matching pass (cheapest)

- Take a clean role-bound exemplar `x = r⊗f`.
- Perturb: in FHRR, draw per-element phase noise `δφ_i ~ N(0, σ²)` and
  form `x̃ = x · e^{iδφ}`.
- Train energy so that `∇E(x̃)` points back along `x − x̃`:
  `L_DSM = E[ || ∇E(x̃) + (x̃ − x)/σ² ||² ]`.

For MHN, `∇E(x̃) = −β · P · softmax(β · P^H x̃)`, so this is a closed-
form gradient w.r.t. the patterns `P`. No negative sampling, no MCMC.

**Anti-homunculus check.** No selection at all. Noise is sampled from
a fixed distribution; gradient is computed locally per atom from its
contribution to the softmax. This is the most defensible variant.

### Lead D — Per-atom Forward-Forward style positive/negative goodness

- Each atom `p_k` has a per-atom goodness `g_k(x) = (p_k · x)² · m_k`
  where `m_k` is the atom's frequency weight.
- Positives are role-bound samples from replay; negatives are
  role-shuffled samples (same content fillers bound to *wrong* roles).
- Update: increase `g_k` on positives, decrease on negatives. This is
  literally contrastive Hebbian per atom.

**Anti-homunculus check.** Role-shuffling is a *data-augmentation*
operation, not a metric-driven choice. The "wrong role" is sampled
uniformly from the role pool. No metric is consulted to construct
negatives.

## Concrete ideas for the project

### Proposal — "EqProp consolidation pass"

```python
# Runs in the offline consolidation phase, alongside Phase 4's
# frequency-weighted consolidation. Hebbian runtime untouched.
def eqprop_consolidation_pass(substrate, replay_buffer, lam=0.05, eta=1e-3):
    for cue, role, true_filler in replay_buffer.sample_role_bound_batch():
        # ---- Free phase: pure settling on the existing substrate ----
        s = substrate.settle(cue, n_steps=12)         # K-branch energy-only
        s_free = s.detach()

        # ---- Nudged phase: same dynamics + small target-pull ----
        s = substrate.settle_nudged(
            cue,
            nudge=lambda s: lam * (true_filler - s),  # local geometric force
            n_steps=12,
        )
        s_nudge = s.detach()

        # ---- Local contrastive Hebbian update per atom ----
        for k, p_k in enumerate(substrate.atoms):
            a_nudge = (p_k.conj() * s_nudge).sum()    # FHRR inner product
            a_free  = (p_k.conj() * s_free ).sum()
            grad_k = (a_nudge * s_nudge - a_free * s_free) / lam
            p_k += eta * grad_k                       # atom-local update
            p_k /= p_k.abs().clamp_min(1e-8)          # FHRR unit-magnitude
```

Headline metric this would move: ΔE between role-prior and content-prior
branches. Mechanism: the nudged-phase pull explicitly carves a basin at
`true_filler` for cues of the form `r⊗·`, so role-seeded settling has
somewhere to fall. Drill-downs to instrument: `hit_role` (should rise
from 0.003), `rank_role` (should drop from 443/489), per-branch
divergence (should rise from ~1e-5).

### Proposal — "DSM warm-start before EqProp"

Run DSM (Lead C) for a few epochs first. DSM does not require sampling
and is partition-function-free, so it's cheap. It seeds the energy
landscape with role-aware curvature. Then EqProp refines the basins.
This mirrors the standard EBM-training recipe (score matching to warm
start, then CD to polish).

### Proposal — "Role-shuffled negatives in the existing replay loop"

Lightest-touch variant. Inside each replay step, with probability 0.5,
shuffle the role index across the batch to produce a role-mismatched
binding. Use as a *negative-phase* Hebbian update (subtract the outer-
product update instead of adding it). This is essentially CHL with
"data-augmentation negatives" — no separate MCMC, no nudged phase.
Cheapest implementation; weakest signal.

## Surprises

- **The energy landscape may already have role-target basins for
  patterns the substrate has seen during training — but only weakly,
  because nothing trained against negatives.** The Hopfield-Fenchel-Young
  result that *loss margin determines exact retrieval* is the
  theoretical statement of why Phase 5 fails. Hebbian-only training has
  zero margin between role-aligned and role-shuffled basins, so MHN
  softmax flow at finite β erases the distinction. This is not a
  pathology of FHRR or of the project's parameters; it's a generic
  result of training an EBM without negatives.

- **The project's own replay buffer is already 80% of a PCD buffer.**
  No new infrastructure is needed to add a negative phase. The missing
  piece is the negative-sample generator, which can be as cheap as
  role-shuffling.

- **Equilibrium propagation literally was invented for this setting**
  (CHL on continuous Hopfield, learning rule local in space, gradient
  of energy at equilibrium). It is not exotic; it is the textbook fit
  for a contextual-completion substrate with replay.

- **Score matching is partition-function-free**, which means none of
  the "energy drifts to -∞" pathologies of CD apply. For a substrate
  that already worries about settling stability, DSM is the safer
  first move.

- **In-context denoising (Smart 2025)** is the smoking gun that
  training MHN with a denoising score objective *places basins where
  you train for*. This is direct evidence that the proposed mechanism
  works on the substrate family the project uses.

## Sources

- [Equilibrium Propagation (Scellier & Bengio 2017)](https://arxiv.org/pdf/1602.05179)
- [Equilibrium Propagation with Continual Weight Updates (Ernoult et al. 2020)](https://arxiv.org/pdf/2005.04168)
- [Scaling EP to Deeper Architectures / Hopfield-Resnet (2025)](https://arxiv.org/html/2509.26003)
- [Jacobian-homeostasis EP without weight symmetry (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6a55f024db3f771194bdadc8f3a35381-Abstract-Conference.html)
- [Directed Equilibrium Propagation Revisited (2025)](https://www.mdpi.com/2227-7390/13/11/1866)
- [Contrastive Hebbian Learning in the Continuous Hopfield Model — Movellan 1991](https://inc.ucsd.edu/mplab/46/media/CHL90.pdf)
- [Implicit Generation and Modeling with Energy-Based Models — Du & Mordatch 2019](https://arxiv.org/pdf/1903.08689)
- [Improved Contrastive Divergence Training of EBMs](https://www.researchgate.net/publication/346578924_Improved_Contrastive_Divergence_Training_of_Energy_Based_Models)
- [Training EBMs with Diffusion Contrastive Divergences (2023)](https://arxiv.org/html/2307.01668)
- [Persistently trained, diffusion-assisted EBMs](https://statweb.rutgers.edu/ztan/Publication/Zhang-Tan-Ou-Stat.pdf)
- [Hopfield-Fenchel-Young Networks (Santos et al., JMLR 2025)](https://arxiv.org/abs/2411.08590)
- [Sparse and Structured Hopfield Networks (ICML 2024)](https://proceedings.mlr.press/v235/santos24a.html)
- [In-Context Denoising = MHN trained by denoising (Smart 2025)](https://arxiv.org/abs/2502.05164)
- [Hopfield Networks is All You Need (Ramsauer 2020)](https://arxiv.org/abs/2008.02217)
- [Forward-Forward (Hinton 2022)](https://www.cs.toronto.edu/~hinton/FFA13.pdf)
- [Self-Contrastive Forward-Forward (Nature Comm 2025)](https://www.nature.com/articles/s41467-025-61037-0)
- [Mono-Forward: backprop-free local-error training (2025)](https://arxiv.org/pdf/2501.09238)
- [Learning to Compose Visual Relations — EBM per relation (Du et al. 2021)](https://arxiv.org/pdf/2111.09297)
- [Concept Learning with Energy-Based Models (Mordatch 2018)](https://arxiv.org/abs/1811.02486)
- [Neural Learning Rules from Associative Networks Theory (2025)](https://arxiv.org/html/2503.19922)
- [Benchmarking Hebbian learning rules for associative memory (2024)](https://arxiv.org/pdf/2401.00335)
- [Towards a Model of Associative Memory with Learned Distributed Representations (ICANN 2024)](https://dl.acm.org/doi/10.1007/978-3-031-72332-2_16)
- [Novel local learning rule for Hopfield via Minimum Probability Flow](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3704313/)
- [Local learning rules to attenuate forgetting (2018)](https://arxiv.org/pdf/1807.05097)
- [Learning to Perform Role-Filler Binding with Schematic Knowledge](https://arxiv.org/pdf/1902.09006)
