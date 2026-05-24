# Resonator Networks for Factor-Aware Retrieval — Research Brief

## Angle

Phase 5's failure mode is that K parallel HAM-settled branches collapse to
the same basin: the FHRR + Modern Hopfield substrate does not contain
separable role-target attractors for a role prior to "find." Log-prior
biasing reshapes retrieval logits without creating structural retrieval
(`hit_role` ≈ 0, `rank_role` near random), which is exactly the
arbitration-shape antipattern. **Resonator Networks** (Frady, Kent,
Olshausen, Sommer 2020) take a constructive stance instead: they decompose
a bound vector into its factors via iterative unbinding + cleanup against
per-factor codebooks, using *search-in-superposition* dynamics. The
role-target representation is *built by the dynamics*, not retrieved from
a pre-stored attractor. This is precisely the architectural inversion
Phase 5's diagnosis points at.

The 2024 self-attention-based resonator (Hsu & Frady–lineage; arXiv
2403.13218) and the 2025 NeurIPS-workshop survey (Renner, Kymn, Frady,
Sommer; OpenReview FNrZd3Ls1d) tighten the connection to Modern Hopfield
Networks and explicitly support FHRR. That makes the substrate
modification *additive*, not replacement-scale.

## Key findings

1. **Resonator update is structurally a Modern Hopfield read with
   unbinding in the query.** The attention-based variant
   ([Hsu & Frady 2024, eq. 8](https://arxiv.org/abs/2403.13218)) is:
   `x̂_t+1^(j) = X · softmax(β · ℜ[X† (s * ô_t^{¬j})] / D)`
   where `X` is the per-factor codebook, `s` is the bound input, `ô_t^{¬j}`
   is the running estimate of *all other* factors, and `†` is complex
   conjugate transpose for FHRR. **This is exactly a Ramsauer
   continuous-Hopfield retrieval whose query is the unbound residual.**
   Drop-in on this project's substrate: replace `MHN.retrieve(cue)` with
   `MHN.retrieve(unbind(cue, current_other_factors))` and iterate.

2. **Resonator networks have known capacity scaling and known failure
   modes — and noise rescues them.** From Frady/Kent/Olshausen/Sommer 2020
   ([Neural Computation; arXiv 1906.11684](https://arxiv.org/abs/1906.11684)):
   factorization capacity scales as roughly `M^F < α D^2` (M = codebook
   size per factor, F = factors, D = dimension); convergence is not
   guaranteed but is near-certain *inside an operational regime*; outside
   it the system enters limit cycles. The hierarchical resonator paper
   ([Renner et al. 2024 Nature MI, arXiv 2208.12880](https://arxiv.org/html/2208.12880v4))
   *explicitly recommends injected noise + sparsity-encouraging
   nonlinearities + hysteresis to escape spurious minima*. This is a
   substrate-pure stochastic-dynamics mechanism — the anti-homunculus
   filter passes trivially.

3. **The attention-based variant fixes catastrophic FHRR failure of the
   original.** From the Hsu/Frady text: "Original FHRR-adapted resonator
   network has almost 0 accuracy" with the sign-based update; the softmax
   update tolerates continuous unit-magnitude phasors and tolerates
   bundled (k > 1) items per factor. This matters because the
   project's FHRR substrate has been operating without a viable
   factorization dynamic at all — the project's MHN retrieval has been
   doing single-step cleanup, not iterated unbinding+cleanup.

4. **Hierarchical / nested resonators exist and have an explaining-away
   step.** The Renner et al. 2024 hierarchical resonator uses six
   coupled factor modules in a partitioned architecture (Cartesian +
   log-polar) and **subtracts the converged factorization from the
   input ("deflation") before searching for the next object.** This is
   the substrate-pure form of K-branch retrieval the project actually
   wants: not "K parallel softmaxed copies of the same dynamics with
   different priors" but "one dynamics that explains away the first
   binding, then re-runs to find the next." Branches are temporal, not
   parallel.

5. **Codebook-learning and resonator-using are deliberately separated.**
   The Compositional Sparse Coding + Resonator paper
   ([arXiv 2404.19126](https://arxiv.org/html/2404.19126)) makes this
   explicit: convolutional sparse coding learns the codebook *offline*,
   the resonator factorizes *online* with the fixed codebook. This maps
   cleanly onto the project's "Hebbian for runtime, error-driven only in
   batch offline passes" rule — emergent codebook grows online via
   Hebbian, then is frozen as the resonator's per-factor codebook for
   each retrieval episode.

6. **A linearithmic cleanup primitive exists** for VSA key-value memory
   using Kronecker rotation products ([arXiv 2506.15793](https://arxiv.org/pdf/2506.15793)),
   reducing per-step cost from O(MD) to O(D log D). Not load-bearing for
   the architecture decision but removes a scale objection.

## Promising leads

- **Treat the project's slow-store as the *role-factor codebook* and the
  current MHN store as the *content-factor codebook*.** Then a single
  resonator with `F=2` factor modules (role, content) decomposes any
  bound input into (role, content) — directly addressing the "no
  role-target basin" diagnosis by *manufacturing* the role-target
  through dynamics.

- **The "Recent Advances" 2025 survey explicitly flags future work on
  "incorporating learning mechanisms for the underlying generative models
  in Resonator Networks"** ([OpenReview FNrZd3Ls1d](https://openreview.net/forum?id=FNrZd3Ls1d)).
  The project's emergent codebook is exactly that mechanism — there's a
  publishable contribution shape here, not just an internal rescue.

- **The Krausse/Sommer/Renner 2025 grid-cell VSA paper** (arXiv
  [2503.08608](https://arxiv.org/abs/2503.08608)) is structured-codebook
  territory: hexagonal-receptive-field 3D modules act as a *spatially
  structured per-factor codebook*. If "role" carries a structured
  geometry (e.g. position-in-window), this generalizes role-binding
  beyond random FHRR atoms.

## Concrete ideas for the project

### Idea R1: Replace K-branch with single-resonator role/content decomposition

Drop the K-parallel-HAM design entirely. Build a 2-factor resonator:
factor A's codebook is the slow-store (roles/schemas), factor B's is
the fast/active codebook (contents). At cue time, run the resonator
update for ~10–20 iterations:

```
role_t+1     = softmax_retrieve(slow_store,  unbind(cue, content_t),  β)
content_t+1  = softmax_retrieve(fast_store,  unbind(cue, role_t),     β)
```

Headline metric: pre-committed energy gap between converged
(role*, content*) and the energy at random (role, content) init —
substrate-pure, no R@K.

**Anti-homunculus check.** The "decision" of which role wins is a fixed
point of the iterated softmax dynamics, parameterized only by β, the
codebooks, and the cue. No supervisor reads a metric and triggers a
response. The role and content modules are *coupled* (each is the
other's query-modifier), not arbitrated — selection is the same shape as
two-population winner-take-all in cortex, which is the canonical local
geometric dynamic. The 2024 paper frames this as exactly equivalent to
Ramsauer attention, which the project already accepts as substrate-pure.

### Idea R2: Resonator with deflation = principled multi-binding retrieval

After resonator R1 converges to (role*, content*), subtract
`bind(role*, content*)` from the cue and re-run. This is the Renner et
al. hierarchical-resonator explaining-away step. The "branches" of
Phase 5 become *sequential decompositions of the same cue*, indexed by
deflation step rather than parallel-seed index. This sidesteps the
collapse-to-same-basin problem because each subsequent retrieval is on a
different (residualized) cue.

**Anti-homunculus check.** Deflation is subtraction of a converged
binding — a closed-form vector operation, not a rule. There is no "if
energy < threshold then stop" supervisor; you run a fixed budget of
deflation steps and report the energy trajectory.

### Idea R3: Stochastic-resonator replay during consolidation

Use the Renner et al. recommendation: add Gaussian noise + a sparsity
nonlinearity to the resonator updates during *offline replay*. Atoms
that survive resonator decomposition across noise injections (i.e. are
recovered as factors of replayed cues under perturbation) get
Hebbian-reinforced as *factor atoms* in the slow-store. Atoms that
only ever appear bundled get a different consolidation regime.

**Anti-homunculus check.** Noise injection is a stochastic substrate
property (à la Langevin dynamics on the energy surface), not an
arbitration. Hebbian reinforcement is gated by *whether the resonator
recovered the atom as a converged factor*, which is itself a fixed-point
property of the dynamics. No rule reads `is_factor=True` and decides;
the consolidation update is `Δw ∝ activity_at_convergence`, which is
substrate-pure.

### Idea R4: Empirical β calibration is now a resonator-network problem

The project already has the "empirical θ′(β) calibration spike" as a
parked idea. In a 2-factor resonator the same β controls both softmaxes;
the operational-regime boundary (Frady et al. 2020 §capacity) is a
function of β, D, M, F. The Vangara/Gopinath E1 calibration generalizes
cleanly. Net effect: a parked single-MHN calibration spike upgrades to
*the* tuning knob of the new architecture.

## Surprises

- **The attention-based resonator update is essentially "Modern Hopfield
  retrieval whose query is an unbound residual."** The substrate change
  is *one line of code* (the query becomes `s * ô_t^{¬j}` instead of
  `s`), and yet it converts single-step cleanup into iterated structural
  decomposition. The project's existing `MHN.retrieve()` is reusable.

- **Original resonator networks are catastrophically bad on FHRR.**
  The sign-based 2020 update is for bipolar VSAs. The project has been
  using FHRR throughout; any naive port of the 2020 paper would have
  failed. The 2024 attention variant is the first FHRR-native
  resonator, and it appeared exactly when the project needed it.

- **The hierarchical-resonator paper explicitly endorses injected noise
  and sparsity nonlinearities for escaping spurious minima.** The
  project has been chasing log-prior biases (arbitration-shape) while
  the published recipe for "branches that don't collapse" is
  Langevin-style perturbation of the dynamics — the *exact* kind of
  substrate-pure mechanism the anti-homunculus filter rewards.

- **The 2025 survey names "learning the generative model" as open
  future work.** The project's emergent codebook is a candidate answer
  to a literature open question, not just a rescue.

- **The resonator literature has *never* used "branches with different
  priors" as its multi-hypothesis mechanism.** It uses *one* dynamics
  with *deflation* across passes, or *one* dynamics with *noise* across
  restarts. Phase 5's K-parallel-priors design has no analogue in this
  literature, which is consistent with the project's empirical finding
  that it does not produce structural retrieval.

## Sources

- [Frady, Kent, Olshausen, Sommer (2020). Resonator Networks 1: An Efficient Solution for Factoring High-Dimensional, Distributed Representations of Data Structures. Neural Computation 32(12).](https://direct.mit.edu/neco/article/32/12/2311/95641)
- [Kent, Frady, Sommer, Olshausen (2020). Resonator Networks 2: Factorization Performance and Capacity. Neural Computation 32(12).](https://direct.mit.edu/neco/article/32/12/2332/95653/Resonator-Networks-2-Factorization-Performance-and)
- [Resonator Networks outperform optimization methods (arXiv 1906.11684).](https://arxiv.org/abs/1906.11684)
- [Hsu et al. (2024). Self-Attention Based Semantic Decomposition in Vector Symbolic Architectures (arXiv 2403.13218).](https://arxiv.org/abs/2403.13218)
- [Hsu et al. (2024). HTML version, full text.](https://arxiv.org/html/2403.13218v1)
- [Renner, Kymn, Frady, Sommer (2025). Recent Advances in Resonator Networks for Neurosymbolic Computing. OpenReview FNrZd3Ls1d.](https://openreview.net/forum?id=FNrZd3Ls1d)
- [Renner et al. (2024). Neuromorphic visual scene understanding with resonator networks. Nature Machine Intelligence 6(6).](https://www.nature.com/articles/s42256-024-00848-0)
- [Renner et al. (2024). Hierarchical resonator network full text (arXiv 2208.12880v4).](https://arxiv.org/html/2208.12880v4)
- [Compositional Factorization of Visual Scenes with Convolutional Sparse Coding and Resonator Networks (arXiv 2404.19126).](https://arxiv.org/html/2404.19126)
- [Krausse, Sommer, Renner, Neftci (2025). A Grid Cell-Inspired Structured Vector Algebra for Cognitive Maps. NICE 2025 (arXiv 2503.08608).](https://arxiv.org/abs/2503.08608)
- [Linearithmic Clean-up for Vector-Symbolic Key-Value Memory with Kronecker Rotation Products (arXiv 2506.15793).](https://arxiv.org/pdf/2506.15793)
- [Spencer Kent reference implementation of Resonator Networks (GitHub).](https://github.com/spencerkent/resonator-networks)
- [Frady et al. focus reading: Resonator circuits (Redwood Center).](https://redwood.berkeley.edu/wp-content/uploads/2021/08/Module5_ResonatorNetworks_FocusReading.pdf)
