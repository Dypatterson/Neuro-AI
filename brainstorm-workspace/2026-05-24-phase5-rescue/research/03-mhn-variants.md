# Research Brief 03 — MHN variants with structured / factorized energy

## Angle

The Phase 5 failure (all K branches collapse to a single basin; role-prior
seeding produces no distinguishable settling) is a *property of the energy
function*, not a tuning problem. The vanilla Ramsauer 2020 energy

```
E(q) = -lse(β · X q) + ½ q²+ const
     = -(1/β) log Σ_μ exp(β ⟨x_μ, q⟩) + ½‖q‖² + const
```

is *content-addressable by construction*: it has one minimum per stored
pattern `x_μ`, with no architectural slot for a "role." Any FHRR
binding-induced role structure has to climb out of the geometry of the
stored content patterns alone, which on the project's substrate it
demonstrably doesn't (`hit_role ≈ 0`, `rank_role ≈ 443/489`).

The thesis I tested: there are MHN variants in the 2022–2025 literature
whose energy is **structured/factorized** in a way that puts role-target
basins into the substrate *intrinsically* — i.e. role-seeded settling
flows to role-distinguished minima because the energy says so, not
because a log-prior bias reshapes logits at the surface.

## Key findings

### 1. Universal Hopfield Networks (Millidge, Salvatori, Song, Lukasiewicz, Bogacz, ICML 2022)

The retrieval step is a sequence of three operations:

```
ξ_out = projection(separation(similarity(M_q, ξ_in)))
```

- `similarity` = `M_q^T ξ` (or kernel `k(M_q, ξ)`)
- `separation` = softmax / sparsemax / polynomial / ID
- `projection` = `M_v · scores`

Concretely the energy formulation generalises to

```
E(ξ) = -F( k(M_q, ξ) )
```

for separation `F` and kernel `k`. **This is the lever.** If `k` is the
linear kernel and `M_q = M_v = stored patterns`, you get vanilla MHN —
one basin per content pattern. But `k` and `M_q ≠ M_v` are degrees of
freedom: you can give `M_q` a role-structured form (e.g. `M_q = role ⊛
content`) while `M_v` is content only, which makes the *similarity*
surface role-conditioned even though the *value* set is unchanged. The
basin shape lives in `k(M_q, ξ)`, not in `M_v`.

### 2. Hopfield-Fenchel-Young Networks (Santos, Niculae, McNamee, Martins, JMLR 2024)

Generalises Ramsauer (α=1, softmax) and Hu 2023 (α=2, sparsemax) to an
α-family with energy:

```
E(q) = -Ω*_β(X q) + ½‖q‖²
```

where `Ω*` is the Fenchel conjugate of a Tsallis/norm entropy. The α=2
(sparsemax) case admits **exact retrieval** of a single pattern (not just
asymptotic), and SparseMAP extends to retrieval of **structured
substructures** of patterns (i.e. retrieval can be constrained to return
combinations satisfying a sparsity pattern).

For Phase 5 the load-bearing observation is: with sparsemax separation,
the support of the retrieval distribution can be *exactly* one pattern.
Once the substrate is in a role's support, it stays there. K branches
seeded into disjoint sparsemax supports cannot collapse to a single
basin — the support boundary is the basin boundary.

### 3. Energy Transformer (Hoover, Liang, Pham, Panda, Strobelt, Chau, Zaki, Krotov, NeurIPS 2023)

Multi-head block whose energy decomposes into two additive terms:

```
E_ET(g) = E_ATT(g) + E_HN(g)

E_ATT = -(1/β) Σ_C lse_B( β · Σ_α (W_K^h g_B)_α (W_Q^h g_C)_α )   [per-head]
E_HN  = -Σ_C Σ_μ G( ⟨ξ_μ, g_C ⟩ )                                 [content basin]
```

Two things matter for us:

- **Per-head additive structure.** The attention energy is summed over
  heads `h`. Each head has its own `W_K^h, W_Q^h` projections. If a head
  is dedicated to "role" projection (e.g. unbinds a role from the FHRR
  query), the attention energy term for that head shapes basins
  conditioned on role overlap, while the content-head term shapes basins
  by content. The total `E_ET` therefore has *role-conditioned* and
  *content-conditioned* minima as separate additive contributions.
- **Explicit Hopfield term added to attention.** Unlike vanilla
  transformers, the ET block exposes the Hopfield energy as a controllable
  additive component. The substrate's MHN can sit in this slot.

### 4. HAMUX / Convex-Lagrangian distributed architectures (Krotov et al., 2025)

Layers are convex Lagrangians composed additively. Each Lagrangian
contributes a term to a global energy that is guaranteed to decrease
along the network dynamics. This means **you can stack a role-Lagrangian
and a content-Lagrangian** and the composite energy is still a Lyapunov
function. Anti-homunculus passes: settling is still gradient flow on a
single energy.

### 5. Self-attention resonator (Hersche IBM Zurich, 2024, arXiv 2403.13218)

Vanilla resonator update for factor j is `x_j ← clip(X_j · X_j^T (s ⊛
∏_{k≠j} x_k))`. Hersche replaces the clip with MHN-style softmax:

```
x_j^(t+1) = X_j · softmax( β · X_j^T ( s ⊛ ⊛_{k≠j} x_k^(t) ) / D )
```

So **each factor has its own codebook `X_j` and its own log-sum-exp
energy on that codebook**:

```
E_j(x_j) = -(1/β) lse( β · X_j^T (s ⊛ ⊛_{k≠j} x_k) ) + ½‖x_j‖²
```

The total system is a coupled set of per-factor Hopfield energies. Each
`x_j` lives in a basin in its own factor space; the coupled dynamics
unbind. This is *exactly* the structured-energy form Phase 5 needs: the
substrate has **F basin landscapes, one per factor (role)**, not one
shared landscape over content.

### 6. Factorizer noise findings (Karunaratne, Hersche, Sebastian, Rahimi, MLNCP@NeurIPS 2024)

Iterative factorizers fall into limit cycles. Noise injected at codebook
initialisation (not throughout iteration) breaks limit cycles without
hurting convergence in expectation. Maps onto Phase 5 directly: K-branch
seeding **is** initialisation noise. If branches are seeded into
different factor-codebook basins, the per-factor energies of (5) are the
mechanism that prevents collapse.

### 7. SQHN — Sparse Quantised Hopfield (Alonso & Krichmar, Nat Comm 2024)

Hidden nodes are sparse one-hot integers; the network implements a
discrete graphical model trained by online MAP. Energy is naturally
**block-structured** by hidden node assignments. Online maximum-a-posteriori
is Hebbian-compatible.

### 8. Vectorial / block-coupled Hopfield (Sclocchi et al. 2025; "amorphous
solid model")

3×3 block-structured couplings produce a **rigid energy landscape with
deep minima for stored patterns** — outperforms scalar Hopfield. Generic
lesson: block structure in J → deeper, better-separated minima.

## Promising leads (ranked by fit to project constraints)

1. **Per-factor coupled-Hopfield substrate (Hersche-style).** Strongest
   direct match. Each role gets its own codebook + own log-sum-exp
   energy. Pure-Python implementable; Hebbian-compatible per codebook;
   exactly the "role-target basin" the project needs.

2. **Energy Transformer multi-head decomposition with FHRR
   role/content heads.** Two heads, additive energies, role and content
   conditioning are *both* gradient flow.

3. **Sparsemax / α-entmax MHN (Hopfield-Fenchel-Young).** Drop-in
   replacement for the current softmax retrieval. Sparsemax supports
   create hard basin boundaries — K branches seeded into disjoint
   supports cannot collapse.

4. **HAMUX-style additive Lagrangian.** Stack role and content
   Lagrangians; settling minimises both. Anti-homunculus by construction
   (one energy, one flow).

5. **Resonator-with-MHN-cleanup** (semantic decomposition, arXiv
   2403.13218). Closest published cousin of what Phase 5 is trying to do.

## Concrete ideas

### Idea A — Per-role Hopfield energy (FHRR-MHN coupled)

Replace the single `E(q) = -lse(β X q)` with one energy per role `r ∈
R`:

```
E_r(q_r) = -(1/β) lse( β · X_r^T q_r ) + ½‖q_r‖²
q_r = unbind(cue, r)                # FHRR: cue ⊛ r*
E_total(q) = Σ_r E_r(q_r)            # additive over roles
```

Each role has its own codebook `X_r` of role-fillers (the atoms that
have ever bound with `r` during Hebbian growth). K-branch seeding picks
a `r` for each branch and settles `q_r` against `X_r` only. Branch
collapse becomes impossible by construction — the role-r branch literally
cannot fall into a role-s basin because they live in disjoint energies.

**Anti-homunculus check.** No supervisor. Each branch is gradient flow on
its own additive term. The "decision" of which basin to land in is the
local minimum of `E_r` for the role the branch was seeded with. Role is
a *coordinate of the energy*, not a switch in front of it.

**Hebbian-compatible?** Yes. When a cue activates atom `a` strongly in
role `r`, increment `X_r ← X_r + η · (a − soft_cleanup_r(a))`. Pure
Hebbian on the per-role codebook.

**Phase-5 graduation prediction.** `ΔE = E_content − E_role` becomes
positive *by construction* under role-seeding because `E_role`'s minimum
is the role-filler basin, and `E_content`'s minimum is the content
basin. The role-target basin retrieval metrics (`hit_role`,
`rank_role`) should leap because retrieval is over `X_r` directly.

### Idea B — Sparsemax MHN with disjoint-support seeding

Swap softmax for sparsemax in the retrieval step. Energy becomes the
Hopfield-Fenchel-Young α=2 form. K branches initialised at queries whose
sparsemax supports are disjoint will land in disjoint basins by support
construction.

**Anti-homunculus check.** No supervisor; sparsemax support is an output
of the energy, not a rule on top of it.

**Risk.** Sparsemax can be too sparse on this substrate (capacity is
sub-linear in N for hard sparsity), but α-entmax is a tunable knob.

### Idea C — Two-head ET-shaped energy on the existing substrate

Drop the ET block's two-term additive structure onto the existing MHN.
One head's `W_K, W_Q` projects to a role-binding subspace
(`W_K = role projection matrix`); the other head's `W_K, W_Q` is the
identity (content). Energies add. K branches at different role queries
sit in different attention basins for the role head while sharing the
content head.

**Hebbian-compatible?** The projection matrices `W_K, W_Q` would need
to be learned offline (back-prop or local error-driven). This is the
weakest fit for the project's "Hebbian-runtime, error-driven-offline"
rule, but **the role projection can be hard-coded from FHRR** (`W_K = I`
with input pre-multiplied by the role-inverse), eliminating the learned
weights and keeping Hebbian-only at runtime.

### Idea D — Per-atom Benna-Fusi α as a *Lagrangian-additive* energy term

The parked Benna-Fusi α idea has an unexploited connection to HAMUX. If
each atom has a frequency-tied multi-timescale α, that α can be folded
into a per-atom Lagrangian whose convexity guarantees Lyapunov descent.
Frequency-weighted consolidation is then a property of the energy
function, not a post-hoc reweighting. Could combine with Idea A: per-role
codebooks with per-atom Benna-Fusi Lagrangians.

## Surprises

- The semantic-decomposition resonator (Hersche 2024) and the Phase 5
  K-branch design are doing **almost the same thing** — they're both
  trying to use MHN settling to disentangle a bound FHRR vector — but
  Hersche puts the role information in **the codebook structure**
  (`X_j` per factor), and Phase 5 puts it in **the prior**. The
  literature has converged on codebook-per-factor; Phase 5 has been
  trying to make it work with one shared codebook and content
  similarity as the gradient. That is the mechanism gap.

- Hopfield-Fenchel-Young (Nov 2024) showed that **exact** retrieval is
  available with sparsemax — the project has been operating under the
  Ramsauer assumption that retrieval is only asymptotically exact. This
  changes the substrate-noise-floor argument: with sparsemax, the
  "noise floor" is zero on a sufficiently sparse substrate. The 5.5e-3
  ΔE threshold may be a softmax-specific artifact.

- The IBM factorizer-noise paper (Dec 2024) showed that **initialisation
  noise alone** breaks limit cycles in iterative factorisation. Phase 5's
  "K-branch seeding" already does this; the missing piece is that the
  branches are seeded into a **shared** energy, not per-factor energies.

- The Energy Transformer's per-head additive energy is exactly the
  structure FHRR wants: roles are heads, and unbinding is a per-head
  projection. The project's substrate is already a single MHN; adding
  a role head is a relatively small architectural change.

## Anti-homunculus filter passes

- **Per-role energies (Idea A):** the role is a *coordinate of the
  energy function*. No rule, no picker. Each branch is gradient flow on
  one additive term. ✅

- **Sparsemax MHN (Idea B):** support is an output of the projection
  step in UHN's similarity→separation→projection chain — fully local. ✅

- **ET two-head (Idea C):** energies add, gradient flow descends total
  energy. ✅

- **HAMUX additive Lagrangian (Idea D):** convex sum of Lagrangians is
  a single Lyapunov function. ✅

All four candidates clear the filter. None require a supervisor module,
none require an if/then rule reading a metric.

## Sources

- [Universal Hopfield Networks (Millidge et al. 2022)](https://arxiv.org/abs/2202.04557)
- [Hopfield-Fenchel-Young Networks (Santos, Niculae, McNamee, Martins 2024)](https://arxiv.org/abs/2411.08590)
- [Sparse and Structured Hopfield Networks (Martins et al. 2024)](https://arxiv.org/abs/2402.13725)
- [On Sparse Modern Hopfield Model (Hu et al. NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/57bc0a850255e2041341bf74c7e2b9fa-Paper-Conference.pdf)
- [Energy Transformer (Hoover, Liang, Pham, Panda, Strobelt, Chau, Zaki, Krotov NeurIPS 2023)](https://arxiv.org/abs/2302.07253)
- [Hierarchical Associative Memory (Krotov 2021)](https://arxiv.org/abs/2107.06446)
- [Self-Attention Based Semantic Decomposition in VSA (Hersche IBM 2024)](https://arxiv.org/abs/2403.13218)
- [On the Role of Noise in Factorizers (Karunaratne, Hersche, Sebastian, Rahimi MLNCP@NeurIPS 2024)](https://arxiv.org/abs/2412.00354)
- [SQHN — Sparse Quantized Hopfield (Alonso & Krichmar Nat Comm 2024)](https://www.nature.com/articles/s41467-024-46976-4)
- [HEN — Hopfield Encoding Networks (Kashyap 2024)](https://arxiv.org/abs/2409.16408)
- [MIMONets / MIMOFormer (NeurIPS 2023)](https://arxiv.org/abs/2312.02829)
- [Resonator Networks 1 (Frady, Kent, Olshausen, Sommer)](https://rctn.org/bruno/papers/resonator1.pdf)
- [Resonator Networks 2 (Kent, Frady, Sommer, Olshausen, Neural Computation 2020)](https://direct.mit.edu/neco/article/32/12/2332/95653/)
- [A Framework for Non-Linear Attention via Modern Hopfield Networks (Farooq 2025)](https://arxiv.org/abs/2506.11043)
- [Generalized Holographic Reduced Representations (2024)](https://arxiv.org/abs/2405.09689)
- [Modern Methods in Associative Memory (HAMUX, Krotov et al. 2025)](https://arxiv.org/abs/2507.06211)
- [A new frontier for Hopfield networks (Krotov, Nature Reviews Physics 2023)](https://ui.adsabs.harvard.edu/abs/2023NatRP...5..366K/abstract)
- [Vectorial Hopfield / amorphous-solid model (Sclocchi et al. 2025)](https://arxiv.org/abs/2507.22787)
