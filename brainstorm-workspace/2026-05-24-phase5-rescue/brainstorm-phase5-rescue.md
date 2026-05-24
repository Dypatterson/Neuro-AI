# Brainstorm: Phase 5 rescue mechanisms

> Generated 2026-05-24 from session context, the Phase 5 design doc, the
> checklist, Report 061, and six parallel deep-research passes (resonator
> networks; active inference / PE-driven branching; Modern Hopfield
> variants; slot attention / competitive grouping; TEM / hippocampal
> relational memory; EBM training of the substrate).

---

## Project understanding

The active phase is 5: K-branch energy-guided structural retrieval on the
post-death substrate. Headline `ΔE = E_content_prior − E_role_prior`
paired per cue, CI > 0, magnitude ≥ 5.5e-3, n_seeds ≥ 10
([phase-5-unified-design.md:269-292](../../notes/emergent-codebook/phase-5-unified-design.md)).
Across reports 041–061 the project tried naive K-branch retrieval,
substrate scale-up (Path A; SNR-invariant in N), and a Varner-style
log-prior softmax-bias spike (Path C). Path C crosses the magnitude
floor on the slow-store substrate at gain=1
([Report 061](../../reports/061_phase5_log_prior_gain1_required_controls.md)) —
but `random_lowest = 0.37` (barely above chance), `hit_role ≈ 0.003`,
and `rank_role ≈ 443/489`. The energy margin exists; the structural
retrieval doesn't. The 2×2 ablation isolates the log-prior channel as
the dominant lever, and the no-schema-store amplification (16.6× floor)
is an arbitration-shape positive control, not portable evidence.

**The diagnosis the project has converged on is correct, and the
research confirms it from six independent angles**: the FHRR + Modern
Hopfield substrate as currently shaped does not contain role-target
attractors. The log-prior reshapes retrieval *logits*, not the energy
*landscape*, so role-seeded settling has nowhere to fall. K-branch
retrieval with shared content energy cannot manufacture the missing
basins; it surfaces their absence.

The good news is that there is a **strong convergent answer in the
literature**, and the substrate change is additive (not architectural
replacement). All five 2024–2026 lines (resonator-MHN, IDP-Hopfield,
sparsemax MHN, slot-Hopfield seam, EqProp on continuous Hopfield)
agree on the same family of fixes. The disagreements are about which
piece is load-bearing.

---

## The convergent finding (six angles, one diagnosis)

| Angle | Identifies the gap as… | Proposed fix family |
|---|---|---|
| Resonator (R) | Branches are seeded with priors on **one shared codebook** instead of decomposed against per-factor codebooks | Per-factor coupled MHN energies + iterated unbinding |
| MHN variants (M) | Ramsauer energy has **one basin per content pattern**; no architectural slot for "role basin" | Per-role codebooks (Hersche), Energy Transformer two-head, sparsemax (Hopfield-Fenchel-Young) |
| TEM (T) | Storage rule (Hebb) yields correlated basins; **TEM's role basins exist because of a KL prior-matching term** in training | Pseudo-inverse / Storkey storage; transition operators; KL two-stream pressure |
| Slot attention (S) | K-branch is **the exact ablation Locatello warns against** — softmax-over-keys gives K slots no way to differentiate | Cross-branch softmax (softmax-over-K) during settling |
| Active inference (A) | The log-prior **reshapes logits, not landscape**; saliency must enter the gradient, not the softmax bias | IDP-style `y ⊙ α` Hadamard gating; PE-modulated Hebbian; EVB-softmax replay budget |
| EBM training (E) | Hebbian-only training has **zero margin** between role-aligned and role-shuffled basins; Hopfield-Fenchel-Young says margin is what makes retrieval exact | EqProp consolidation; PCD with replay buffer; denoising score matching; role-shuffled negatives |

**Restated as one sentence**: vanilla Modern Hopfield is content-addressable
by construction; making it role-addressable requires either splitting
the energy across role-factor codebooks (R, M, T), shaping the energy
landscape per-cue via a saliency that enters the gradient (A), or
training the energy contrastively against role-shuffled negatives (E, T).
Slot attention adds a single missing primitive — cross-K normalization —
that makes the K branches non-redundant during settling (S).

---

## Ideas and approaches

Ranked, with explicit anti-homunculus checks. Each idea names which
research angle generated it so you can cross-reference the briefs in
`research/`. Ideas come in **three tiers** by engineering cost and
architectural commitment.

---

### Tier 0 — Cheap diagnostics that isolate the bottleneck before committing

These are <1-day spikes whose only purpose is to tell you which of the
deeper fixes is load-bearing. Run them before committing to any Tier 1
or Tier 2 architecture.

#### Idea D1: Pseudo-inverse storage swap (T-E)

**What.** Without changing anything else, post-consolidation, recompute
the MHN weight matrix using Kymn-style pseudo-inverse (or Storkey-style
local correction) over the consolidated atoms. Re-run the Phase 5
headline. Kymn et al. 2022 ([PMC9759586](https://pmc.ncbi.nlm.nih.gov/articles/PMC9759586/))
showed HRR role-filler retrieval works when storage is pseudo-inverse,
fails when storage is pure Hebb.
**Why now.** The TEM brief identifies this as the smoking gun: the
storage rule, not the binding algebra, is the load-bearing parameter.
If `hit_role` jumps from 0.003 to 0.1+ with the swap, the substrate
*does* contain role information and pure-Hebb softmax has been masking
it. If it doesn't, the atoms themselves are wrong-shape and Tier 1 is
required.
**How to explore.** Take the existing post-death substrate (no
retraining). Compute `W = X (XᵀX + λI)⁻¹ Xᵀ` over the slow-store
atoms. Run the report-061 control matrix with this W. Compare
`hit_role`, `rank_role`, `random_lowest` to the v2 row at gain=1.
**Anti-homunculus check.** ✅ **PASS (unconditional)** per reviewer
agent audit 2026-05-24. Storage operation only; runtime retrieval is
unchanged energy-only K-branch. The pseudo-inverse is a *property* of
the atom set, structurally equivalent to changing the storage-rule
constant (Hebb ↔ Storkey are different constants).
**Failure mode that would re-introduce arbitration.** Adding a
runtime branch like `use pseudo-inverse W if rank_role > τ else
Hebb W` — keep the swap global and unconditional.
**Sources.** Kymn/Stewart 2022; TEM brief Idea E.

#### Idea D2: Sparsemax retrieval swap (M-E)

**What.** Swap the MHN softmax for sparsemax (Hopfield-Fenchel-Young
α=2). Sparsemax supports admit exact retrieval — branches initialised
into disjoint sparsemax supports cannot collapse to one basin.
**Why now.** The MHN brief flags that the 5.5e-3 magnitude floor is
"a softmax-specific artifact." Under sparsemax the noise floor may
collapse to zero on a sufficiently sparse substrate, and the K-branch
collapse becomes geometrically impossible if seeded into disjoint
supports.
**How to explore.** Replace `softmax(β·Xq)` with `sparsemax(β·Xq)` in
`MHN.retrieve()`. Run the headline. Optionally sweep α in the
α-entmax family (1 = softmax, 2 = sparsemax, in-between = tunable
sparsity).
**Anti-homunculus check.** ✅ **PASS (unconditional)** per reviewer
agent audit 2026-05-24. Sparsemax is the gradient of a Fenchel
conjugate; the sparse support is an *output* of the energy
minimization, not a rule on top of it (Hopfield-Fenchel-Young, Santos
2025).
**Failure mode that would re-introduce arbitration.** Per-cue α in
α-entmax (e.g. `α=2 when cue entropy high, α=1.2 otherwise`) imports
a controller. Use global α only, or a learned α that's a substrate
parameter trained the same way as β.
**Sources.** Santos et al. JMLR 2025 ([arXiv:2411.08590](https://arxiv.org/abs/2411.08590));
MHN brief Idea B.

#### Idea D3: Slot-style cross-branch softmax (S)

**What.** During K-branch settling, after computing per-pattern logits
ℓ_k(p) = β·⟨s_k, p⟩, add a normalization across the K axis:
`α_k(p) = softmax_k(ℓ_k(p))`. Mix into each branch's update.
**Why now.** The slot-attention brief argues this is *the* missing
primitive. K-branch with softmax-over-keys is the exact ablation
Locatello et al. 2020 warns against, and the observed collapse is the
predicted outcome. Cross-K normalization makes branches push each other
out of overlapping basins by local dynamics.
**How to explore.** One-line change in the K-branch HAM-settling loop.
Test at K ∈ {2, 4, 8}. Drill-down: per-branch state divergence (E5)
should rise above the current 6.7e-6–2.7e-3.
**Anti-homunculus check.** ⚠️ **CONDITIONAL (Lyapunov check required)**
per reviewer agent audit 2026-05-24. Pointwise inner-product
computation is local, but the K branches no longer evolve under K
independent Lyapunov functions — they evolve under one coupled flow
over K×N. The slot-attention analogy is suggestive but NOT
load-bearing for the anti-homunculus claim: Locatello's contraction
comes from end-to-end training with a permutation-invariant decoder
loss, not from the softmax-over-K primitive in isolation.
**Commit gate before D3 lands.** Write down the joint energy
`E(s_1,…,s_K)` whose gradient flow IS the proposed update; prove (or
numerically demonstrate) that it decreases monotonically under the
proposed step; confirm the fixed-point structure isn't degenerate.
Half-day analytical pass. If no such joint energy exists, the update
is implementing a competitive *rule* (not a gradient) and D3 is
arbitration-shape, regardless of how local-looking each pointwise step
is.
**Failure mode that would re-introduce arbitration even if Lyapunov
passes.** Adding an early-exit `if max_k α_k(p) > τ then commit branch
k to pattern p`. The cross-K softmax must remain a soft re-weighting
that runs to convergence.
**Sources.** Locatello et al. NeurIPS 2020 ([arXiv:2006.15055](https://arxiv.org/abs/2006.15055));
slot brief Idea 1.

> **Decision recipe.** Run D1 and D3 first; they're orthogonal and
> single-PR each. If D1 helps and D3 doesn't, the bottleneck is
> storage geometry (go Tier 1 path T/E). If D3 helps and D1 doesn't,
> it's branch-coupling (go Tier 1 path S). If both help, do both at
> Tier 1 (path M). If neither helps, the atoms themselves don't carry
> role information; go Tier 2 (training-time intervention).

---

### Tier 1 — Single architectural changes with high convergence support

Each of these is one self-contained mechanism. Pick one based on the
Tier-0 diagnostics. They are designed to be **mutually-stackable** if
you want — none precludes the others.

#### Idea P1: Per-role coupled MHN codebooks (R, M, T — strongest convergence)

**What.** Replace the single MHN energy `E(q) = -lse(β · X q)` with a
sum over roles: for each role r in the role bank R, maintain its own
codebook `X_r` of atoms that have bound with r during consolidation,
and compute
```
q_r = unbind(cue, r)           # FHRR: cue ⊛ r*
E_r(q_r) = -(1/β) lse(β · X_rᵀ q_r) + ½‖q_r‖²
E_total(q) = Σ_r E_r(q_r)
```
K-branch seeding picks a role for each branch and settles `q_r` in its
own role energy. Branch collapse becomes *geometrically impossible* —
role-r branch and role-s branch live in disjoint additive terms.
**Why this is the strongest idea.** Three of six briefs (resonator,
MHN-variants, TEM) converge on this from independent literatures:
- Resonator brief R1: this is exactly the Hsu/Frady 2024 attention-based
  resonator update ([arXiv:2403.13218](https://arxiv.org/abs/2403.13218)),
  which is the first FHRR-native resonator.
- MHN brief Idea A: same form, framed as "role as a coordinate of
  the energy function."
- TEM brief: TEM's hippocampal `M` is a Hebbian store indexed by
  per-role `g_t`; per-role codebooks are the FHRR translation.

**Phase-5 graduation prediction.** `ΔE = E_content − E_role > 0` becomes
*structural*: the role-r minimum is the role-filler basin; the content
minimum is the content basin. `hit_role` should leap because retrieval
is directly over `X_r`. `rank_role` should drop sharply.
**How to explore.**
1. Build the per-role codebooks from the existing slow-store by
   indexing atoms by which role they have most often bound with
   (Hebbian assignment, not arbitration — frequency-weighted).
2. Implement the energy as a sum of per-role MHN retrieves.
3. Run the report-061 control matrix.
**Anti-homunculus check.** ⚠️ **CONDITIONAL (role-bank construction
must be substrate-dynamic, atom→role assignment must be soft
frequency attribution)** per reviewer agent audit 2026-05-24. The
*runtime* path is clean: each branch is gradient flow on its own
additive term `E_r(q_r)`; role is a coordinate of the energy, not a
switch in front of it. The conditions live in two places:

1. **Role-bank construction.** The role inventory `R` must emerge from
   the substrate's own binding statistics — not be discovered by an
   offline clustering with hyperparameters set by reading a
   separability score. A clustering with `k` chosen by an elbow rule
   imports a controller. Roles should be substrate entries that happen
   to be heavy binding keys (emergent codebook → role-bank by usage
   statistics), the same way atoms become schemas today.
2. **Atom → role assignment.** Implement as **soft frequency
   attribution**:
   `w_{i,r} = count(atom_i ⊛ role_r) / Σ_{r' ∈ R} count(atom_i ⊛ r')`
   with atom_i contributing to every per-role energy `E_r` with weight
   `w_{i,r}`. **No hard thresholds, no tie-break rules, no
   thresholded ownership.** An atom that has bound 70/30 between two
   roles contributes to both their codebooks, weighted accordingly.

The full dynamic-form specification is in
[2026-05-24-phase5-p1-role-bank-dynamic-form.md](../../notes/notes/2026-05-24-phase5-p1-role-bank-dynamic-form.md).
That spec must be reviewer-agent audited and merged BEFORE P1
implementation begins.

**Failure modes that would re-introduce arbitration.**
- `if hit_role(atom_i, role_r) > threshold then assign atom_i to X_r`
  (thresholded ownership)
- `if atom_i has equal frequency with role_r and role_s, route it to
  whichever currently has fewer atoms` (tie-break rule)
- Role discovery via "run k-means with k chosen by an elbow rule"
  (offline metric-driven categorization)
- Capacity reallocation across roles based on usage ("`X_r` is
  filling up, expand it")
**Cost.** Medium — two weeks. Needs per-role codebook bookkeeping
(deferred but compatible with the existing emergent codebook). Pure
Python implementable. Hebbian for runtime preserved (Hebbian per
codebook).
**Sources.** Hsu/Frady 2024; Hersche IBM 2024; MHN brief; resonator
brief R1.

#### Idea P2: Resonator with deflation (R)

**What.** Drop K parallel branches entirely. Run **one** 2-factor
resonator (`role`-codebook = slow-store; `content`-codebook = active
codebook) that iteratively unbinds:
```
role_t+1     = MHN.retrieve(slow_store, unbind(cue, content_t),  β)
content_t+1  = MHN.retrieve(fast_store, unbind(cue, role_t),     β)
```
On convergence, subtract `bind(role*, content*)` from the cue and
re-run. The "branches" of Phase 5 become **sequential deflations**, not
parallel seed conditions.
**Why now.** The resonator brief surfaces a load-bearing finding: the
resonator literature has *never* used parallel-priors-on-same-cue as
its multi-hypothesis mechanism. It uses deflation across passes, or
noise across restarts. Phase 5's design shape has no analogue in this
literature — consistent with its empirical collapse.
**Anti-homunculus check.** ✅ **PASS (unconditional, on its own
terms)** per reviewer agent audit 2026-05-24. Deflation is closed-form
vector subtraction; the coupled fixed-point iteration is two MHN
retrievals composed (gradient flow on a coupled energy); no
arbitration anywhere. The fixed budget of deflation steps with
energy-trajectory reporting is what keeps the discipline.
**Failure mode that would re-introduce arbitration.** Adding
`if residual energy < ε then stop deflation` is exactly the
controller P2 avoids by running fixed budget. Keep the budget fixed;
report the energy trajectory; let downstream consumers measure it.
**Cost.** Medium — replaces the K-branch loop with a coupled
fixed-point iteration. The 2024 attention-based variant is one line
on top of `MHN.retrieve()`.
**Sources.** Renner et al. 2024 Nature MI ([arXiv:2208.12880](https://arxiv.org/html/2208.12880v4));
resonator brief R2.

#### Idea P3: Input-driven plasticity (saliency) — landscape, not logits (A)

**What.** Implement Betteti et al. 2025
([Science Advances](https://www.science.org/doi/10.1126/sciadv.adu6991);
arXiv:2411.05849): replace the log-prior softmax bias with a per-atom
saliency `α_μ = ⟨ξ^μ, u⟩` (cue-derived) that enters the dynamics as a
Hadamard gate on the retrieval: `τẏ = -y + M_y Ψ(x ⊙ α)`. The prior
becomes a *landscape modulator* instead of a logit bias.
**Why now.** This is the only published mechanism that does exactly
what Report 061 identified as the missing ingredient. It is, in the
active-inference brief's framing, "the IDP paper independently
invented the mechanism the project needs."
**Anti-homunculus check.** ✅ **PASS (cleanest of the Tier-1
proposals)** per reviewer agent audit 2026-05-24. α is per-atom local
(inner product `⟨ξ^μ, u⟩`); `x ⊙ α` is element-wise; the dynamics
`τẏ = -y + M_y Ψ(x ⊙ α)` is gradient flow on a reshaped energy
landscape. The 2026-05-09 note's distinction between "logits" and
"landscape" (the project's own Path C failure mode) maps directly
onto IDP.
**Failure mode that would re-introduce arbitration.** Hard clamps on
α as stability hacks (`if α_μ < 0, set to 0; if α_μ > 1, set to 1`)
are sneaky controllers. Use smooth activations (sigmoid, softplus) or
a learned scale, never hard clamps.
**Cost.** Small — single-cue substrate change; existing K-branch loop
is preserved.
**Sources.** Betteti et al. 2025; active-inference brief Idea A.

#### Idea P4: EqProp consolidation pass (E)

**What.** Add an offline-batch **equilibrium-propagation** pass to
consolidation. For each role-bound cue `r⊗f` in the replay buffer:
- *free phase*: substrate settles on cue alone → `s_free`
- *nudged phase*: settle with a small λ-scaled pull toward true filler `f*` → `s_nudge`
- atom update: `Δp_k ∝ (1/λ)[(p_k·s_nudge) s_nudge − (p_k·s_free) s_free]`
This is Movellan-style contrastive Hebbian on the continuous Hopfield —
proved equivalent to BPTT in the limit, but using only the substrate's
own settling dynamics. No backprop graph, no auxiliary module.
**Why now.** The Hopfield-Fenchel-Young result (Santos et al. JMLR
2025) says exact retrieval requires *margin* between basins. Hebbian-
only training has zero margin. EqProp is the most-local error-driven
rule that exists for energy networks. It runs in the offline pass the
project's rules already permit.
**Anti-homunculus check.** ⚠️ **CONDITIONAL (`f*` provenance must be
the replay-binding record, NOT re-retrieval)** per reviewer agent
audit 2026-05-24. Both phases are pure settling on the substrate; λ
is a local geometric force; the CHL update is local. The conditional
gate is the **chicken-and-egg risk** on `f*`:

- **PASS case**: `f*` is recovered from the replay sample's binding
  record — the replay buffer stores the un-bound filler vector
  alongside the bound atom (or sufficient provenance to recover it
  without re-running the substrate). Structurally equivalent to a
  supervised target in contrastive-Hebbian setups; data-driven, not
  metric-driven.
- **FAIL case**: `f*` is produced by running a Hebbian retrieval on
  the same substrate being trained. Then `f*` is no longer ground
  truth — it's self-bootstrapped, and the update becomes "make the
  substrate more confident in whatever it currently retrieves." This
  is the EBM-equivalent of a self-fulfilling supervisor.

**Commit gate before P4 lands.** Confirm the replay buffer stores
filler-vector provenance. If it doesn't, sequence P4 *after* P1 or
P2 so unbinding is reliable, then revisit. (This is exactly the
sequencing the brainstorm already recommends — now binding.)
**Failure modes that would re-introduce arbitration even if `f*` is
clean.** `if ‖s_free − s_nudge‖ < ε skip nudged phase`, or λ scaled
by any confidence metric. Keep λ a fixed substrate parameter.
**Cost.** Medium — adds a second settling phase per replay sample.
Pseudocode is in EBM brief Lead A.
**Sources.** Scellier & Bengio 2017
([arXiv:1602.05179](https://arxiv.org/pdf/1602.05179));
Movellan 1991 CHL; EBM brief Lead A.

#### Idea P5: Transition operators (the TEM-of-FHRR) (T)

**What.** Add a small bank of FHRR **transition vectors** `{T_a}` to
the codebook (e.g. "next-token," "same-window-different-position,"
"role-shift"). Maintain an internal `g_t` slot updated by `g_t = T_a ⊛
g_{t-1}` unconditionally. Bind `p = g_t ⊛ x_t` when consolidating.
**Why now.** TEM's `g_t` is updated by `W_a g + b` and that's *what
gives g a manifold geometrically distinct from x*. The project's
substrate has no transition generator, so there is no signal shaping
the structural representation away from the content representation.
This is the smallest change that introduces TEM-style path-integration
into FHRR.
**Anti-homunculus check.** ⚠️ **CONDITIONAL (`T_a` selection must be
data-driven or fixed-prior-sampled, NOT metric-based)** per reviewer
agent audit 2026-05-24. The brainstorm's original check covered
*application* (yes, applied every step) but not *selection*. Three
readings of `T_a` selection, with different verdicts:

1. **Fixed sequence** (always "next-token"): trivially clean, but
   defeats the purpose — TEM's whole point is that different actions
   produce different transitions.
2. **Sample from the input stream** (`a_t` is given by what
   transition actually happened in the data — which token came next,
   which window-shift occurred): ✅ PASS. This is TEM's grammar:
   action is an input-stream variable, not an internally-computed
   decision.
3. **Metric-based selection** (e.g. `if cue entropy high, apply
   T_role-shift`): ❌ FAIL. P5 becomes a homunculus.

**Commit gate before P5 lands.** Specify in the design that `T_a` is
selected by whatever produced the transition in the input data, or
sampled from a fixed prior — never by reading substrate state.
**Failure modes that would re-introduce arbitration.** A "smart"
transition selector that picks `T_a` to minimize predicted surprise,
maximize energy descent, or balance codebook capacity is exactly the
controller. Keep selection input-driven or fixed-prior-sampled.
**Cost.** Small — one extra FHRR multiply per consolidation step. No
new modules.
**Sources.** Whittington et al. 2020 Cell (TEM); TEM brief Idea A.

---

### Tier 2 — Larger commitments that combine multiple Tier-1 ideas

These are the "if we're going to refactor, refactor toward this"
candidates. Don't do these until Tier 0/1 has confirmed which mechanism
gap is load-bearing.

#### Idea M1: Per-role codebooks + cross-K slot competition + IDP saliency

Stacks P1 + D3 + P3. Per-role energies give role-distinct basins; cross-K
softmax keeps branches non-redundant; IDP saliency makes the cue
modulate the landscape per-retrieval. Cost: 4-6 weeks; the
highest-leverage architectural commit.

**Anti-homunculus check.** ⚠️ **CONDITIONAL (inherits P1 + D3
conditions)** per reviewer agent audit 2026-05-24. P3 is clean
unconditionally. P1 is clean conditional on role-assignment-as-soft-
frequency-attribution + substrate-dynamic role-bank construction (see
P1 spec note). D3 is clean conditional on the joint-energy Lyapunov
check. Composition risk is mostly inherited from D3: with per-role
energies, the coupled K×N flow now needs to be checked over an
**additive Lagrangian** (HAMUX-style). Defer M1 until at least one
Tier-1 idea has demonstrated structural retrieval; the stack hides
which component does the work otherwise.
**Failure mode that would re-introduce arbitration.** A stacking
module that *routes* between P1, D3, and P3 based on which subsystem
is "active" — supervisor pattern, banned. Fix: HAMUX-style additive
Lagrangians where all three contribute to one energy with no router
in front.

#### Idea M2: EqProp + replay-shuffled negatives + DSM warm-start

Stacks P4 + EBM Lead D + EBM Lead C. Three-stage training pipeline:
warm-start the substrate with denoising score matching (no negatives,
partition-function-free); then run EqProp consolidation passes with
role-shuffled negatives. This is the "carve role-target basins by
construction" route. Cost: substantial, but each piece is well-studied.

**Anti-homunculus check.** ✅ **PASS** per reviewer agent audit
2026-05-24, *inheriting P4's conditional gate on `f*` provenance*.
Role-shuffled negatives are pure data augmentation — at training
time, construct (cue, wrong-role-filler) pairs by permuting role
assignments across a batch, metric-free. DSM warm-start is also
training-time data manipulation (corrupt-and-reconstruct), no
controller. This is structurally identical to how the project already
uses shuffled-token controls.
**Failure mode that would re-introduce arbitration.** Hard-negative
mining (`select the most-confused negatives based on current energy`)
*is* metric-driven and would turn M2 into arbitration. Use uniform
random shuffling, never hard-negative mining.

#### Idea M3: Full hierarchical resonator + emergent codebook learning

Renner et al. 2024 Nature MI hierarchical resonator extended with the
project's emergent codebook as the per-factor codebook-learning
mechanism. The Renner survey explicitly names "learning mechanisms
for the underlying generative models" as open future work; the
project's codebook is a candidate answer. **There is publishable
contribution shape here**, not just an internal rescue. Cost: large;
also the most ambitious framing.

**Anti-homunculus check.** ⚠️ **CONDITIONAL (codebook-learning rule
must remain Hebbian-online + error-driven-batch-only)** per reviewer
agent audit 2026-05-24. The runtime (resonator deflation + per-factor
MHN) inherits cleanly from P1/P2. The novel surface is the *learning
rule* for the per-factor codebooks, which has large surface area for
drift.
**Failure mode that would re-introduce arbitration.** Any controller
that meta-allocates codebook capacity across factors based on usage
metrics (`factor r codebook is filling up, expand it`) is arbitration
by definition (STATUS.md operational policy, line 33: online
error-driven codebook updates banned). Capacity must be either fixed
per factor or grow by the same substrate dynamics that grow the
single emergent codebook today (Hebbian saturation + death).
Classifier-head "decide which factor a new atom belongs to" is also
arbitration; assignment must remain frequency-attribution per P1's
spec.

---

## Cross-cutting themes

1. **The substrate, not the prior, is what needs to change.** Five of
   six briefs make the same architectural point: log-prior biasing
   reshapes logits without creating basins. The fix lives in the
   energy function (per-role codebooks, sparsemax supports, IDP
   saliency, EqProp margin) or in the storage rule (pseudo-inverse,
   Storkey), not in the prior weight.
2. **Cross-coupling between branches is missing.** The slot-attention
   diagnosis sits on top of the per-role one: even with role-distinct
   energies, K-branch retrieval still runs branches independently.
   Cross-K normalization is the smallest additional change that gives
   branches a *reason to differ*. It's substrate-pure (softmax) and
   anti-homunculus-clean.
3. **The replay buffer is already 80% of a PCD buffer.** EBM brief's
   strongest observation — the offline-batch consolidation pass the
   project already has is mechanically compatible with EqProp,
   denoising score matching, contrastive divergence, and role-shuffled
   negatives. Adding a negative phase requires a sampler, not new
   infrastructure.
4. **The 5.5e-3 magnitude floor is softmax-specific.** Under sparsemax
   (Hopfield-Fenchel-Young α=2) retrieval is *exact* on sufficiently
   sparse substrates. The noise-floor argument may not survive a
   sparsemax substrate. Worth confirming before locking the floor as
   the gating threshold for the post-Tier-1 substrate.
5. **The project is closer to the frontier than the framing suggests.**
   Three briefs independently noted: there is no published work
   combining FHRR with active inference; no published work fusing
   slot-style cross-K with Hopfield settling; and the 2025 resonator
   survey explicitly names "learning the generative model" as open
   future work. The project's emergent codebook is the missing piece
   for several published research lines, not a rescue for an internal
   problem.

---

## Challenges and counterarguments

- **Per-role codebooks (P1) need a role bank.** The project does not
  currently maintain an explicit role inventory. The simplest move is
  to derive role identifiers from FHRR position vectors and recurring
  binding patterns in replay; the deferred consolidation-geometry
  regime classifier could double as the role-assignment mechanism.
  If role assignment is unstable or arbitrary, the per-role energies
  fragment the codebook without giving structural retrieval.
- **EqProp (P4) requires a "true filler" target.** Replay samples
  contain the full bound atom; extracting the filler given the role
  requires a clean unbinding, which is the very thing P1/P2 are
  supposed to provide. There is a chicken-and-egg risk; the project
  should warm-start P4 only after P1 or P2 is in place.
- **Sparsemax (D2) can be too sparse on small substrates.** Capacity
  is sub-linear in N for hard sparsity. The α-entmax family gives a
  tunable knob, but the calibration is non-trivial and overlaps with
  the parked θ′(β) calibration spike.
- **Cross-K softmax (D3) changes the energy.** The K-branch settling
  loop is no longer K independent gradient flows; it is one coupled
  flow over K×N. This needs an explicit Lyapunov check before claiming
  it's anti-homunculus-clean by analogy to slot attention. Worth a
  reviewer-agent pass before landing the PR.
- **IDP (P3) shares Hsu/Frady's β as a knob.** The IDP paper is on a
  Hopfield substrate, not on FHRR; the FHRR adaptation is novel and
  may have its own instabilities. Frame as a spike, not a commit.
- **Tier-2 stacks are ambitious.** Don't commit to M1/M2/M3 until at
  least one Tier-1 idea has produced a clean ΔE pass with structural
  retrieval (`hit_role` materially > random). Otherwise the stack hides
  which component is doing the work.

---

## Rabbit holes worth following

- **Bakermans / Whittington / Behrens 2025 Nature Neuro**
  ([PMC12081289](https://pmc.ncbi.nlm.nih.gov/articles/PMC12081289/))
  identifies replay as the carver of remote relational fields for novel
  compositions. The project's replay does not currently condition on
  transitions/compositions. This is a small architectural change with
  large theoretical payoff and tight conceptual link to Idea P5.
- **SysBinder (Singh et al. ICLR 2023,
  [arXiv:2211.01177](https://arxiv.org/abs/2211.01177))** does emergent
  factor binding via block-structured slots, *without supervision*.
  Closest existing analogue to Phase 5's role-vs-content distinction.
  If you go path S, read this carefully — block-slot decomposition
  of FHRR state (slot brief Idea 5) is its direct translation.
- **In-context denoising (Smart 2025 ICML,
  [arXiv:2502.05164](https://arxiv.org/abs/2502.05164))** — an existence
  proof that training MHN with a denoising score objective places
  basins where you train for. Direct evidence for the EBM brief's
  proposal.
- **Bono et al. 2023 eLife "Neural learning rules for SR"** —
  spike-timing-dependent plasticity with theta phase precession
  produces successor-representation-like firing fields. If P5 is
  promising, this gives the local plasticity rule.
- **Compositional Sparse Coding + Resonator
  ([arXiv:2404.19126](https://arxiv.org/html/2404.19126))** —
  cleanly separates offline codebook learning from online resonator
  factorization. Maps exactly onto the project's "Hebbian online,
  error-driven only in batch offline passes" rule.
- **HAMUX (Krotov et al. 2025,
  [arXiv:2507.06211](https://arxiv.org/abs/2507.06211))** —
  convex-Lagrangian additive energies. Could be the framework for
  stacking per-role + per-atom Benna-Fusi-α + content energies as
  one Lyapunov function (links the parked Benna-Fusi α idea to P1).

---

## Post-spike addendum (2026-05-24, written after the spike wave)

The "My recommended first move" section below was written *before* the
Tier-0 spike wave ran. It is now historically accurate but stale as
guidance. **The spike wave executed the recommended D1 + D3 + S1 wave
plus an additional E1 ("Path C done right") closure spike from a
GPT-generated suggestion**:

- D1 (pseudo-inverse storage swap) — null on `hit_role`; `rank_role` worsens 205→403 ([Report 062](../../reports/062_phase5_spikes_d1_d3_local_smoke.md))
- D3 (cross-K softmax, additive form per [Lyapunov pass](../../notes/notes/2026-05-24-spike-D3-lyapunov-analytical.md)) — null on `hit_role`; ΔE negative 0/3 seeds ([Report 062](../../reports/062_phase5_spikes_d1_d3_local_smoke.md))
- E1 (zero-mean role/content asymmetric logit field) — null on `hit_role` at every λ; `rank_role` monotonically worsens 205→498 ([Report 063](../../reports/063_phase5_spike_e1_centered_log_prior.md))
- S1 (replay-trace schema check) — encoder-time provenance missing; P1 needs ~30 LOC schema extension ([note](../../notes/notes/2026-05-24-spike-S1-replay-trace-schema.md))

The decision-recipe outcome routes to Tier-2 (Path D = training-time
intervention). The single-source synthesis for the next session is
[notes/notes/2026-05-24-phase5-session-close-and-next-moves.md](../../notes/notes/2026-05-24-phase5-session-close-and-next-moves.md),
which captures the three remaining options (M2 RECOMMENDED, M1, closure
paper) with reasoning. Read that and STATUS.md, not the section below,
when picking up next session.

The "My recommended first move" section below is preserved verbatim for
historical traceability.

---

## My recommended first move (HISTORICAL — pre-spike; superseded by post-spike addendum above)

**Day-1 work**: Implement D1 (pseudo-inverse storage swap) and D3
(cross-branch softmax). These are orthogonal, each is a one-PR change,
each isolates a different mechanism gap, and together they tell you
whether Tier 1 should go path T/E/M (storage geometry) or path S
(branch coupling) or P1 (both, per-role codebooks).

**Week-1 work, conditional on D1/D3 outcome**: Idea P1 (per-role
coupled MHN codebooks) is the strongest single bet because it has
three independent literature confirmations (R, M, T) and addresses
the diagnosis at the right altitude (the energy function). If D1
shows storage geometry is load-bearing, P1 includes the fix. If D3
shows branch coupling is load-bearing, P1 provides the role-distinct
basins for coupling to work over.

**Hold in reserve**: P4 (EqProp consolidation) is the cleanest
training-time intervention and should be added *on top of* P1 once
P1 has demonstrated structural retrieval. EqProp without role-distinct
basins to push margin between is shooting in the dark; with them, it
should drive `hit_role` and `rank_role` toward proper structural
retrieval.

**Update STATUS.md** with a third path: "Path D = substrate-shape
rescue (per-role codebooks)" alongside the open Path A/B'/C choice.
This brainstorm's strongest claim is that the path decision is
mis-framed: the architecture has been treating the missing role
basin as a tuning problem; the literature treats it as a structural
property of the energy function.

---

## Sources

Comprehensive lists are in the individual research briefs. The
load-bearing citations for the headline argument:

- **Per-role coupled MHN**: [Hsu et al. arXiv:2403.13218](https://arxiv.org/abs/2403.13218) (FHRR-native attention-based resonator)
- **Hopfield-Fenchel-Young / sparsemax**: [Santos et al. arXiv:2411.08590](https://arxiv.org/abs/2411.08590) (JMLR 2025)
- **Slot attention as the source of K-branch differentiation**: [Locatello et al. arXiv:2006.15055](https://arxiv.org/abs/2006.15055) (NeurIPS 2020)
- **Input-driven plasticity (saliency as landscape modulator)**: [Betteti et al. 2025 Science Advances / arXiv:2411.05849](https://www.science.org/doi/10.1126/sciadv.adu6991)
- **HRR role-filler retrieval via pseudo-inverse**: [Kymn/Stewart 2022 Sci Reports PMC9759586](https://pmc.ncbi.nlm.nih.gov/articles/PMC9759586/)
- **TEM**: [Whittington et al. 2020 Cell PMC7707106](https://pmc.ncbi.nlm.nih.gov/articles/PMC7707106/)
- **Equilibrium propagation**: [Scellier & Bengio 2017 arXiv:1602.05179](https://arxiv.org/pdf/1602.05179)
- **EBM training of MHN by denoising**: [Smart 2025 ICML arXiv:2502.05164](https://arxiv.org/abs/2502.05164)
- **Energy Transformer**: [Hoover et al. NeurIPS 2023 arXiv:2302.07253](https://arxiv.org/abs/2302.07253)
- **Hierarchical resonator (deflation)**: [Renner et al. 2024 Nature MI arXiv:2208.12880](https://arxiv.org/html/2208.12880v4)
- **Replay as relational-field carver**: [Bakermans et al. 2025 Nature Neuro PMC12081289](https://pmc.ncbi.nlm.nih.gov/articles/PMC12081289/)
- **Resonator survey naming open work**: [Renner/Kymn/Frady/Sommer 2025 OpenReview FNrZd3Ls1d](https://openreview.net/forum?id=FNrZd3Ls1d)
- **HAMUX additive Lagrangians**: [Krotov et al. 2025 arXiv:2507.06211](https://arxiv.org/abs/2507.06211)
