# Research Brief E — Cue, Binding-Algebra, and Dimension Reformulations

**Scope.** Phase 5's role-fidelity-uniformity finding ([report 050](../../../reports/050_phase5_beta_smoke_seed17.md)):
all 1064 atoms at D=4096 have `f_i = 0.9858 (std = 0.0000)` because the FHRR unbind
crosstalk noise floor `≈ 1 − 1/√D` dominates any per-atom signal. The four
2026-05-20 design notes (A+B → A1 → A1' → β) all live on the substrate side. This
brief enumerates the other three axes — **cue construction, binding algebra,
substrate dimension** — and ranks reformulations by `expected impact / cost`.

**Frame.** `f_i = mean_{j≠i}(1 − |G_jk|)` is a pairwise-cosine quantity on
*unbound* roles. The structural near-1 value at D=4096 is the
1-vs-population SNR of HRR cross-talk variance, exactly the cell predicted by
Plate / Krotov-Hopfield 2016 (`P_error ∝ K/N^(n−1)` at n=1) and by the
Ganesan et al. 2021 noise analysis. **The metric is not broken; the SNR cell
is wrong.** Three independent levers move it: dimension D, binding algebra
(which sets n and the cross-talk distribution), and cue regime (which sets
the operating point where f_i is measured).

---

## Section 0 — What "fixes role-fidelity-uniformity" means

A reformulation passes if it produces:

1. **Variance:** `std(f_i) >> 0` across atoms in the same retrained substrate
   (so the β prior can actually discriminate between schemas).
2. **Signal-floor separation:** the mean of f_i for in-substrate atoms is
   bounded away from the noise floor for random atoms by at least one
   substrate-noise standard deviation.
3. **Cheap-test:** a clear analytic prediction of when the variance appears
   (so we don't burn n=10 Colab seeds to discover the regime is still
   noise-dominated).

The current substrate at D=4096, FHRR-convolution-binding, fixed-cue (0.05,
0.6) fails (1) and we know (3) only after the fact.

---

## Section 1 — Dimension axis (lowest-friction lever)

### 1.1 D=512 (or D=1024) — the obvious move

**Formulation.** Drop substrate from D=4096 to D=512. Identical FHRR
algebra, identical phasor encoding, just narrower phase vector. The unbind
noise floor moves from `1 − 1/√4096 = 0.9844` to `1 − 1/√512 = 0.9558`.

**Predicted f_i behavior.** The role-fidelity *mean* falls (from ~0.984 to
~0.956), but more importantly the per-atom *variance* rises. At D=512 the
1064-atom substrate is at ~2× its theoretical bundle capacity (≈ 0.14·512 ≈
72 for Hopfield, larger for FHRR; Plate's per-pattern capacity for FHRR
binding ≈ D/(4·ln(N)) bits at unit noise — so ~512/27 ≈ 19 reliably
distinguishable pairs per atom). Atoms whose neighborhoods are
near-orthogonal stay near 0.956; atoms in dense local clusters drop to 0.85
or below. That spread is what β needs.

**Implementation cost.** Drop-in. Change `TorchFHRR(dim=4096)` →
`TorchFHRR(dim=512)`. Phase 4 D1 graduation needs re-verification at the
new dim — risk: D=512 may not have enough capacity for the corpus.

**Anti-homunculus check.** PASS by construction. Dimension is set once at
substrate-construction time; no controller adapts it during training. The
substrate's energy landscape is geometrically different, but the dynamic
form of every existing mechanism is preserved.

**Verdict.** **Yes — this fixes role-fidelity-uniformity at lowest cost.**
It does not fix it elegantly (it pays in capacity), but the math is direct
and the implementation cost is zero substrate-code lines.

**Falsification.** Run report-050's f_i computation on a D=512 retrained
substrate. If `std(f_i) > 0.02` (an order of magnitude over D=4096's
0.0000), the dimension-induced variance hypothesis stands; if not,
something else is suppressing variance and the substrate isn't the issue.

---

### 1.2 Multi-scale: D=512 *alongside* D=4096

**Formulation.** Two substrates run in parallel, sharing the codebook but
not the binding layer:
- D=512 substrate: high-SNR per-atom role-fidelity measurement
- D=4096 substrate: existing capacity for the binding store

The β prior reads f_i from the D=512 measurement; the energy-weighted
bundle and retrieval stay on D=4096. The codebook is the bridge — same
role identifiers map to D=512 phasors and D=4096 phasors via two
independent random projections (or two registered hashes of the codebook).

**Predicted f_i behavior.** Variance at D=512 (per above) is the measured
signal. Inference still happens on the high-capacity D=4096 substrate, so
no capacity is lost. **This is the form the project's anti-homunculus
discipline actually wants:** the diagnostic and actuator live on different
geometric objects; the higher-D substrate doesn't "know" its f_i is
being measured on a smaller substrate.

**Implementation cost.** Two-substrate wiring in `replay_loop.py` and
`consolidation.py`. Synchronized add_pattern calls. Cost: ~1 day of
plumbing. The diagnostic on D=512 is *new code* but the formula is
literally `compute_role_fidelity()` from
[role_fidelity.py](../../../src/energy_memory/phase5/role_fidelity.py).

**Anti-homunculus check.** PASS. Each substrate's energy is internal;
the two substrates' coupling is via the shared codebook (a static
substrate-construction-time object), not via a controller. The "decision"
of which f_i value to use is a static architecture choice, not a runtime
arbitration.

**Verdict.** **Highest impact / cost ratio in this brief.** Preserves the
existing D=4096 capacity *and* gives β the variance signal it needs.
Closest in shape to the project's diagnostic-actuator-pair discipline.

**Falsification.** Same as 1.1 — std(f_i) > 0.02 on the D=512 substrate.
Plus a sanity check that the D=4096 substrate's Phase 4 D1 graduation
metric (Δms_w3) is unchanged.

**Note on prior art.** Hierarchical/multi-scale VSA is an open area; the
Schlegel et al. 2022 comparison survey ([arxiv 2001.11797](https://arxiv.org/abs/2001.11797))
catalogues 11 VSA variants but does not study multi-D layered substrates.
The 2025 "Attention as Binding" paper ([arxiv 2512.14709](https://arxiv.org/abs/2512.14709))
treats VSA structure across DNN layers but at a different abstraction. **The
project would be doing something genuinely novel here**, but the
machinery is simple — two FHRR substrates with synchronized add calls.

---

## Section 2 — Binding-algebra axis (medium-friction lever)

### 2.1 Permutation-binding (Kanerva MAP / Sahlgren / Plate-permutation)

**Formulation.** Replace `bind(role, filler) = role * filler` (FHRR
Hadamard product of phasors) with `bind(role, filler) = permute(filler,
hash(role))` — the role is a *permutation index*, not a vector. Unbinding
is `unbind(bound, role) = permute(bound, -hash(role))` — **exactly
invertible, zero noise**. Bundling is unchanged (sum of bound vectors).
The project already has `TorchFHRR.permute(vector, shift)`
([torch_fhrr.py:86-97](../../../src/energy_memory/substrate/torch_fhrr.py#L86)).

The Recchia et al. 2015 study ([PMC4405220](https://pmc.ncbi.nlm.nih.gov/articles/PMC4405220/))
shows random-permutation binding **beats** circular convolution on
paired-associate retrieval (M=457 vs 381 successful retrievals,
F(1,48)=11.85, p=0.001) at D=2048. Permutation binding is also robust to
input distribution (Gaussian vs sparse ternary: nearly identical accuracy).

**Predicted role-fidelity behavior.** Under permutation binding, the
unbind operation is *deterministic and exact*. The role-fidelity noise
floor for a single (role, filler) pair is zero. **Cross-talk in bundles
of W=4 bindings remains** — that's the binding-count cross-talk Plate
analyzes — but the per-role contribution is no longer mixed with FHRR
phasor jitter. f_i_perm = mean_{j≠i}(1 − |G_jk|) is now measuring whether
the *bundle* leaks role j's content into role i's unbinding, which is
analytically a sum of W−1 random-permuted vectors projected onto the
target role — variance ∝ (W−1)/D, *not* the FHRR phasor-cross-talk floor.

**At W=4, D=4096:** noise variance ≈ 3/4096 ≈ 7.3e-4; std ≈ 0.027.
**Per-atom variance in f_i appears naturally** because atoms in dense
local clusters bundle near-aligned content, while orthogonal-neighborhood
atoms don't. **This is the closest in spirit to what the project wanted
from β.**

**Implementation cost.** Medium. The substrate has `permute()` already;
the bind/unbind calls in `replay_loop.py` and `phase5/role_fidelity.py`
need conditional dispatch on binding type. Phase 3 codebook semantics
change: the codebook now indexes permutation hashes, not phasor vectors.
Phase 4 D1 needs full re-verification. Cost: ~2-3 days; touches the
substrate-binding boundary that the project has not touched since
Phase 0.

**Anti-homunculus check.** PASS. The binding algebra is a static
substrate property, not a runtime decision. No controller chooses
between FHRR and permutation; the substrate is constructed once with
its binding type.

**Verdict.** **Yes — this most directly addresses the role-fidelity-
uniformity pathology.** The FHRR phasor-cross-talk floor is the
*algebraic* source of the uniformity; permutation binding removes
exactly that term. Recchia et al.'s empirical superiority on retrieval is
independent corroboration.

**Caveat.** Permutation binding loses one HRR property: bundles of
permutation-bound pairs are *not* themselves permutation-invertible
without remembering the role list. The project already separates role
list from substrate (codebook), so this caveat doesn't bite.

**Falsification.** Run report-050's f_i on a permutation-bound substrate.
Predicted std(f_i) ≥ 0.02; predicted mean(f_i) bounded by the
W=4-bundle-crosstalk floor `1 − (W−1)/D ≈ 0.9993` (so the *mean* is
*higher* than FHRR's 0.9858, but the *variance* finally exists).

---

### 2.2 MAP (Multiply-Add-Permute, bipolar Hadamard)

**Formulation.** Vectors are bipolar (±1) in R^D; binding is element-wise
multiplication (each vector is its own inverse: `bind(a, b) * b = a`).
The project's FHRR is the complex-valued analogue; **MAP is the
real-valued specialization** with discrete components. Reports
3-4× speedup vs HRR with FFT-convolution (Schlegel survey).

**Predicted role-fidelity behavior.** Bipolar Hadamard binding has the
same n=1 cross-talk structure as FHRR — both are element-wise products —
so the noise-floor scaling is the same `1 − 1/√D`. **Does NOT fix the
role-fidelity-uniformity pathology** because the underlying SNR cell is
identical.

**Implementation cost.** Medium (substrate rewrite, but mostly mechanical).

**Anti-homunculus check.** PASS.

**Verdict.** **No — same noise floor as FHRR.** Cited for completeness;
not a candidate.

---

### 2.3 Sparse Block Codes (Frady, Kleyko, Sommer 2021 —
[arxiv 2009.06734](https://arxiv.org/abs/2009.06734))

**Formulation.** Vector is D-dimensional but partitioned into B blocks of
D/B components each, with exactly *one* nonzero element per block.
Binding is **block-local circular convolution**. The sparsity is `s =
B/D`. Theoretically equivalent to dense FHRR via tensor-product
unfolding, but stored compactly.

**Predicted role-fidelity behavior.** "Variable binding for block-codes
has ideal properties, whereas binding for general sparse vectors also
works, but is lossy" (Frady et al.). The "ideal" claim is that block-
local convolution is exactly invertible per-block. **At equal D and
W=4, sparse block-codes have a lower noise floor than dense FHRR** by a
factor of √(s) — the per-block crosstalk only sees D/B dimensions of
mixing, not D.

**Implementation cost.** High. The block structure is fundamental;
sparse storage, block-wise FFT, sparse codebook all need to be added.
**This is a substrate rewrite, not a drop-in.** Phase 0-4 all need
re-verification.

**Anti-homunculus check.** PASS.

**Verdict.** **Yes for the long term, no for Phase 5 graduation.**
Frady's "ideal binding properties" claim is *exactly* what would close
the uniformity pathology, but the cost is months of work. Park as a
Phase 7+ direction.

---

### 2.4 HLB — Walsh-Hadamard-Derived Linear Binding (Tovsky 2024,
NeurIPS — [arxiv 2410.22669](https://arxiv.org/abs/2410.22669))

**Formulation.** After a projection step, binding reduces to element-
wise multiplication; unbinding is element-wise division. The projection
step gives a noise term η^π < η^◦ (the standard binding noise). O(d)
complexity. SOTA on extreme multi-label classification.

**Predicted role-fidelity behavior.** The noise term is reduced relative
to HRR but the *scaling form* with D is similar to FHRR. The paper
demonstrates task-level improvements rather than noise-floor numerics
at fixed D. Uncertain whether the std(f_i) > 0 condition is met.

**Implementation cost.** Medium. The structural similarity to
element-wise multiplication makes adaptation to the FHRR substrate
clean, but the projection step changes the substrate's state-space
shape.

**Anti-homunculus check.** PASS.

**Verdict.** **Maybe — needs an empirical noise-floor measurement
before committing.** Not high enough confidence to displace 1.2 or 2.1.

---

### 2.5 VTB — Vector-Derived Transformation Binding (Gosmann &
Eliasmith 2019)

**Formulation.** Binding is matrix-vector multiplication where the
matrix is constructed from the second argument. List capacity on par
with HRR; stack capacity better. "Influences vector length less, which
benefits neural implementation."

**Predicted role-fidelity behavior.** Compared to FHRR, list-encoding
capacity is parity, so the per-binding noise floor is similar. The
stack advantage suggests *better hierarchy* but not better single-role
fidelity. Likely **does not fix the uniformity pathology** — same SNR
cell.

**Implementation cost.** High (matrix-construction-and-multiplication
per binding; doesn't match the project's complex-phasor substrate).

**Anti-homunculus check.** PASS.

**Verdict.** **No.** Better for hierarchy, not the role-fidelity floor.

---

## Section 3 — Cue-regime axis (lowest-friction, often-overlooked)

### 3.1 Cue regime sweep

**Observation.** The project's `binding_noise_std=0.05` and
`content_distortion=0.6` are not justified anywhere I can find in the
codebase notes. They were picked once, never swept.

**Formulation.** Run report-050's f_i computation across a 2D sweep:
- `binding_noise_std ∈ {0.0, 0.01, 0.05, 0.10, 0.20, 0.40}`
- `content_distortion ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 0.95}`

For each cell, compute std(f_i) across the substrate's atoms.

**Predicted behavior.** At low noise (0.0, 0.0), the unbind is exact and
f_i is identically 1 for every atom (still uniform, but at the ceiling).
At very high noise (0.40, 0.95), f_i is identically the random-vector
floor `1 − 1/√D` (uniform at the floor — exactly report 050's regime).
**There may or may not be an intermediate regime where atoms'
neighborhoods matter** — i.e., where the cue is informative enough to
discriminate dense from sparse local geometry but not so weak that
crosstalk dominates. **If such a regime exists, the project has just
been operating outside it.**

**Implementation cost.** **Hours.** No substrate change. Just an
experiment script that calls `compute_role_fidelity()` across the 36
cells using the existing seed-17 substrate.

**Anti-homunculus check.** PASS. The cue regime is part of the
*experiment*, not the architecture.

**Verdict.** **Run this first.** If a non-uniform regime exists, the
project's existing 2026-05-20 β implementation works; only the cue
config changes. If no regime exists, the substrate is genuinely
saturated and we go to 1.2 or 2.1.

**Falsification.** If max_cell(std(f_i)) < 0.005, the substrate is
saturated and cue-regime is not the lever.

---

### 3.2 Alternative cue construction: corrupted-binding instead of
corrupted-cue

**Formulation.** Instead of perturbing the cue vector by `binding_noise_std`,
perturb *which roles are present in the bundle* — e.g. randomly drop one
role from the W=4 binding. The f_i now measures "can role i be recovered
when the binding has W=3 roles instead of W=4?" — a question about
*bundle composition* rather than *cue noise*.

**Predicted behavior.** Per the Krotov-Hopfield SNR formula, this
changes K (bundle size) directly, which is the strongest knob on the
cross-talk variance. At W=3 vs W=4, variance drops by 25%; per-atom
variance becomes detectable above noise floor for atoms in dense
clusters.

**Implementation cost.** Low (~half day). Modify the cue construction in
[role_fidelity.py](../../../src/energy_memory/phase5/role_fidelity.py).

**Anti-homunculus check.** PASS.

**Verdict.** **Yes — high value-per-line.** Tests a different
crosstalk regime cheaply.

---

### 3.3 Settling-based fidelity (HEN / Krotov-HAM-style)

**Formulation.** Replace pairwise-distance f_i with:
> Cue the substrate with the unbinding result; settle for k iterations;
> count the fraction of settlings that converge to the target atom
> (top-1) vs a metastable mode.

The "settling" is the substrate's own Modern-Hopfield dynamics — already
implemented in `memory/torch_hopfield.py`.

**Predicted behavior.** This is the architectural antidote from HEN
(Kashyap 2024) and HAM (Krotov 2021). **A settling that lands on the
right atom 92% of the time is a fundamentally different signal than a
pairwise distance of 0.9858** — bounded by [0,1] on a per-atom basis,
naturally variable across atoms, no D-dependence in the upper bound.

**Implementation cost.** Medium (~1 day). The settling infrastructure
exists; the f_i computation needs a new path that calls it.

**Anti-homunculus check.** PASS — this is the canonical
"diagnostic-as-actuator" form. The dynamics doing the measuring is
identical to the dynamics doing the work.

**Verdict.** **Yes — most principled answer.** This is what the
literature flag B in [literature.md](../context/literature.md) calls
"metastable-state rate under settling." It bypasses the unbind-distance
pathology entirely.

**Caveat.** Slower (k settling iterations per atom). With 1064 atoms ×
k=12 settling steps, this is ~10K Hopfield retrieve calls per f_i
measurement — feasible per the recent torch_hopfield optimization
(report mentioned in CLAUDE.md re 2026-05-15 refactor).

---

## Section 4 — Ranking by impact × cost ratio

| Rank | Candidate | Impact | Cost | Why |
|------|-----------|--------|------|-----|
| 1 | **§3.1 Cue regime sweep** | Medium | Hours | Tests whether the existing β + existing substrate works in a regime never tried. Costs nothing on the substrate side. **Run this first.** |
| 2 | **§1.2 Multi-scale (D=512 ‖ D=4096)** | High | ~1 day | Preserves D=4096 capacity; gives β a variance signal from D=512; cleanest anti-homunculus shape. |
| 3 | **§3.3 Settling-based f_i** | High | ~1 day | Principled diagnostic-as-actuator form; HEN/HAM-canonical; bypasses unbind-distance pathology entirely. |
| 4 | **§2.1 Permutation-binding** | High | ~2-3 days | Removes the algebraic source of the cross-talk floor; the project already has `permute()`; Recchia et al. show empirical superiority. |
| 5 | **§1.1 D=512 alone** | Medium | Hours-of-runtime | Same lever as §1.2 but loses capacity. Use as a falsification probe for §1.2. |
| 6 | **§3.2 Drop-a-role cue** | Medium | Half day | Cheap probe of the W knob in the cross-talk SNR formula. |
| 7 | **§2.4 HLB** | Maybe | Medium | Needs empirical noise-floor numbers before committing. |
| 8 | **§2.3 Sparse Block Codes** | High (long term) | Weeks | Right shape but substrate rewrite; park for Phase 7+. |
| 9 | §2.2 MAP, §2.5 VTB | Low | Medium-high | Same SNR cell as FHRR; no role-fidelity benefit. |

---

## Section 5 — Recommended next step

**Two-stage protocol:**

1. **Day 1 (today / next session):** Run §3.1 cue-regime sweep on the
   existing seed-17 substrate. 36 cells × ~30 s/cell = ~20 min wall.
   If `max std(f_i) > 0.02` for any cell, the cue regime is the lever;
   re-run report 050's β smoke test in that cell. STOP.

2. **Day 2 (if §3.1 fails):** Implement §1.2 multi-scale substrate. The
   D=4096 substrate is unchanged; add a synchronized D=512 substrate
   whose only job is to provide f_i to β. Re-run report 050.

3. **Day 3 (if §1.2 also fails):** §3.3 settling-based fidelity. The
   pairwise-distance formulation is replaced; β reads settling-success
   rate.

Permutation-binding (§2.1) sits behind these because it's a bigger
substrate change and the project has not invariantly broken from FHRR
since Phase 0; the order respects that risk gradient.

---

## Section 6 — Cross-axis observation

The four 2026-05-20 design notes assume the substrate is the right object
to fix. **The cue-regime sweep (§3.1) is the cheap experiment that
either confirms or falsifies that assumption.** If §3.1 shows variance,
the substrate is fine and the operating point was wrong; if §3.1 shows
no variance anywhere, the substrate genuinely doesn't have a per-atom
role-fidelity signal at D=4096 and one of §1.2 / §2.1 / §3.3 is
mandatory.

**The 5-hour cue sweep is the right next experiment, regardless of
which longer-term lever the project picks.**

---

## Sources

- Schlegel, Neubert, Protzel (2022). *A comparison of vector symbolic
  architectures.* AI Review 55(6) 4523–4555.
  [arxiv 2001.11797](https://arxiv.org/abs/2001.11797) — VSA taxonomy
  reference.
- Frady, Kleyko, Sommer (2021). *Variable binding for sparse distributed
  representations.* [arxiv 2009.06734](https://arxiv.org/abs/2009.06734)
  — sparse block codes "ideal binding properties."
- Recchia et al. (2015). *Encoding sequential information in semantic
  space models: comparing HRR and random permutation.*
  [PMC4405220](https://pmc.ncbi.nlm.nih.gov/articles/PMC4405220/) —
  random permutation beats circular convolution empirically.
- Ganesan et al. (2021). *Learning with HRRs.* NeurIPS — the complex
  unit-magnitude projection π and the noise-floor analysis FHRR
  inherits.
- Krotov & Hopfield (2016). *Dense Associative Memory.* — the SNR
  formula `P_error ∝ K/N^(n−1)` that names the noise cell.
- Vangara & Gopinath (2026). *Geometry of Consolidation.* — cap-coverage
  bound `(θ'/d̄)^(d_eff/2)` already operationalized by the project.
- Gosmann & Eliasmith (2019). *Vector-Derived Transformation Binding.*
  Neural Computation. — VTB capacity numbers vs HRR.
- Tovsky et al. (2024). *Walsh-Hadamard Derived Linear Binding.*
  NeurIPS. [arxiv 2410.22669](https://arxiv.org/abs/2410.22669) —
  HLB with reduced noise term.
- Frady & Kent (2020). *Resonator Networks.* [arxiv 2007.03748](https://arxiv.org/abs/2007.03748)
  — factor-search for bound representations; not directly used here
  but relevant if multi-role unbinding becomes the bottleneck.
