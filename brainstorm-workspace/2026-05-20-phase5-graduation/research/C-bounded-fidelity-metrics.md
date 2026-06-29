# C — Bounded, scale-free, alternative role-fidelity metrics

Date: 2026-05-20. Brief target: alternatives to the project's current per-atom
`f_i = mean_{j≠k}(1 - |G_jk|)` role-fidelity, which goes structurally
noise-dominated on the D=4096 FHRR substrate (uniform 0.9858 across 1064 atoms).

The pathology is *not* that the substrate is broken — d_eff = 35.23 is
preserved, duplicates throttled, original atoms dominate. The pathology is
that pairwise unbind cosine distance is the wrong instrument: at high D, FHRR
cross-talk variance dominates the signal, so every atom's pairwise unbind
geometry looks the same.

The fix shape: replace pairwise distance with a measurement that has
non-zero variance on this exact substrate because it is structurally
*bounded* (so the noise has nowhere to go) or *spectral* (so the noise
attenuates faster than D grows) or *relational* (so the discriminating signal
is the relationship between unbind and the things-it-isn't, not pairwise
geometry).

This brief catalogues five candidates, ranked at the end.

Working files referenced:
- Current `f_i`: `/Users/dypatterson/Desktop/Neuro-AI/src/energy_memory/phase5/role_fidelity.py:40-92`
- Anchor synthesis: `/Users/dypatterson/Desktop/Neuro-AI/notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md`
- Ganesan extracted text: `/Users/dypatterson/Desktop/Neuro-AI/tmp/pdf_text/4720_learning_with_holographic_redu.txt`
- Vangara–Gopinath extracted text: `/Users/dypatterson/Desktop/Neuro-AI/tmp/pdf_text/Geometry of Consolidation.txt`
- Papyan extracted text: `/Users/dypatterson/Desktop/Neuro-AI/tmp/pdf_text/papyan-et-al-2020-prevalence-of-neural-collapse-during-the-terminal-phase-of-deep-learning-training.txt`

---

## 1. Ganesan Jp + Jn  →  bounded query-cosine fidelity (the headline candidate)

### Source

Ganesan et al. 2021, NeurIPS, *Learning with Holographic Reduced
Representations*. Equations (3), (6), (7); description at p. 5–6 of the
PDF extract.

### Exact formulas

Complex unit-magnitude projection (eq. 3):

```
π(x) = F^{-1}( ..., F(x)_j / |F(x)_j|, ... )
```

The FHRR substrate is *already* in the projected regime by construction —
phase-only vectors are π-projected — so this is not new infrastructure for
the project; it is the precondition that makes the Jp/Jn formulas meaningful.

Their loss has two halves applied to a statement `s = Σ_{cp ∈ Yp} p ⊗ cp +
Σ_{cm ∈ Ym} m ⊗ cm`, query response `ŝ`, with `p` = present-marker, `m` =
absent-marker, `c_i` = class concept (in our application: role vectors and
filler atoms):

```
Jp = Σ_{cp ∈ Yp} ( 1 - cos( p* ⊗ ŝ, cp ) )                  (eq. 6)

Jn = cos( m* ⊗ ŝ,  Σ_{cp ∈ Yp} cp )                          (eq. 7)
```

`a*` here is Plate's *pseudo-inverse* (FHRR conjugate); under π-projection
`a* = a†` exactly.

The Ganesan claim, demonstrated in their Fig. 1 (p. 4): with π in place,
the present-response is ≈ 1 and the absent-response is ≈ 0 *for thousands
of bound terms in d=256*. The variance does not compound with binding count
the way naive HRR does. The response is on a calibrated bounded scale by
construction.

### Adapted to the project (per-atom role-fidelity)

Re-mapping Ganesan's vocabulary to ours: "class concept" → role vector
`r_k`; "statement" → schema binding `s_i = Σ_k r_k ⊗ a_{i,k}`; "present
markers" → there is no present-marker structure, but the same idea works
without one because we are measuring per-atom *role-recovery*, not
multi-label membership.

For each schema `s_i` with W bound (role, filler) pairs `{(r_k, a_{i,k})}`,
define for each role position k:

```
P_{i,k}^+ = cos( r_k* ⊗ s_i,  a_{i,k} )                      [Jp-style]

P_{i,k}^- = mean_{j ≠ k} cos( r_k* ⊗ s_i, a_{i,j} )          [Jn-style]
```

`P^+` should be ≈ 1 if role-k is cleanly recoverable from `s_i`; `P^-` is
the expected cross-talk leakage into other co-bound fillers. The per-atom
role-fidelity is the *separation gap*:

```
f_i^{Ganesan} = mean_k ( P_{i,k}^+ - max_{j ≠ k} cos( r_k* ⊗ s_i, a_{i,j} ) )
```

This is bounded in [-2, 2], structurally; the project's cosine-on-FHRR
already lives in this scale.

A simpler symmetric variant — the *role-recovery margin*:

```
f_i^{margin} = mean_k ( cos(r_k* ⊗ s_i, a_{i,k}) - mean_{j ≠ k} cos(r_k* ⊗ s_i, a_{i,j}) )
```

### Why it has variance where pairwise distance doesn't

The current f_i measures `1 - |⟨f_j, f_k⟩|` between unbinds of *position
vectors*. Two unbinds being far apart is necessary-but-not-sufficient for
role recovery and at D=4096 every two unbinds *are* far apart because cosine
between independent FHRR vectors concentrates near zero with variance ~ 1/D.
That signal is dimensionally suppressed.

Ganesan's Jp/Jn instead asks: *can the right filler be picked out by
unbinding the right role?* The relevant quantity is the difference between
the correct-filler cosine and the wrong-filler cosines — i.e., a
*discriminability gap*, which is bounded above by 1 and has its variance
governed by the binding count W, not the substrate dimension D. Ganesan
shows empirically (Fig. 1) that this gap does *not* close as binding count
grows past 1000 in d=256, whereas pairwise variance does compound — so the
gap shrinks more slowly than the distance noise grows.

For our atoms: schemas with cleanly-bound (role, filler) pairs will have a
large positive gap; schemas where the binding is corrupted (e.g., one filler
duplicated across roles, or one role drifting into another) will have a
near-zero or negative gap. This is *per-atom* variance, not a substrate-wide
constant.

### Anti-homunculus check

`P^+` and `P^-` are local geometric quantities computed from the FHRR
substrate alone: unbind, take cosine. No supervisor; no threshold; no
"if X then Y". The gap *is* the measurement. Pass.

### Implementation cost

20–30 lines in `src/energy_memory/phase5/role_fidelity.py` next to
`compute_role_fidelity`. Reuses substrate `unbind` and `cosine`. No new
infrastructure. Identical signature shape (returns `[N]` float tensor).

---

## 2. Vangara–Gopinath cap-coverage-of-unbind-cluster

### Source

Vangara & Gopinath 2026 (NeurIPS submission, project trilogy paper),
*The Geometry of Consolidation*. Theorem 1, eqs. on p. 1 and 4.

### Exact formula

For a cluster C of unit-norm items on S^{d-1} with mean within-cluster
cosine distance `d̄` and effective dimension `d_eff` (participation ratio of
covariance spectrum), and retrieval cap `θ' = 1 - θ`:

```
ε_id ≥ 1 - c₁ · m · (θ' / d̄)^{d_eff / 2}                (Theorem 1)
```

with universal constant `c₁ ≈ 0.05` at the 95% quantile in the tight regime
(d̄ < θ').

The two parts of the construction are:
- `d̄ = mean_{i,j ∈ C, i≠j} (1 - ⟨x_i, x_j⟩)` — mean within-cluster cosine distance
- `d_eff = (Σ_k λ_k)² / Σ_k λ_k²` — participation ratio of the cluster
  covariance spectrum `{λ_k}`

The headline regime indicator is `d̄ / θ'` (smaller is tighter; <1 is the
"safe" regime where every consolidator works).

### Adapted to per-atom role-fidelity

For each schema `s_i` and each role position k, construct the *unbind
cluster* `U_{i,k} = { unbind(s_j, r_k) : s_j is any schema containing
role k }` — i.e., the set of recovered fillers for role k across the schema
store. The role is "intact at atom i" iff the unbind `unbind(s_i, r_k)` lies
within the cap of the cluster `U_{i,k}`'s centroid.

Per-atom role-fidelity then becomes:

```
f_i^{cap} = mean_k indicator[ cos( unbind(s_i, r_k), centroid(U_{i,k}) ) ≥ θ ]
```

For a *soft* (continuous) version that drops the indicator and uses the
bound itself:

```
f_i^{spectral} = mean_k ( θ' / d̄_{i,k} )^{d_eff_{i,k} / 2}
```

where `d̄_{i,k}` and `d_eff_{i,k}` are computed on the local unbind cluster
around atom i's unbind for role k (e.g., its K-nearest unbind neighbors in
that role's cluster).

### Why it has variance where pairwise distance doesn't

Three reasons:

1. **It's spectral, not pairwise.** `d_eff` is the participation ratio of
   the *covariance spectrum*, not a mean of pairwise dots. The project's
   substrate-level `d_eff = 35.23` proves the spectrum has structure at
   D=4096 even though pairwise distances saturate; the *cluster-local*
   `d_eff_{i,k}` will likewise have structure per-atom even when pairwise
   `1 - |G_jk|` does not.

2. **It's regime-aware.** Different schemas may sit in the tight regime
   (d̄ < θ') vs. the spread regime (d̄ ≥ θ'). The current f_i has no notion
   of regime; it returns the same number whether the cluster is well-behaved
   or pathological. The cap-coverage form *explicitly* differs across
   regimes.

3. **It has a published bound.** The variance behaviour is theoretically
   characterized in the paper. If empirically all atoms come out at the
   same value, that itself is a finding (it means all clusters sit in
   identical regimes), and we can read off *which* regime from `d_eff` and
   `d̄ / θ'`.

### Anti-homunculus check

The bound `ε_id ≥ 1 - c₁ m (θ'/d̄)^{d_eff/2}` is a property of the cluster
geometry; nothing decides anything. Cap-coverage is an empirical outcome
read off from the substrate — see project's own
`emergent-codebook/consolidation-geometry-diagnostic.md` for the
already-audited anti-homunculus shape of the same measurement at the
substrate level. The atom-level version inherits the audit. Pass.

### Implementation cost

40–80 lines. Need:
- Per-(schema, role) unbind cluster construction (concatenate unbinds across
  schemas)
- `d_eff` via SVD of the unbind cluster's centered matrix (one SVD per
  (schema, role) pair, but can be batched)
- `d̄` via vectorized cosine
- The bound's RHS via standard ops

The project already has substrate-level `d_eff` infrastructure
(`phase2/metrics.py` or equivalent — verify). Re-use that. The new work is
defining the *local* unbind cluster per (schema, role), which is one new
indexing routine.

---

## 3. Papyan NC2 simplex-ETF role codebook (structural elimination)

### Source

Papyan, Han & Donoho 2020, *Prevalence of Neural Collapse during the
Terminal Phase of Deep Learning Training*. PNAS 117(40). Definition
(Simplex ETF) p. 24653 (lines 144–159 of extract); NC2 statement at lines
376–380.

### Exact formula

Standard simplex equiangular tight frame with C vertices in R^C:

```
M⋆ = √(C / (C-1)) · ( I - (1/C) 𝟙 𝟙^T )                       (eq. 1)
```

General simplex ETF: `M = α U M⋆ ∈ R^{p×C}`, where `α > 0` is a scale, and
`U ∈ R^{p×C}` (p ≥ C) is a partial-orthogonal matrix (`U^T U = I`).

The defining property — NC2 explicit form — is that for all distinct c, c':

```
⟨μ̃_c, μ̃_{c'}⟩  =  -1 / (C - 1)             (lines 377–380)
```

i.e., every pair of role vectors has the same cosine `-1/(C-1)`. This is the
maximally-separated equiangular configuration on S^{p-1} for C vectors.

### Adapted to role codebook (W=4 case)

For W=4 roles in D=4096 FHRR space (complex unit-magnitude phasors):
- Place role vectors at a simplex-ETF embedding of W=4 points: pairwise role
  cosine is structurally `-1/3 ≈ -0.333`, regardless of D.
- Initialize the role vectors at the ETF and *hold them fixed* during
  schema-store growth (`p, m, c1..L will all be initialized as we have
  described, and will not be altered during training`, Ganesan p. 5 — same
  recipe).

For the FHRR specifically: an ETF in R^p maps to an FHRR phase configuration
by setting the phases to encode the ETF coordinates. Concretely, generate
the ETF as above in real space and then unit-magnitude-FFT-project (which is
already what `random_vector` does in the substrate at line 27 of
`torch_fhrr.py` per project notes), or — simpler — generate W vectors in
real space at the ETF, then encode each as `exp(i · φ)` where φ are random
phases that satisfy the pairwise cosine constraint.

### Role-fidelity in the ETF regime

Under fixed-ETF roles, role-fidelity *as currently defined* becomes a
structural constant: every atom's role-pair cosine is `-1/(W-1) = -1/3`,
exactly. So `f_i` collapses to a fixed number — *worse* than the current
noise-dominated 0.9858 for discriminating atoms.

The point of ETF is *different*: it removes the noise floor at the
substrate level so that *any other metric* becomes meaningful. Specifically:
- Ganesan Jp/Jn (candidate #1) now has its denominator (the cross-talk
  baseline) at the structural minimum, so the discriminability gap is
  *maximal*.
- Cap-coverage (candidate #2) now has `d̄` set by binding noise alone, not
  by role-codebook variance.

ETF is an *enabling* intervention. As a role-fidelity metric per se, it's a
non-starter — but as a substrate change that makes the *other* metrics
informative, it's the cleanest single-shot fix in the catalogue.

### Why it has variance where pairwise distance doesn't

(For ETF-as-role-codebook, not ETF-as-metric.) Under fixed ETF roles:
- All atoms share the same role substrate → the only thing per-atom variance
  can encode is the *binding-specific* (filler, role) geometry, not the
  role-codebook noise.
- The pairwise-unbind cosine variance at fixed roles is set by binding noise
  ~ √(W-1)/√D, which at D=4096, W=4 is ~ 0.027 — small but non-zero, with
  variance entirely driven by per-atom binding quality.

Per-atom f_i would still need *something else* to compute (e.g., Ganesan
gap), but that something else is now uncontaminated.

### Anti-homunculus check

Fixed-codebook initialization is a *structural choice*, not a decision
made during inference. No supervisor; no rule. The codebook is what it is.
Pass.

### Implementation cost

15–40 lines:
- Generate ETF (one-shot at substrate construction): one matrix product
  in `M⋆ = √(C/(C-1)) (I - (1/C) 𝟙 𝟙^T)`.
- Embed into FHRR (one FFT projection).
- Mark roles as `requires_grad=False` (or simply don't include them in any
  online update path).

Risks: the project's emergent-codebook line of work is committed to
*growing* the role codebook. ETF means the role codebook is fixed at init,
which conflicts with the emergent-codebook program at first glance — but
arguably only with the role half of it. Schema-store learning continues
unchanged.

---

## 4. SNR / discriminability-index from Krotov-Hopfield 2016

### Source

Krotov & Hopfield 2016, *Dense Associative Memory for Pattern Recognition*.
Eqs. 4–6 (per project's `literature.md` Group A annotation).

### Exact formula

For a Dense AM with polynomial energy of degree n storing K patterns in N
neurons, the per-pattern noise floor under random patterns is:

```
P_error  ≈  √( (2n-3)!! / 2π ) · ( K / N^{n-1} ) · exp( -N^{n-1} / (2K (2n-3)!!) )
```

The relevant *bounded* quantity is the per-pattern SNR:

```
SNR_i  =  mean_energy_gap_i  /  σ_i
```

where:
- `mean_energy_gap_i` = energy of pattern i at its stored state minus the
  mean over random off-pattern probes
- `σ_i` = std of the energy at random probes (a per-pattern noise estimate)

This maps onto the project's substrate via the soft-Hopfield retrieval
energy: for each atom, compute the retrieval energy at its stored binding
vs. retrieval energy at random shuffled-binding controls; the gap, divided
by the shuffled-binding std, is a per-atom discriminability index.

### Why it has variance where pairwise distance doesn't

This is a *retrieval-dynamics* probe, not a static-geometry probe. Different
schemas will have different retrieval energies even when their static
unbind geometry is uniformly noisy. The SNR explicitly grows as
`√(N^{n-1}/K)` — at high D and moderate W, the SNR is *large* and varies
across patterns based on the consolidation quality of each pattern, not on
the substrate noise floor.

Critically: if the effective interaction order n is 1 (linear softmax at
low β), the SNR is structurally weak and the noise floor IS what we're
hitting; if n ≥ 2 (high β, kernelized retrieve), the noise floor recedes
super-linearly. **This is a substrate diagnostic, not a fix.** It tells us
whether we are stuck because of dimensionality (which f_i^{Ganesan} would
also reveal) or because of effective polynomial order (which only this
diagnostic reveals).

### Anti-homunculus check

Per-atom retrieval energy is a local dynamical quantity; the gap-over-std
is a measurement of that local energy landscape. No arbitration. Pass.

### Implementation cost

50–100 lines plus a control condition (shuffled-binding probes). Needs
careful seeding to compare the same atom against the same shuffled set
across runs. The control infrastructure is already established for the
project's existing experiments.

---

## 5. Settling-based role-fidelity (HEN / Saighi-style)

### Source

- Kashyap et al. 2024, *Modern Hopfield Networks meet Encoded Neural
  Representations* (HEN) — metastable-state rate as separability proxy.
- Saighi & Rozenberg 2025, *Autonomous Retrieval for Continuous Learning*
  — self-inhibition as autonomous probe.

### Exact formula

For each atom `s_i` and each role k:

1. Cue with the bound `(role_k, ?)` query — i.e., the partial query that
   should retrieve atom i's filler at position k.
2. Run the soft-Hopfield retrieve to convergence.
3. Measure two things:
   - `μ_{i,k}` ∈ {0, 1}: did the retrieve land on the correct filler?
     (identity score)
   - `m_{i,k}` ∈ {0, 1}: did the retrieve land on a metastable mixture? (HEN
     metastable-state flag — `1 - max_pattern_cosine ≥ threshold`)

Per-atom role-fidelity is then:

```
f_i^{settle} = mean_k ( μ_{i,k} · (1 - m_{i,k}) )
```

— the fraction of roles that converge cleanly to the right filler.

Saighi extension: after step 2, *inhibit* the just-retrieved filler in the
codebook (add `-A_i` to its activation; per Saighi eqs. on the adaptation
term), then re-cue with the same `(role_k, ?)`. A high-fidelity binding
will land on a *different* filler on the second pass; a low-fidelity
binding will collapse or repeat.

### Why it has variance where pairwise distance doesn't

The settling endpoint is sensitive to the energy landscape around the cue,
not to the substrate's mean pairwise geometry. Two atoms with identical
pairwise unbind cosines can have completely different settling endpoints if
their bindings sit in differently-shaped basins. Project notes (May-9 PROJ
synthesis) already commit to metastable-state-rate as a Phase 3 diagnostic;
extending it to per-atom is a small step.

The HEN-style observation is that pattern *separability* (controllable via
encoder) governs metastable-state rate. In our setting, the encoder is the
role codebook; per-atom variance in metastable-state rate is variance in
how well each atom's binding survived consolidation, which is exactly what
the project wants to read off.

### Anti-homunculus check

Both `μ` and `m` are read off from the natural retrieval-dynamics
trajectory. No supervisor decides; the energy landscape settles where it
settles. Pass.

### Implementation cost

80–150 lines. Needs:
- A per-(atom, role) partial-cue construction
- Soft-Hopfield retrieve loop (already exists in `torch_hopfield.py`)
- Metastable-flag computation (already in `phase2/metrics.py` per
  project notes)
- Aggregation per atom across roles

The Saighi-extension adds another ~30 lines for the self-inhibition pass.

The largest cost is *compute*: a settling-based metric is W × N retrievals,
not a single O(N·W²·D) vectorized op. On the W=4, N=1064 scale this is
likely tractable (~4k retrievals, each a few-iter loop).

---

## Honorable mentions (not promoted to full candidates)

- **Spike-Train Hyperdimensional Computing fidelity (Schlegel et al.,
  Kanerva).** Measures binding fidelity via *bit-flip distance in
  sparse-binary VSA*, which is structurally bounded in [0, 1]. Not adopted:
  the project is committed to FHRR phasors, and translating to SBC for a
  metric only would introduce a translation overhead that exceeds the
  candidate metrics above.

- **Smolensky Tensor Product Representation fidelity.** TPR has *no*
  binding noise (it's exact at the cost of O(d^n) storage), so the
  "fidelity" question doesn't arise; not portable to FHRR's compressed
  binding.

- **Plate's "expected response" closed-form.** Plate 1995 gives a closed
  form for the expected response of a clean HRR unbind as a function of d
  and binding count. Useful as a *baseline* (the predicted noise floor) to
  compare against the observed Jp gap, but not itself a per-atom metric.

- **Sparse Distributed Memory address-decoder activation count.** SDM's
  fidelity probe is "how many locations does this address activate?"
  Per-atom variance is natural. Not adopted because the project's
  substrate is not address-based; would require an address-decoding overlay.

- **MESH heteroassociation injectivity test.** MESH's scaffold-pattern
  heteroassociation has a structural injectivity guarantee — for each
  scaffold state there is at most one pattern. Tests of per-atom
  injectivity would be a fidelity probe but require the scaffold layer
  the project doesn't have yet.

---

## Ranking: most likely to discriminate atoms on the current substrate

(Ranked by expected per-atom signal-to-noise on the existing D=4096, W=4,
N=1064 substrate, with no architectural change other than the metric
itself.)

1. **Ganesan Jp/Jn role-recovery margin (#1).** Single-shot drop-in,
   bounded in [-2, 2] by construction, variance driven by binding quality
   not substrate dimension, ~25 lines. Highest signal/cost ratio. Direct
   fit to the noise-floor pathology because it replaces pairwise distance
   with a discriminability *gap*.

2. **Krotov-Hopfield SNR (#4).** Diagnostic-grade per-atom score from
   retrieval energy gap over shuffled-binding std. Reveals whether the
   problem is dimensional or interaction-order. Slightly more work
   (~75 lines plus controls) but provides architectural insight the
   Ganesan metric does not.

3. **Settling-based metastable-state rate (#5).** Per-atom variance is
   guaranteed because settling endpoints are basin-dependent and basins
   differ across atoms even when geometry looks uniform. Highest compute
   cost, but uses infrastructure the project has already built; aligns
   with anti-homunculus + dynamic-form commitments most cleanly of all
   candidates.

4. **Vangara–Gopinath cap-coverage of unbind cluster (#2).** Spectrally
   grounded, regime-aware, has a published bound — but the construction
   requires defining per-(schema, role) unbind clusters which is more
   plumbing. Most theoretically motivated; most implementation overhead;
   expected per-atom variance high but not guaranteed if all unbind
   clusters happen to share the same regime.

5. **Simplex-ETF role codebook (#3).** Not itself a per-atom metric; an
   *enabling intervention* that maximizes the headroom of the other
   metrics. Recommended as a follow-up if #1 / #2 reveal that role-codebook
   noise (not binding noise) is what's dominating per-atom variance. Mild
   conflict with the emergent-codebook program; defer until needed.

The recommended path is **#1 first, immediately** — it is the smallest
change that directly attacks the documented failure mode and is cheap
enough that running it on the existing substrate within the current session
is plausible. **#2 and #5 are the natural follow-ups** depending on what #1
reveals. **#3 is held in reserve** as the structural lever if all
per-atom metrics still show uniform values, which would mean the bottleneck
is the role codebook itself rather than the metric.
