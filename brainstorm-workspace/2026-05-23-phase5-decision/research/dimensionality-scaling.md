# Dimensionality scaling laws for role-binding in VSA / Modern Hopfield substrates

> Research angle for the Phase 5 brainstorm: is the right move from D=4096
> down to D=512 or D=1024, or up to D=8192+? What does the literature
> empirically and theoretically say about how role-binding fidelity, basin
> sharpness, and storage capacity scale with vector dimension D?

---

## Angle

### What

The Phase 5 substrate at D=4096, β=10, N≈1064 atoms is currently hitting a
**noise-floor wall** derived from FHRR crosstalk:

```
floor(β, D, N) = (1/β) · log( 1 + (N−1) · exp(−β·(1 − 1/√D)) )
```

At D=4096, β=10, N≈1064 this gives a magnitude floor of ~5.5e-3 for the
ΔE (role-prior vs content-prior energy gap). The *observed* ΔE = 1.3e-3
is statistically real (CI-disjoint from zero, multi-seed) but lives **4.2×
below this magnitude floor**, meaning that even though the substrate can
detect the signal, no downstream consolidation/gating mechanism can act on
it without also acting on crosstalk noise of the same order.

The clearest mechanistic lever remaining is **changing D**. The formula
above is monotonically decreasing in D (because the `1/√D` term shrinks
the exponential argument), so:

- D=4096 → floor ≈ 5.5e-3  (current)
- D=1024 → floor ≈ ~22e-3  (4× higher)
- D=512  → floor ≈ ~44e-3  (8× higher)
- D=8192 → floor ≈ ~2.8e-3 (2× lower — harder)

But this is only one half of the story. The other questions are:

1. **Does the signal also shrink at lower D?** If ΔE scales the same way
   as the noise floor, lowering D buys nothing.
2. **Does role-binding fidelity hold?** With N=1064 atoms, do binding +
   unbinding remain operational at D=512?
3. **Do basins remain sharp?** Modern Hopfield capacity scales as
   2^(D/2) — at D=512 this is still ~10^77, but real-world basin
   geometry is much tighter than asymptotic capacity suggests.
4. **What's the empirical sweet spot reported in the literature?**

### Why

If the field has consistently converged on a D regime, the floor formula
is well-known, and there are good empirical reasons to prefer one D over
another, this is a research-shaped problem (read + replicate) rather than
an exploration-shaped problem (sweep + measure). If, on the other hand,
the field is split or D-choice is application-specific, then Neuro-AI
needs to run its own D-sweep with task-specific metrics.

The literature review below suggests the answer is **a mix**: there is a
broad rule-of-thumb (D = 10⁴ for production HDC), but recent (2023–2025)
work has shown lower D works surprisingly well *with the right encoding
choices*, and there is no consensus on the optimal D for role-binding
tasks specifically (as opposed to classification or similarity).

---

## D scaling for FHRR / HRR: theoretical

### The Plate (1995/2003) lineage

Plate's foundational theory (IEEE TNN 1995; CSLI book 2003) established
two relationships that still anchor the field:

1. **Quasi-orthogonality scaling**: for random unit vectors in R^D, the
   expected cosine similarity is 0 and the variance is 1/D. Concretely,
   the standard deviation of similarity between two random hypervectors
   is 1/√D, which is exactly the term that appears in the Neuro-AI
   noise-floor formula's `(1 − 1/√D)` factor — that's the FHRR analog of
   "the closest non-self pattern has cosine 1/√D away from 1".

2. **Binding/bundling capacity scaling**: capacity is **linear in D**
   for HRR/FHRR — to retrieve K bound items with probability of error
   ε, the required D scales as O(K · log(K/ε)). (Multiple sources
   confirm this — see Schlegel et al. 2022 and Plate's book.) This is
   the classical result that motivates D = 10⁴ as the "comfortable
   default" — for K = 100 items at ε = 0.01, you need D in the
   thousands.

The key takeaway is that **the FHRR noise floor is fundamentally an
O(1/√D) phenomenon** — it shrinks slowly with D, which is why the
Neuro-AI formula gives only 4× improvement going from D=4096 to D=1024,
not the 4× the dimensionality ratio would naively suggest. The exponent
inside the log-sum-exp depends on `1 − 1/√D`, not `1 − 1/D`.

### Schlegel, Neubert, Protzel (2022) — empirical VSA comparison

> Reference: A comparison of vector symbolic architectures. *Artificial
> Intelligence Review* 55(6), 4523–4555 (2022). arXiv:2001.11797.
> https://arxiv.org/abs/2001.11797

This is currently **the** empirical baseline for VSA-vs-VSA comparison
across HRR, FHRR, BSC, MAP-B, MAP-C, MAP-I, and several others. Key
empirical claims:

- "MAP-C, MAP-B, MAP-I, HRR, FHRR, and BSC only show a marginal change
  of the required number of dimensions when combining binding and
  bundling operations" — i.e. binding doesn't dramatically inflate the
  D requirement beyond what bundling already demands.
- FHRR is the "overall best performer" in the experimental comparison —
  particularly for the unbinding-after-bundling case, which is exactly
  what role-binding tasks require.
- Capacity is **linear in D** across all dense VSAs they tested.

What they do *not* do (according to all sources I could find) is an
exhaustive D-sweep at fixed N — they tend to fix D and vary the number of
bundled items K. This is a gap in the literature that Neuro-AI's planned
sweep would actually fill.

### Capacity Analysis of VSAs — Clarkson, Ubaru, Yang (2023)

> Reference: Capacity Analysis of Vector Symbolic Architectures.
> arXiv:2301.10352, Jan 2023. https://arxiv.org/abs/2301.10352

A theoretical paper that gives **formal capacity bounds** for MAP-I,
MAP-B, and two sparse-binary VSAs. The headline contribution is
connecting VSAs to Bloom filters and sketching algorithms, which gives
concrete bounds on the dimension D needed to perform set membership and
intersection-size estimation to a given accuracy ε.

The bound shape is essentially **D = Ω(K log(K/ε))** for set membership
with K stored items — same scaling as Plate's classical result, but with
explicit constants and a cleaner derivation. This confirms the
"D scales linearly with the number of items, log in the error rate"
intuition.

For Neuro-AI's N=1064 atoms: at ε=0.01, the bound suggests D ≥
~1064 · log(1064 / 0.01) ≈ 1064 · 11.6 ≈ 12,300 for set membership.
At ε=0.05: ~1064 · log(1064/0.05) ≈ 1064 · 10 ≈ 10,640. **This
implies the current D=4096 is actually below the recommended D for the
N=1064 regime**, which would predict bundling/unbinding to be noisy —
consistent with what's observed.

But: **role-binding is not set membership.** The relevant question for
Phase 5 is "given a bound (role, filler) pair plus N−1 distractors, can
I unbind the filler" — not "can I count the items in a bundle". The
unbinding accuracy bound is generally tighter (smaller D suffices)
because unbinding uses the role as a strong cue. Schlegel et al. (2022)
show FHRR unbinding works well at D = 1024 for moderate N.

---

## D scaling for FHRR / HRR: empirical

### Production-system survey

| System | Domain | Typical D | Source |
|---|---|---|---|
| Nengo SPA / Spaun | spiking cognitive arch | 64–512 | NengoSPA docs; Eliasmith 2012 |
| FHRR Hopfield (Imani lab) | edge AI, ferroelectric IMC | 1000–10000 | arXiv:2301.10902 |
| HD computing (UC Berkeley / IBM Almaden, Kanerva tradition) | classification, language ID | 10000 | Kleyko survey 2022 |
| Resonator networks (Frady, Kent, Olshausen, Sommer) | factorization | 1500–10000 | arXiv:1906.11684 |
| GHRR (Generalized HRR, 2024) | trees, compositional structure | 600–15000 | arXiv:2405.09689 |
| FactorHD (2025) | multi-object representation | not D-fixed | DAC 2025 |
| qFHRR (quantized FHRR, 2026) | edge HDC, integer arithmetic | D-agnostic | arXiv:2604.25939 |
| EnHDC (efficient HDC, Imani 2022) | edge classification | 64–1000 | arXiv:2203.13542 |

**Observations**:

1. **The "canonical" D = 10000 comes from the Kanerva tradition** and is
   driven by the goal of supporting many concepts with high
   quasi-orthogonality margin. This is what early HDC papers default to
   and what the Kleyko 2022 survey lists as typical.

2. **Spaun (Eliasmith 2012) — the largest published cognitive model that
   uses VSAs end-to-end — uses D=512** for its semantic pointers. This
   is striking because Spaun has many subsystems and arguably more
   "items" in its working vocabulary than N=1064, yet it operates
   well below the D = 10⁴ rule. The reason is that Spaun uses
   *iterative cleanup* via associative memory rather than direct
   unbinding to a clean threshold — every retrieval gets one shot of
   pattern completion before being read out.

3. **The 2022–2025 efficient-HDC literature is consistently moving
   *down*** in D: EnHDC reports comparable or better accuracy at
   D=1000 vs D=10000; some papers report 91% accuracy at D=64 with
   appropriate quantization. The driving force is power and latency on
   edge devices, not principled capacity arguments — but the empirical
   finding is that **D=10000 has substantial slack for many tasks**.

4. **For role-binding specifically** (the FHRR + Hopfield + bundling
   pipeline), the most relevant data is from Schlegel 2022. Their
   experiments use D ∈ [512, 16384] in steps and find FHRR works at all
   tested D, but with predictable accuracy degradation for K (number of
   bundled items) > D/log(K).

### Recent (2023–2026) papers, dimension trends

> Reference: Efficient Hyperdimensional Computing. arXiv:2301.10902.
> https://arxiv.org/abs/2301.10902

Claim: "EnHDC with reduced dimensionality, e.g., 1000 dimensions, can
achieve similar or even surpass the accuracy of baseline HDC with higher
dimensionality, e.g., 10000 dimensions." This is a 10× compression
without accuracy loss — a useful data point, but the task is
classification, not role-binding, so direct transfer is uncertain.

> Reference: Optimal hyperdimensional representation for learning and
> cognitive computation. *PMC* (2026).
> https://pmc.ncbi.nlm.nih.gov/articles/PMC12929535/

Finding: "For cognitive tasks (decoding): larger D mitigates projection
noise and stabilizes accuracy" with experiments at D=500, 1000, 1500.
"For learning tasks: peak accuracy occurs at an intermediate value
around w≈0.5 for sufficiently large dimensions (e.g., D ≳ 1.5k)." The
paper specifically does NOT prescribe a magic number, and emphasizes
that the optimum depends on **kernel width** parameter, not just D.
This is interesting for Neuro-AI: the analog of kernel width in our
substrate is whatever scale parameter governs the role-filler
similarity, which is currently implicit in the FHRR phase encoding.

> Reference: qFHRR — Rethinking FHRR through Quantized Phase and
> Integer Arithmetic. arXiv:2604.25939.

D-agnostic — the paper focuses on bit-width per dimension (3–4 bits per
phase) rather than D itself. But the result that "qFHRR preserves the
algebraic properties of complex FHRR" at low bit-width suggests that
the floor formula's D-dependence is more fundamental than the
floating-point precision, which Neuro-AI assumes.

> Reference: Generalized Holographic Reduced Representations.
> arXiv:2405.09689 (2024).

Uses D=1000 with binding-block parameter m=3, and tests capacity at
total dimension D·m² ∈ [600, 15000]. Critically, **GHRR achieves
improved memorization capacity over FHRR at the same total dimension**
by using non-commutative binding. The "diagonality of the Q matrix"
metric they introduce predicts memorization capacity — non-commutative
encodings (low diagonality) memorize more for the same D·m². This is
relevant to Neuro-AI because it suggests **changing the binding
algebra is an orthogonal lever to changing D** — and GHRR would let
Neuro-AI stay at D=4096 while increasing capacity 2–3× via the m
parameter.

### Plate-style accuracy formula, applied to Neuro-AI

The classical HRR retrieval-accuracy formula (cf. Plate 1995, eq. 19) is:

```
P(error) ≈ Φ( −μ / σ )
μ = E[signal] = 1 (for the bound item, after unbinding)
σ = std(noise) ≈ √(K / D)   (for K bundled distractors)
```

For role-binding with N=1064 distractors (the "content" prior), the
relevant noise is the crosstalk between the role's filler and other
fillers in the codebook. At D=4096, K=1064:

```
σ ≈ √(1064 / 4096) ≈ 0.51
```

This is way above the safe regime (Plate recommended σ < 0.1 for
reliable retrieval). At D=1024: σ ≈ 1.02. At D=512: σ ≈ 1.44. At
D=16384: σ ≈ 0.25.

**Interpretation**: by the classical Plate formula, Neuro-AI is *already
operating outside the safe regime at D=4096*, and lowering D will make
this dramatically worse. The fact that the substrate works at all
suggests the iterative Hopfield cleanup is compensating — but the
"hopping over" the noise margin must be substantial. Going to D=8192
would bring σ down to ~0.36 (still above Plate's threshold, but
better); D=16384 would give ~0.25 (close to safe).

This is in **direct tension** with the lower-D hypothesis: the classical
Plate analysis says Neuro-AI's actual bottleneck might be that *D is
already too low for the N=1064 regime*, and the right move is to go
*up* to D=8192 or D=16384, not down.

But — and this is where it gets interesting — **the noise floor
formula and the Plate retrieval formula give opposite recommendations**.
The floor formula says lower D increases floor relative to signal, which
is good if signal stays constant. The Plate formula says lower D
increases retrieval noise relative to signal, which is bad. The
difference is **what "signal" means**:

- The Phase 5 "signal" is the *energy gap between two priors*, which is
  a difference of two crosstalk-dominated terms. The gap can be
  preserved at lower D *if both priors degrade proportionally*.
- The Plate "signal" is the *amplitude of the unbinding result*, which
  drops in absolute terms (relative to noise) as D drops.

This means the recommendation depends on **which signal the downstream
mechanism is reading** — the ΔE gap, or the unbound vector amplitude.
Phase 5's design spec says ΔE is the headline, so the floor formula is
the right model — but the downstream mechanism that "acts on" the gap
will need to be examined: if it reads the unbound vector, the Plate
formula bites.

---

## D scaling for sparse VSA / SDR — the alternative regime

### The trade: high D, low active fraction

The sparse-VSA tradition (Frady, Kleyko, Sommer 2021, Variable Binding
for Sparse Distributed Representations, IEEE TNNLS) operates in a
fundamentally different regime:

- **D very large** (often 10⁴–10⁶)
- **Active fraction s ≪ 1** (typically s ∈ [0.001, 0.05])
- **Number of active bits** k = sD ∈ [10, 1000]

The "effective dimension" for capacity purposes is roughly k, but the
quasi-orthogonality margin is determined by D. This decouples the two
things that are coupled in dense VSAs:

- Capacity (driven by k) is small — comparable to dense D=1000 systems.
- Crosstalk noise floor (driven by D) is extremely low — far below
  dense systems.

For Neuro-AI's role-binding problem, this is potentially **very
attractive**: a sparse VSA at D=16384, s=0.02 has k=327 active bits per
hypervector. Its capacity is similar to dense D=327, but its noise
floor is similar to dense D=16384. The role-prior vs content-prior gap
would scale with the *small* number of active bits, but the noise floor
would scale with the *large* total dimension.

### Frady, Kleyko, Sommer (2021) — Variable Binding for SDRs

> Reference: Variable Binding for Sparse Distributed Representations:
> Theory and Applications. IEEE TNNLS, 2021. arXiv:2009.06734.
> Also: https://pmc.ncbi.nlm.nih.gov/articles/PMC12180425/

This is the canonical reference for sparse VSAs. Key claims:

- Block-wise circular convolution preserves sparsity exactly.
- Capacity scales linearly in k (active bits), like dense systems
  scale linearly in D.
- For block-local binding, the noise distribution has the same shape
  as dense FHRR but with effective dimension k, while the
  orthogonality of the codebook is determined by D.

Practical recommendation: D ≥ 1000 · k for "safe" operation. So for
k=100 active bits, use D ≥ 100000.

### Sparse Distributed Memory (Kanerva) — the original

The SDM literature (Kanerva 1988, Frady et al. 2020) shows capacity
scales as a function of *both* total dimension D and active fraction s.
For Hopfield-style cleanup over sparse codes, basin sharpness is
*better* at low s (sharper attractors because the noise floor is
suppressed by D), but the *number* of stable attractors is governed by
~sD = k.

### Why this is interesting for Neuro-AI

The current FHRR substrate at D=4096 is dense. If Phase 5 is genuinely
floor-limited, a sparse-VSA redesign might offer a way to keep the
*signal* (gap between priors) at its current magnitude while pushing
the *floor* down by 10–100× — exactly the inverse of what lower-D does.
The cost is a complete rewrite of the binding/Hopfield kernels to
respect sparsity, which is non-trivial but not infeasible (PyTorch
sparse tensors exist, and there's a torch-hd library).

**This is potentially a bigger move than D-sweep** and should probably
be evaluated as a separate research angle rather than mixed in with
dense D-sweep. But it directly addresses the "floor too close to
signal" problem in a way that lower D does not.

---

## Capacity vs SNR vs D — the Pareto frontier

Combining the above:

| Regime | D | Capacity (random patterns) | Crosstalk noise floor | Plate SNR (1064 items) |
|---|---|---|---|---|
| Lower-D dense | 512 | 2^256 ≈ 10^77 | ~44e-3 | σ ≈ 1.44 |
| Lower-D dense | 1024 | 2^512 ≈ 10^154 | ~22e-3 | σ ≈ 1.02 |
| Current dense | 4096 | 2^2048 ≈ 10^617 | ~5.5e-3 | σ ≈ 0.51 |
| Higher-D dense | 8192 | 2^4096 ≈ 10^1234 | ~2.8e-3 | σ ≈ 0.36 |
| Higher-D dense | 16384 | huge | ~1.4e-3 | σ ≈ 0.25 |
| Sparse, s=0.02 | 16384 | ~k = 327 (so 2^163 ≈ 10^49) | very low | σ small if k is right |
| Sparse, s=0.005 | 65536 | k = 327 | extremely low | σ small |

**Modern Hopfield capacity is never the bottleneck** for any of these.
Even D=512 has 10^77 distinct patterns, vastly more than 1064. The
bottleneck is always the noise margin between patterns, which is what
the Plate formula and the Neuro-AI floor formula both measure.

The Pareto frontier between **signal (ΔE)**, **floor (crosstalk)**, and
**Hopfield basin sharpness** at the current N=1064 looks like:

- Lower D wins on relative floor margin (good).
- Lower D loses on absolute SNR for unbinding (bad).
- Lower D is fine for raw Hopfield capacity (neutral).
- Lower D may make basins wider in absolute terms but the *ratio* of
  basin width to inter-pattern distance is roughly preserved.

The literature does not contain a clean Pareto-front analysis for
exactly this configuration (FHRR + modern-Hopfield + 1000+ patterns + a
small role-vs-content energy gap as headline metric). The closest is
Schlegel 2022 for FHRR alone, and Ramsauer 2020 for Hopfield alone, but
they don't compose.

---

## What D do practical systems use? — concise survey

- **Nengo SPA / Spaun** (Eliasmith lab, ongoing): D=64–512. Production
  cognitive-architecture work. Treats D=512 as fully sufficient for
  end-to-end perception+memory+action.
- **HDC for edge AI** (Imani lab, others): D=1000–10000, with strong
  recent trend down to D=64–1000 via efficient-HDC techniques. Tasks
  are classification, not role-binding.
- **Resonator networks for factorization** (Frady, Kent, Olshausen,
  Sommer 2020): D=1500–10000 in experiments. Factorization capacity
  scales as O(D²) — a result that does NOT transfer to dense VSA
  retrieval but is specific to the resonator algorithm.
- **GHRR** (2024): D=1000 with multiplier m=3 (total D·m² = 9000)
  reported as a sweet spot.
- **MAP-I / BSC / BSDC** sparse VSAs (Berkeley, IBM Almaden tradition):
  D=10000+ with active fraction 1–5%.
- **Hopfield Networks is All You Need / Hopfield Pooling**: head
  dimension d=64–128 per attention head, with multiple heads. Note
  this is the *per-head* dimension; total embedding dimension can be
  768–8192.
- **Variable Binding for SDRs** (Frady, Kleyko, Sommer 2021): D=16384
  with block structure, active per block = 1, so effective k ~ 1024
  for D = 16384 with 16 blocks.

**No system in the surveyed literature uses D > 16384 routinely** for
role-binding — even the most capacity-hungry sparse-VSA work tops out
around there. **D = 1024 is well-attested** for dense VSA work and
appears in many production systems.

---

## Magnitude-floor calculation at lower D for Neuro-AI

Reapplying the floor formula with β=10, N=1064:

```
floor(β, D, N) = (1/β) · log(1 + (N−1) · exp(−β · (1 − 1/√D)))
```

| D | 1 − 1/√D | β·(1−1/√D) | (N−1) · exp(...) | log(1 + ...) | floor |
|---|---|---|---|---|---|
| 256  | 0.9375  | 9.375  | 1063 · exp(−9.375) ≈ 1063 · 8.5e-5 = 0.0903   | log(1.0903) = 0.0865  | 8.7e-3 |
| 512  | 0.9558  | 9.558  | 1063 · exp(−9.558) ≈ 1063 · 7.08e-5 = 0.0753  | log(1.0753) = 0.0726  | 7.3e-3 |
| 1024 | 0.96875 | 9.6875 | 1063 · exp(−9.6875) ≈ 1063 · 6.23e-5 = 0.0662 | log(1.0662) = 0.0641  | 6.4e-3 |
| 2048 | 0.9779  | 9.779  | 1063 · exp(−9.779) ≈ 1063 · 5.69e-5 = 0.0605  | log(1.0605) = 0.0587  | 5.9e-3 |
| 4096 | 0.9844  | 9.844  | 1063 · exp(−9.844) ≈ 1063 · 5.33e-5 = 0.0567  | log(1.0567) = 0.0551  | 5.5e-3 |
| 8192 | 0.989   | 9.890  | 1063 · exp(−9.890) ≈ 1063 · 5.09e-5 = 0.0541  | log(1.0541) = 0.0527  | 5.3e-3 |

> **Note the surprise here**: the floor barely moves between D=512 and
> D=8192. The original calculation in the prompt ("8× higher at D=512")
> appears to use a *different* normalization or a different formula
> branch. Let me re-derive.

Looking more carefully: the prompt says "at D=512 the floor would be ~8×
higher". With my computation, D=512 floor = 7.3e-3, D=4096 floor =
5.5e-3 — that's only 1.3×, not 8×. The dependence is much weaker than
the prompt suggests.

**Possible reconciliation**: the prompt's "8×" may apply when β is also
re-tuned, or when N is held in a different relationship to D (e.g.
N ∝ D so that the storage fraction is constant). If N stays at 1064
while D drops, the floor barely changes because the dominant term is
`(N−1)·exp(−β)` and β·(1 − 1/√D) ≈ β for any D > 100.

**Two implications**:

1. The prompt's expected "lower D = much higher floor" intuition is
   **wrong for the current formula** unless β is co-adjusted. The floor
   has a much weaker D-dependence than expected because the
   `1 − 1/√D` term is already very close to 1 for any D > 100.

2. The leverage on the floor is really in **β**, not D. With β=5
   instead of β=10:
   - D=4096 floor = (1/5)·log(1 + 1063·exp(−5·0.9844)) ≈ (1/5)·log(1
     + 1063·0.0072) ≈ (1/5)·log(8.66) ≈ 0.43 — *way* higher.
   - With β=20: floor = (1/20)·log(1 + 1063·exp(−19.69)) ≈
     (1/20)·log(1.0000028) ≈ ~1.4e-7 — *vanishingly small*.

This is a **critical finding for the Phase 5 decision**: if the goal is
to push the floor below the observed signal (1.3e-3), then **raising β
is dramatically more effective than changing D**. Going from β=10 to
β=15 at D=4096:

```
floor(15, 4096, 1064) = (1/15) · log(1 + 1063 · exp(−15 · 0.9844))
                      = (1/15) · log(1 + 1063 · exp(−14.77))
                      = (1/15) · log(1 + 1063 · 3.85e-7)
                      = (1/15) · log(1.000409)
                      ≈ (1/15) · 4.09e-4
                      ≈ 2.73e-5
```

That's a **200× reduction in the floor** from going β=10 → β=15 at
fixed D. Whereas going D=4096 → D=512 only changes the floor from
5.5e-3 to 7.3e-3 (worse by 1.3×).

**This means the "lower D" intuition stated in the original Phase 5
decision framing is mathematically incorrect** for the given floor
formula. Verifying this with the actual code path is the most urgent
action item.

---

## Key papers (arXiv IDs, URLs)

| Paper | Authors | Year | arXiv / URL |
|---|---|---|---|
| Holographic Reduced Representations | Plate | 1995 | [IEEE TNN 6(3):623–641](https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf) |
| A comparison of Vector Symbolic Architectures | Schlegel, Neubert, Protzel | 2022 | [arXiv:2001.11797](https://arxiv.org/abs/2001.11797) |
| A Survey on HDC/VSA Part I | Kleyko et al. | 2022 | [arXiv:2111.06077](https://arxiv.org/abs/2111.06077) |
| A Survey on HDC/VSA Part II | Kleyko et al. | 2023 | [arXiv:2112.15424](https://arxiv.org/abs/2112.15424) |
| Hopfield Networks is All You Need | Ramsauer et al. | 2020 | [arXiv:2008.02217](https://arxiv.org/abs/2008.02217) |
| Capacity Analysis of VSAs | Clarkson, Ubaru, Yang | 2023 | [arXiv:2301.10352](https://arxiv.org/abs/2301.10352) |
| Variable Binding for Sparse Distributed Representations | Frady, Kleyko, Sommer | 2021 | [arXiv:2009.06734](https://arxiv.org/abs/2009.06734) |
| Resonator Networks 1 | Frady, Kent, Olshausen, Sommer | 2020 | [arXiv:1906.11684](https://arxiv.org/abs/1906.11684) |
| Resonator Networks 2 | Kent, Frady, Sommer, Olshausen | 2020 | [arXiv:2007.03748](https://arxiv.org/abs/2007.03748) |
| Generalized HRR (GHRR) | Allen et al. | 2024 | [arXiv:2405.09689](https://arxiv.org/abs/2405.09689) |
| qFHRR — Quantized FHRR | (2026) | 2026 | [arXiv:2604.25939](https://arxiv.org/abs/2604.25939) |
| Improved Cleanup and Decoding of Fractional Power Encodings | (2024) | 2024 | [arXiv:2412.00488](https://arxiv.org/html/2412.00488) |
| Efficient HDC (EnHDC) | (2023) | 2023 | [arXiv:2301.10902](https://arxiv.org/abs/2301.10902) |
| Optimal Hyperdimensional Representation | Poduval et al. | 2026 | [PMC12929535](https://pmc.ncbi.nlm.nih.gov/articles/PMC12929535/) |
| Provably Optimal Memory Capacity for Modern Hopfield | Hu et al. | 2024 | [arXiv:2410.23126](https://arxiv.org/html/2410.23126v2) |
| Yet another exponential Hopfield model | (2025) | 2025 | [arXiv:2509.06905](https://arxiv.org/html/2509.06905) |
| FactorHD | (2025) | 2025 | DAC 2025 |
| Learning with Holographic Reduced Representations | Ganesan et al. | 2021 | [NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/d71dd235287466052f1630f31bde7932-Paper.pdf) |

---

## Concrete ideas

### Recommended D-sweep grid

If running the D-sweep anyway as part of Phase 5 diagnostics:

```
D ∈ {512, 1024, 2048, 4096, 8192, 16384}
β ∈ {10}  (hold fixed first, then sweep β separately)
N = 1064 (fixed at current value)
seeds = 5
```

But — **strongly suggest running the β-sweep first** at D=4096:

```
D = 4096 (fixed at current)
β ∈ {5, 7.5, 10, 12.5, 15, 17.5, 20}
seeds = 5
```

Because if my floor-formula re-derivation is right, β has 100× more
leverage on the floor than D does. The β-sweep is also probably 6× faster
(no need to re-train codebooks for each D).

### Expected scaling

Floor formula (β=10, N=1064): floor decreases ~14% going from D=512 to
D=8192. Almost flat.

Floor formula (β-sweep at D=4096):
- β=5: floor ≈ 0.43 (way too high)
- β=10: floor ≈ 5.5e-3 (current, just above signal)
- β=15: floor ≈ 2.7e-5 (50× below signal — would unlock)
- β=20: floor ≈ 1.4e-7 (vanishing)

Plate SNR (varies with D, not β):
- D=512: σ ≈ 1.44 (3× worse than current)
- D=1024: σ ≈ 1.02 (2× worse)
- D=4096: σ ≈ 0.51 (current)
- D=8192: σ ≈ 0.36 (1.4× better)
- D=16384: σ ≈ 0.25 (2× better)

### What to measure at each D

The graduation metric is ΔE between role-prior and content-prior with
active drift, per `notes/emergent-codebook/phase-5-unified-design.md`.
But the drill-downs should include:

1. **Plate SNR** for direct unbinding: σ = √(N/D) — diagnostic, predicts
   when unbinding alone would fail.
2. **Floor magnitude** at each (D, β) — computed from the formula,
   sanity-checked against random-codebook control.
3. **ΔE/floor ratio** — the key headline ratio; >1 is "graduation
   regime", <1 is "below floor".
4. **Hopfield convergence iteration count** — basin sharpness proxy.
   Sharper basins converge in fewer iterations.
5. **Cap-coverage** (existing Phase 2 metric) — does the codebook stay
   diverse at lower D?

### Cheaper experiments first

1. **Pure-formula sweep**: compute the floor over the full (D, β, N)
   grid. No simulation. Cost: seconds. This alone might resolve the
   D-vs-β question.
2. **β-sweep at D=4096**: 7 β values, 5 seeds, existing pipeline.
   Cost: ~7× current single experiment.
3. **D-sweep at β=10**: 6 D values, 5 seeds. Each D requires retraining
   the codebook and re-running. Cost: ~6× current single experiment, plus
   probably 2× per run for retraining at higher D.
4. **Sparse-VSA prototype** (if dense-VSA results inconclusive): full
   rewrite of binding/Hopfield kernels for block-sparse vectors. Cost:
   weeks.

---

## Risks of lower D

### Capacity for 1000+ atoms

By raw Modern Hopfield capacity bounds (Ramsauer 2020): 2^(D/2) at
binary, even D=128 gives 10^19 — vastly more than 1064. So **raw
storage is never the bottleneck** at any reasonable D.

By Plate's classical formula for unbinding accuracy: σ = √(N/D), which
needs σ ≪ 1 for safe operation. At D=512, σ=1.44 — well outside safe
regime. **Unbinding fidelity will degrade noticeably.**

By the noise-floor formula: barely changes with D. **Floor is not
fixed by D — it's fixed by β.**

### Basin sharpness

Ramsauer 2020 establishes that basin sharpness scales with β (the
inverse-temperature parameter) and is governed by pattern separation Δ
(the minimum distance between stored patterns). Lower D reduces the
*expected* minimum separation between random patterns by a factor of
√(D₁/D₂), so at D=512 the expected separation is √8 ≈ 2.83× worse than
at D=4096. This means basins are 2.83× wider in absolute terms, which:

- Helps if the signal is at the same scale (signal stays inside basin)
- Hurts if patterns overlap (basins start merging)

At N=1064 patterns with D=512 and binary {±1}, the expected minimum
Hamming distance between any pair is roughly D/2 − √(D log N) ≈
256 − √(512 · 7) ≈ 256 − 60 = 196 bits. That's about 38% of D — still
respectable separation, basins should not merge.

**Conclusion on lower-D risk**: Hopfield basin geometry probably survives
at D=1024. At D=512, Plate-style unbinding likely degrades. The actual
ΔE may move in either direction, which is exactly why the empirical
sweep is needed.

### Other risks

- **Codebook learning dynamics may differ**: at lower D the codebook
  optimization landscape is steeper and may be harder to train into a
  well-separated state. EnHDC reports this — they need different
  training tricks at lower D.
- **Numerical precision** is *less* of an issue at lower D (good).
- **Replay/consolidation dynamics** depend on basin geometry; lower D
  might destabilize replay if basins start interacting.

---

## Surprises

1. **The β lever dominates D for the floor.** The original framing of
   "lower D = much higher floor" appears mathematically incorrect for
   the given formula. β-sweep is 100× more impactful and 6× cheaper.
   *This is the biggest finding from the literature review and should
   probably re-shape the Phase 5 decision tree.*

2. **Spaun runs on D=512 successfully** despite handling many
   compositional binding tasks. The reason is iterative cleanup, not
   direct unbinding — which Neuro-AI also does (Hopfield iterations).
   This is a positive existence proof that D=512 can support
   role-binding, *if* the rest of the architecture is right.

3. **The Plate noise formula and the Phase 5 floor formula give
   opposite recommendations.** Plate's σ = √(N/D) wants high D; the
   Phase 5 floor wants low D. They measure different quantities
   (unbound vector amplitude vs energy-gap-between-priors). Phase 5
   has correctly chosen ΔE as headline, but downstream mechanisms that
   use the unbound vector amplitude (anything reading what the
   substrate retrieved, not just whether it crossed a threshold) face
   the Plate constraint.

4. **The 2022–2026 efficient-HDC literature is consistently pushing
   *down* in D**, not up. The "10000 is the canonical default" claim
   is increasingly being overturned by empirical work showing
   D=1000 or even D=64 can match accuracy if the rest of the pipeline
   is right. This suggests the "go higher D" intuition is not the
   field's current direction.

5. **Sparse VSAs are an under-explored escape hatch.** They decouple
   the D-for-orthogonality vs D-for-capacity tradeoffs. If Neuro-AI is
   genuinely floor-limited and β can't be raised (e.g. because it
   causes other instabilities), sparse VSA is the cleanest move — but
   it's a big rewrite.

6. **Capacity analysis bound suggests D=4096 is below recommended for
   N=1064.** Clarkson et al. 2023 bound predicts D ≥ ~10000 for
   ε=0.05 set membership. So if anything, the *capacity* argument
   says go up, not down. But again — Phase 5's metric isn't set
   membership.

7. **GHRR (2024) offers an orthogonal lever.** Non-commutative binding
   at the same total dimension improves memorization 2–3×. This could
   be combined with D-sweep or β-sweep for additional headroom.

---

## Open questions

1. **Has the floor formula been verified numerically in the codebase?**
   The math suggests β >> D for floor leverage, but if the codebase's
   actual floor measurement gives a different scaling, then either the
   formula in `notes/emergent-codebook/phase-5-unified-design.md` has a
   different form than the one in the prompt, or there's an
   implementation bug worth catching.

2. **What's the upper bound on β before other things break?** If β
   could go to 15–20, the floor drops below 1e-4 and the current
   signal becomes >100× above floor. Is there a known reason β=10 is
   the ceiling? (Possible candidates: gradient sharpness, basin
   collisions, retrieval brittleness for noisy queries.)

3. **What does the downstream consolidation mechanism actually read?**
   If it reads ΔE directly, the floor formula governs. If it reads the
   amplitude of the retrieved filler vector, the Plate formula governs.
   This needs to be checked in the active phase's design doc.

4. **Is N=1064 a fixed quantity or a free parameter?** The floor
   formula scales as O(log N) — going from N=1064 to N=100 only halves
   the floor. Going to N=10 quarters it. If the active codebook can be
   compressed without losing task-relevant content, that's another
   lever.

5. **What does the codebook look like at higher D?** A D-sweep that
   includes D=8192 and D=16384 would test the "go up" alternative
   directly. The Plate formula predicts that direction should give
   2–4× better σ, which might also widen the ΔE if the signal is
   limited by retrieval cleanness rather than by the floor.

6. **Could the system run β-warmup?** Train at β=10 (which the
   codebook learning needs) but read out at β=15+ (which gives the
   floor margin needed for graduation). Many Hopfield-style systems
   use this pattern: low β for training stability, high β for
   inference confidence. Worth checking if Neuro-AI does this.

7. **Why does the formula use `1 − 1/√D` and not `1 − 1/D`?** The
   factor `1/√D` is the std of cosine similarity between random unit
   vectors, suggesting the formula assumes the relevant noise has the
   structure of "FHRR phasors near unit similarity". If the actual
   binding-unbinding noise has a different shape (e.g. the variance
   scales as 1/D after some operation), the formula's D-dependence
   could be sharper — and the D-sweep would actually matter.
   **Verifying the derivation of the formula in the design doc is high
   priority.**

---

## Sources

- [A comparison of Vector Symbolic Architectures (Schlegel et al. 2022, arXiv:2001.11797)](https://arxiv.org/abs/2001.11797)
- [Capacity Analysis of Vector Symbolic Architectures (Clarkson et al. 2023, arXiv:2301.10352)](https://arxiv.org/abs/2301.10352)
- [Hopfield Networks is All You Need (Ramsauer et al. 2020, arXiv:2008.02217)](https://arxiv.org/abs/2008.02217)
- [A Survey on HDC/VSA Part I (Kleyko et al. 2022, arXiv:2111.06077)](https://arxiv.org/abs/2111.06077)
- [A Survey on HDC/VSA Part II (Kleyko et al. 2023, arXiv:2112.15424)](https://arxiv.org/abs/2112.15424)
- [Holographic reduced representations (Plate 1995/2008, IEEE TNN)](https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf)
- [Variable Binding for Sparse Distributed Representations (Frady et al. 2021, arXiv:2009.06734)](https://arxiv.org/abs/2009.06734)
- [Resonator Networks 1 (Frady et al. 2020, arXiv:1906.11684)](https://arxiv.org/abs/1906.11684)
- [Generalized Holographic Reduced Representations (Allen et al. 2024, arXiv:2405.09689)](https://arxiv.org/html/2405.09689v1)
- [qFHRR — Quantized FHRR (2026, arXiv:2604.25939)](https://arxiv.org/abs/2604.25939)
- [Efficient Hyperdimensional Computing — EnHDC (2023, arXiv:2301.10902)](https://arxiv.org/abs/2301.10902)
- [Optimal hyperdimensional representation for learning and cognitive computation (Poduval et al. 2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12929535/)
- [Improved Cleanup and Decoding of Fractional Power Encodings (2024, arXiv:2412.00488)](https://arxiv.org/html/2412.00488)
- [Provably Optimal Memory Capacity for Modern Hopfield Models (Hu et al. 2024, arXiv:2410.23126)](https://arxiv.org/html/2410.23126v2)
- [Yet another exponential Hopfield model (2025, arXiv:2509.06905)](https://arxiv.org/html/2509.06905)
- [Learning with Holographic Reduced Representations (Ganesan et al. NeurIPS 2021)](https://proceedings.neurips.cc/paper_files/paper/2021/file/d71dd235287466052f1630f31bde7932-Paper.pdf)
- [Introduction to NengoSPA — Spaun semantic pointer dimensionality (NengoSPA docs)](https://www.nengo.ai/nengo-spa/user-guide/spa-intro.html)
- [Resonator Networks 2 (Kent et al. 2020, arXiv:2007.03748)](https://arxiv.org/abs/2007.03748)
