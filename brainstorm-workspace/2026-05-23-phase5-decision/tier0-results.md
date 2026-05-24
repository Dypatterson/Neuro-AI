# Tier 0 Diagnostic Results — 2026-05-23

> **WALK-BACK #2 (2026-05-23, post-Path-A-implications-discussion):
> Tier 0.1's optimistic Path A trajectory table is CAVEATED.** The
> table below (line 97-105) shows Path A capacity-proportional
> ratios assuming the observed ΔE stays constant as N shrinks. That
> assumption is empirically contradicted by [report 053](../../reports/053_phase5_headline_n10_directional_subnoise.md)'s
> own §discussion text:
>
> > "signal scaled 50× from N=12 to N=1064 but noise scaled 95× over
> > the same range, so SNR is essentially unchanged (0.45 → 0.24)."
>
> **Implication:** if SNR is approximately invariant in N (with a
> slight worsening as N grows), then dropping N from 1024 to 266
> shrinks ΔE roughly proportionally with the floor — Path A's
> capacity-proportional trajectory likely lands at SNR 0.40-0.55,
> **still sub-floor by ~2×**. Path A moves the embarrassment from
> "4× below floor" to "2× below floor" but probably does not
> graduate Phase 5.
>
> **What Path A IS still good for:** strengthening the architectural
> finding for a closure paper. A deliberate scale-down probe that
> confirms substrate-saturation generalizes across scales becomes
> the seventh instance of the finding.
>
> **What Path A is NOT a viable path to:** rescuing Phase 5 graduation.
>
> **Net path ranking after this walk-back:**
> 1. **Path C first** (1-2 day Varner spike) — dominates Path A as
>    opening move; if it works Path A is unnecessary; if it fails
>    Path A is still on the table but with stronger evidence that
>    mechanism-level fixes are insufficient.
> 2. **Path B'** (close + PE-driven replay) — the honest version of
>    the SNR-invariance insight: the K-branch energy-margin mechanism
>    appears wrong-shaped for sharp-basin substrates at any scale.
>    Phase 4 already validated Δms_w3 as a metric class the
>    substrate exposes variance in; PE-driven replay reads from
>    that class.
> 3. **Path A** as a closure-paper strengthening probe (NOT as a
>    graduation rescue) — only if the user wants the publishable
>    finding to be unimpeachably scale-spanning.
>
> ---
>
> **WALK-BACK #1 (2026-05-23, post-Codex-review): Tier 0.2's PASS
> verdict is RESCINDED.** External review (Codex) identified that the diagnostic
> I built computes global pattern-pair stats `S = (μ_self − μ_cross)² / (σ_self² + σ_cross²)`
> over all stored patterns, NOT the Varner 2026 (arXiv:2603.20115)
> separation index `S = (c̄_within − c̄_between) / [½(σ_within + σ_between)]`
> which requires a **designated functional subset vs background**
> partition computed in **PCA space**. The cross-seed S = 39–42
> numbers measure a different quantity than the brainstorm's "S > 0.30
> implies log-prior bias works" framing assumed.
>
> **What's still valid in Tier 0.2:** the global pattern-pair
> measurements themselves (μ_cross = 0.252, σ_cross = 0.117, 7.5× the
> theoretical 1/√D floor) are real and tell us atoms have clustered
> onto a lower-dimensional manifold — but NOT zero-dimensional. The
> "basins collapsed" hypothesis is **not falsified** by this diagnostic;
> it remains open.
>
> **What changes in the recommendation:** Path C's "now top-ranked
> after empirical evidence" framing is downgraded. The Varner spike
> is still a reasonable 1-2 day exploratory experiment, but no longer
> with substrate-evidence backing. The Varner-correct diagnostic (per-
> cue functional-subset-vs-background PCA Fisher index) was not built.
>
> **Also fixed in the same walk-back:**
> - `verify_magnitude_floor.py` was using report 053's `observed_de = 1.3e-3`;
>   updated to use report 057's cross-seed β=10 mean `+0.002289` and
>   report 058's best-cell `+0.002478`. Corrected ratios make Path A's
>   capacity-proportional case *stronger*, not weaker (D=512, N=133:
>   ratio 1.40× → 2.47× the floor).
> - `.gitignore` had a project-wide `research/` rule that was silently
>   ignoring `brainstorm-workspace/2026-05-23-phase5-decision/research/`;
>   added `!brainstorm-workspace/**/research/` exception.
>
> See [STATUS.md banner](../../STATUS.md) for the project-level walk-back.
> Original Tier 0.2 result below is preserved as written; the corrected
> framing is in the WALK-BACK section after it.

---

Per the brainstorm's recommended sequence before committing to Path A
or Path B. Three checks, each under 2 hours. Run in service of the
user's standing question: *"make sure we are not missing anything
before we move forward."*

---

## Tier 0.1 — Magnitude-floor formula verification ✅

**Script:** [scripts/verify_magnitude_floor.py](../../scripts/verify_magnitude_floor.py)

**Spec formula** (from [magnitude-floor pre-commit note](../../notes/notes/2026-05-21-phase5-headline-magnitude-floor.md):49-73):

```
noise = 1 / sqrt(D)
floor = (1/β) · log(1 + (N-1) · exp(-β · (1 - noise)))
```

**Spec table reproduced exactly** (D=4096, β=10, varying N):

| N | spec | re-derived |
|---|---|---|
| 12 | 5.8e-5 | 5.84e-5 ✓ |
| 100 | 5.2e-4 | 5.24e-4 ✓ |
| 500 | 2.6e-3 | 2.61e-3 ✓ |
| 1064 | 5.5e-3 | 5.49e-3 ✓ |
| 4000 | 1.9e-2 | 1.92e-2 ✓ |

Formula is correctly stated in the spec.

### Verdict on STATUS.md framing

STATUS.md's framing ("at D=512 the noise floor would be ~8× higher;
at D=1024 it would be ~4× higher") is **NOT supported by the spec
formula at fixed N=1064**:

| D | floor at fixed N=1064 | ratio vs D=4096 |
|---|---|---|
| 256 | 8.63e-3 | **1.57×** (STATUS implied ~16×) |
| 512 | 7.24e-3 | **1.32×** (STATUS implied ~8×) |
| 1024 | 6.39e-3 | **1.16×** (STATUS implied ~4×) |
| 2048 | 5.85e-3 | 1.07× |
| 4096 | 5.49e-3 | 1.00× ← current |
| 8192 | 5.25e-3 | 0.96× |

Mechanism: `(1 − 1/√D)` is 0.96 at D=512 and 0.984 at D=4096, so
β·(1−1/√D) ≈ β dominates the exponent across this entire D range.
**The D-lever in the floor formula is weak.**

### Path A is NOT dead — but the premise has to change (and even then, see walk-back #2)

> **CAVEAT (walk-back #2):** the table below assumes ΔE stays at
> +0.002478 as N shrinks. The empirical record from [report 053](../../reports/053_phase5_headline_n10_directional_subnoise.md)
> §discussion ("signal scaled 50× from N=12 to N=1064 but noise scaled
> 95× over the same range, so SNR is essentially unchanged 0.45 →
> 0.24") contradicts that assumption. **Read the "Realistic ratio"
> column as the more likely outcome.** The "Optimistic ratio" column
> is preserved as the upper-bound case the Tier 0.1 analysis originally
> assumed; the SNR-invariance-realistic column is what report 053's
> own scaling implies.

There IS a trajectory through this formula where Path A might help:
**if lower D also produces fewer atoms** (capacity-proportional
substrate redesign). The Path-A scaling table at the **report 058
best-cell ΔE = +0.002478** (the operating-point signal, corrected
per Codex audit 2026-05-23 from the older report 053 value of
+0.00130 that the brainstorm originally cited):

| D | N (∝ D) | floor | optimistic ratio (signal=2.48e-3 const) | realistic ratio (SNR≈0.40-0.55) |
|---|---|---|---|---|
| 256 | 66 | 5.50e-4 | **4.51×** ← above floor | ~0.40-0.55× |
| 512 | 133 | 9.28e-4 | **2.67×** ← above floor | ~0.40-0.55× |
| 1024 | 266 | 1.63e-3 | **1.52×** ← above floor | ~0.40-0.55× |
| 2048 | 532 | 2.96e-3 | 0.84× | ~0.42-0.50× |
| 4096 | 1064 | 5.49e-3 | 0.45× ← current | 0.45× ← current (observed) |

**The realistic column matters more than the optimistic one.** Under
the empirical SNR-invariance from report 053, Path A's
capacity-proportional trajectory does NOT cross the floor at any
(D, N) point. The optimistic-column "1.52× above floor at D=1024"
is contingent on signal staying constant — which it almost certainly
won't.

For historical context, at the older report 053 ΔE = +0.0013 (the
value the brainstorm originally cited): D=512/N=133 was 1.40×
optimistic, sub-floor realistic. The Codex audit corrected ΔE to
report 058's +0.002478, but the SNR-invariance walk-back here is
what determines Path A's actual viability — neither ΔE value
changes the conclusion that signal and noise scale together with N.

If Path A is pursued anyway (e.g., as a closure-paper strengthening
probe rather than a graduation rescue), the cheapest viable D for a
deliberate scale-down probe is **D=1024 with N=266** (4× substrate
compression).

**This is the only trajectory through which Path A *could* rescue
the headline.** It requires committing to a substrate redesign that
simultaneously:

1. Reduces D to 1024 (or lower if extra margin is desired),
2. Reduces surviving atom count to ~266 at D=1024 (or proportionally
   fewer at lower D — so the substrate is genuinely smaller, not just
   embedded in fewer dims with the same 1064 atoms forced into it),
3. **Argues that the role-prior signal does not shrink proportionally
   with N** — i.e. that report 053's empirical SNR-invariance
   observation does NOT extrapolate to the new operating point. This
   is the load-bearing architectural claim. Fewer schemas means less
   discriminative capacity in the population-averaging reading; only
   survives if the signal happens to be set by per-atom geometry
   rather than population mass. **No empirical evidence currently
   supports the per-atom reading; report 053's data supports the
   population-mass reading.** Without a counter-argument backed by
   either theory or a small empirical probe, requirement #3 is
   probably unmet — meaning Path A's "rescue" framing collapses to
   "scale-down probe that documents the failure at a second scale."

### What this changes about Path A

The original brainstorm framed Finding 1 as "Path A is weakly
supported by the formula." After two walk-backs, the correct framing
is:

> **Path A is unlikely to graduate Phase 5 under report 053's
> observed SNR-invariance scaling.** It might bring the signal closer
> to the floor but probably not cross it. Its real value is as a
> deliberate scale-down probe that strengthens the substrate-
> saturation finding for a closure paper — confirming "the failure
> generalizes across scales" makes the architectural finding
> publishable in the strongest form.

If the user pursues Path A anyway, the first action is not the
D-sweep diagnostic but a design-note argument addressing **why
report 053's SNR-invariance scaling should NOT apply at the new
operating point**. Candidate reasons (none currently load-bearing):
- The K-branch signal source might be per-atom geometry rather than
  population averaging; report 053's scaling assumed the latter.
- Lower-D substrates might expose different basin-discrimination
  dynamics that aren't captured by the magnitude-floor formula.
- The discovery channel might saturate at smaller N in a way that
  changes the signal/noise relationship.

Without one of these arguments empirically supported, D-sweep work
would chase a ~1.3× lever instead of the ~3–5× the optimistic
capacity-proportional trajectory would have delivered — and the
realistic SNR-invariant trajectory delivers no floor crossing at all.

### Pessimistic trajectory check

If D shrinks but N stays at 1064 (i.e., mass-death dynamics don't
scale capacity with D, which is plausible because A+B+A1' actively
manages substrate population independent of D):

| D | N=1064 | floor | ratio vs floor |
|---|---|---|---|
| 256 | 1064 | 8.63e-3 | 0.15× |
| 512 | 1064 | 7.24e-3 | 0.18× |
| 1024 | 1064 | 6.39e-3 | 0.20× |
| 4096 | 1064 | 5.49e-3 | 0.24× ← current |

In this case Path A makes things WORSE, not better. The floor at lower
D rises (1.3×) while the signal probably doesn't (capacity-blind).

### Net for Tier 0.1

**Path A is salvageable but only along a capacity-proportional
trajectory.** The user needs to make that commitment explicitly in a
design note before D-sweep work begins. If the commitment can't be
made, Path A is foreclosed and the strategic tree reduces to:

- Substrate-pure mechanism redesign (Input-Driven Plasticity Hopfield,
  Varner log-prior softmax bias, LSR, Dynamic Manifold Hopfield — see
  brainstorm Tier 1/2 ideas)
- Path B (close + pivot to pair #4)

---

## Tier 0.2 — Fisher separation index ✅

**Script:** [scripts/fisher_separation_diagnostic.py](../../scripts/fisher_separation_diagnostic.py)

**Status:** complete on three seeds of the actual A+B+A1' substrate.

### Calibration on synthetic substrates

Random-FHRR baseline (what the substrate "should" look like if A1'
preserved random orthogonality):

| Substrate | μ_self | μ_cross | σ_cross | Fisher S |
|---|---|---|---|---|
| random D=4096, N=1064 | 1.000 | 0.000 | 0.011 | **8191** |
| random D=512, N=133 (Path A target) | 1.000 | 0.000 | 0.031 | **1014** |

d_eff-collapsed simulation (atoms spanning n_base effective dims):

| Substrate | μ_self | μ_cross | σ_cross | Fisher S |
|---|---|---|---|---|
| collapsed D=4096, N=1064, n_base=40 | 1.000 | 0.143 | 0.127 | **45.7** |
| collapsed D=4096, N=1064, n_base=5 | 1.000 | 0.237 | 0.365 | **4.4** |

### Key calibration finding (refines brainstorm Finding 2)

The Varner 2026 (arXiv:2603.20115) Fisher-S threshold of 0.30 is a
**very coarse filter** at D=4096 with N~1000 atoms. Even a synthetic
substrate with severe d_eff collapse (n_base=5 — only 5 effective
spanning directions for 1064 atoms) gives S = 4.4, an order of
magnitude above the 0.30 PASS threshold.

This means: **a PASS verdict from this diagnostic on the real
substrate is necessary but not sufficient** to conclude "K-branch
failure is a softmax-bias issue." The substrate's *patterns* are
almost certainly separable in cosine space; the question Phase 5
actually hits is whether *priors* bias settling enough to discriminate
between patterns at retrieval time.

If the real-snapshot Fisher S comes back below 0.30, that would be a
much more severe finding than the brainstorm anticipated — it would
mean even my n_base=5 synthetic is more optimistic than reality.

### What the diagnostic still tells us

The relevant information is the **comparison to random baseline**, not
the threshold:

- If real-snapshot Fisher S is **within 2-3 orders of magnitude of
  8191** (e.g., S > 100): the substrate retains substantial pattern
  separability; Varner log-prior softmax bias and Betteti IDP are
  both worth trying.
- If real-snapshot Fisher S is **between 1 and 100**: the substrate
  is somewhere on the collapse continuum between n_base=40 and
  n_base=5; mechanisms that operate on the per-atom level may still
  work but the substrate is much more degenerate than the design
  intended.
- If real-snapshot Fisher S is **near or below 1**: pattern geometry
  has substantially collapsed; mechanism-level fixes (IDP, log-prior)
  are unlikely to rescue and a substrate redesign (Path A) becomes
  the only viable substrate-fix path. Closure (Path B + pair #4)
  becomes proportionally stronger.

### Real-substrate result — cross-seed verdict

Three seeds of the n=10 headline campaign substrate
(`phase5_headline_substrate_seed{N}/snapshots/phase3_phase4_w4_step1800.pt`,
downloaded via Drive MCP) run through the diagnostic:

| Seed | N atoms | Fisher S | μ_cross | σ_cross | σ_cross / (1/√D) |
|------|---------|----------|---------|---------|------------------|
| 17 | 1024 | **39.31** | 0.260 | 0.118 | 7.55× |
| 11 | 1024 | **42.34** | 0.250 | 0.115 | 7.38× |
| 23 | 1024 | **39.91** | 0.247 | 0.119 | 7.63× |
| mean | 1024 | **40.5** | 0.252 | 0.117 | 7.52× |

For comparison, the binary-death substrate (12 atoms, seed 17,
pre-A1') gave S = 10.55 — already passing, but with much higher
cross-correlation (μ_cross = 0.38, σ_cross = 0.19).

### Verdict: ALL THREE SEEDS PASS by 2 orders of magnitude

**The substrate-saturation finding is NOT a "basin-collapse" finding.**
The A1' substrate retains substantial pattern separability — Fisher S
sits at ~40 across all three seeds tested, well above the 0.30
threshold. The patterns are demonstrably distinguishable in cosine
space.

This **falsifies the strongest possible reading of the
substrate-saturation finding** — namely that basin geometry itself has
collapsed. It does not falsify the finding entirely (the K-branch
ΔE is still sub-floor for substantive architectural reasons), but it
relocates the failure mode:

- **NOT a basin-geometry problem.** S=40 means patterns are well-
  separated; basins exist; in principle distinguishable.
- **Likely a prior-bias problem.** Even though basins are separable,
  the current per-pattern prior bias
  `prior_bias = γ · similarity(prior, patterns)`
  is not steep enough to pull settling into different basins for
  role-prior vs content-prior branches. This is the regime where
  Varner 2026 (arXiv:2603.20115) Pattern Multiplicity predicts a
  log-prior softmax bias will help.

### Substrate is moderately collapsed but not pathologically so

The μ_cross = 0.25 across seeds is notable. For independent random
FHRR patterns (no correlation), μ_cross ≈ 0 with σ_cross ≈ 1/√D.
Observed σ_cross is **7.5× the theoretical noise floor**, meaning
the A+B+A1' mechanism has indeed clustered the 1024 atoms onto a
manifold lower-dimensional than D=4096.

Mapping to the synthetic calibration:

| Substrate | μ_cross | σ_cross | S |
|---|---|---|---|
| Random D=4096 N=1064 | 0.00 | 0.011 | 8191 |
| Synthetic collapsed n_base=40 | 0.143 | 0.127 | 45.7 |
| **A1' real (seeds 17/11/23)** | **0.25** | **0.117** | **40** |
| Synthetic collapsed n_base=5 | 0.237 | 0.365 | 4.4 |

The real substrate sits *between* the n_base=40 and n_base=5
synthetic regimes — closer to n_base=40 in σ_cross (basin width)
but closer to n_base=5 in μ_cross (basin center pull). Intuitively:
the A1' substrate has atoms clustered around something like 30–50
effective modes, but with tighter intra-mode variation than the
collapse simulation produced.

### What this means for the strategic decision

> **WALK-BACK (Codex audit 2026-05-23):** the prior text of this
> section (preserved below in struck-through form) claimed *"Path C
> Varner log-prior bias is strongly supported by the substrate's
> geometry — S > 0.30 passes by 130×."* That claim is **withdrawn**.
> The S = 40 number measures global pattern-pair separability, NOT
> the Varner per-cue functional-subset Fisher index (which requires
> a within/between PCA split — not what this script computed). The
> "S > 0.30 ⇒ log-prior bias works" prediction therefore does NOT
> apply to this diagnostic's output.

What the global pattern-pair stats *do* legitimately tell us:

- **Atoms are not collapsed to a single ray** (S=40 ≫ 0 means there
  IS pairwise separability). The strongest reading of basin collapse
  is ruled out by these numbers.
- **Atoms ARE clustered on a lower-dimensional manifold** than
  D=4096 (σ_cross = 7.5× the theoretical 1/√D floor; μ_cross = 0.25
  vs ~0 for random). The substrate has structure beyond independent
  random patterns.

What they do **NOT** tell us:

- Whether priors can bias settling into discriminable basins (this is
  the actual Phase 5 K-branch question).
- Whether Varner-style log-prior bias would help (that requires the
  Varner-correct per-cue functional-subset/background PCA index,
  which was not built).

**Strategic implication:** Path C (Varner log-prior spike) remains a
reasonable cheap exploratory experiment (1-2 days, substrate-pure,
no Phase 4 retrain), but it is no longer "empirically the strongest"
next move. The Tier 0.2 result narrows the space of possible
mechanisms (basin geometry is not the limiting layer; the limiting
layer is somewhere between "priors don't bias enough" and "the
substrate's mode clustering creates artefacts paired ΔE can't
resolve"), but does not pick a path.

The Path A capacity-proportional case has *strengthened* (see Tier
0.1 update with corrected ΔE — D=1024/N=266 now crosses the floor
too). The Path B' case is unchanged.

---

## Tier 0.3 — Pair #4 headline metric audit ✅

**Source:** [2026-05-20 pair #4 design note](../../notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md)
vs [phase-5-unified-design.md](../../notes/emergent-codebook/phase-5-unified-design.md):256-281.

### What the Phase 5 design spec says (load-bearing)

[phase-5-unified-design.md:256-277](../../notes/emergent-codebook/phase-5-unified-design.md):

> **Structural retrieval verified iff Δ final-state energy is
> CI-disjoint from zero, role-prior branches vs content-prior
> branches, on a held-out cue set designed for structural retrieval.**
> [...]
> Headline: mean ΔE across n_seeds × n_cues, with 95% CI.
> If CI excludes zero, the system retrieves structurally — role priors
> land at lower-energy joint states than content priors. **This is the
> Phase 5 graduation criterion.**

Augmented by the [magnitude-floor pre-commit](../../notes/notes/2026-05-21-phase5-headline-magnitude-floor.md):86-98:

> **Mean ΔE ≥ 5.5e-3 AND bootstrap 95% CI lower bound > 0**

### What the pair #4 design note says

[pair #4 design note:320-324](../../notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md):

> **Phase 5 headline (graduation criterion):** **Δ meta_stable_rate
> at W=3 with `m_i`-weighted replay vs the κ=0 control**, n ≥ 10
> seeds, **CI disjoint from zero in the direction of reduced
> metastability** (Δ ≤ −0.1, CI upper bound < 0).

And explicit pivot statement at [pair #4 design note:498-501](../../notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md):

> The Phase 5 graduation criterion moves from "ΔE_K4 CI-disjoint from
> zero" (the chase that just ended) to "Δ meta_stable_rate at W=3
> CI-disjoint from zero" (the substrate-pure metric class Phase 4 D1
> already validated).

### Finding: explicit pivot, NOT silent drift

The brainstorm's Finding 4a flagged this as a potential "silent
discipline drift." That framing was wrong. The pair #4 design note
**explicitly acknowledges** the pivot from ΔE to Δms_w3. This is a
deliberate redirection of the Phase 5 graduation target, written
down, with literature grounding (HEN, Saighi, Benna-Fusi) and an
anti-homunculus PASS.

### The actual discipline issue

The pivot is *recorded in a pair #4 design note* but **not reflected
in the Phase 5 unified design spec**. Per CLAUDE.md session-start
protocol:

> The §"Headline metric" + §"Required controls" sections of the active
> phase's design document [are] the load-bearing source of truth for
> what "graduation" means. [...] STATUS.md banners can drift away from
> the design spec over multiple sessions.

A future agent in a fresh session reading
phase-5-unified-design.md:256-281 would see ΔE as the Phase 5 headline
— not Δms_w3. This is the failure mode CLAUDE.md was tightened to
prevent. **The pivot is correct discipline at the pair-#4 layer but
incomplete discipline at the project layer.**

### Three additional concerns surfaced by the audit

1. **The Δ ≤ −0.10 bar is not labeled as differential.** Phase 4 D1
   already achieved Δms_w3 = −0.79 absolute
   ([report 038](../../reports/038_phase4_d1_graduation.md)). Pair #4's
   Δ ≤ −0.10 is the *κ-vs-κ=0 differential* (pair #4 mechanism's
   marginal contribution over Phase 4 baseline). The design note's
   text doesn't make this explicit. Risk: a future report could
   confuse the two and either over-celebrate a 10pp differential or
   under-celebrate it as "much worse than Phase 4." Edit to design
   note text: replace "Δ ≤ -0.1" with "Δ ≤ -0.1 *of κ-vs-κ=0 control,
   on the same seeds*".

2. **Pair #4 has already failed its own pre-committed smoke gate.**
   Per STATUS.md (2026-05-21 walk-back banner) and report 054:
   on the A1' substrate the trajectory-c_i smoke produced
   `m_max ≈ 0.0082`, below the pre-committed `m_max > 0.05` gate.
   STATUS describes the cleaner result as "substrate-wide
   architectural finding now has FOUR instances" — i.e. pair #4 is
   substantively foreclosed already, in addition to the Phase 5
   headline being saturated.

   This means pair #4 in its current design **is not a viable Path B**.
   The brainstorm's metastability-research agent noted this too:
   metastability-as-signal is poor; surprise / prediction-error
   (UPER, ReaPER, SuRe — 2024–2025) is the cleaner signal class.

3. **The audit constraint #10 framing.** Pair #4 changing the Phase 5
   headline from ΔE to Δms_w3 is **not** a "retuning of κ/μ_obs/β/γ/
   K_main/formulation" per audit constraint #10. It's a different
   *metric class*, which is permitted by the constraint. But it IS a
   change to the design spec's headline, which requires explicit user
   agreement per the standing discipline (CLAUDE.md:
   "If you find a contradiction between [STATUS.md and the active
   checklist] and another document mid-session — including the active
   phase's design spec — that is itself a finding to surface, not
   paper over.")

   This audit IS the surfacing.

### Recommendation for Path B

Path B as currently designed has two problems:

1. The pair #4 smoke has already failed its own pre-committed gate
   (m_max < 0.05). Pursuing pair #4 implementation now would be
   retroactively justifying it against its own falsification, which
   is exactly the failure mode audit constraint #10's spirit
   prohibits.

2. The metric pivot from ΔE to Δms_w3 lacks a Phase 5 spec update.

If Path B is the right strategic move, the correct sequence is:

- **B.1** Surface the pair #4 smoke-falsification status as a STATUS
  walk-back (it's already partially documented in the 2026-05-21
  banner; promote it to a first-class finding).
- **B.2** Reframe pair #4 as **falsified for the trajectory-c_i
  formulation specifically**, with a fresh pivot to surprise/PE-driven
  replay (per the metastability-research agent's findings:
  UPER arXiv:2506.09270, ReaPER arXiv:2506.18482, SuRe arXiv:2511.22367
  — biological consensus is **prediction-error**, not metastability).
- **B.3** Write a Phase 5 closure note that:
  - Documents the substrate-saturation finding (6 instances) as the
    primary Phase 5 deliverable.
  - Explicitly updates the Phase 5 spec headline from ΔE to whatever
    Phase 5'/closure-pivot headline the new mechanism uses.
  - Cites the pair #4 architectural lesson (HEN-style trajectory
    reformulation is necessary on sharp-basin substrates, AND even
    that lesson didn't graduate — the substrate is saturated at a
    deeper layer).
- **B.4** Implement and run the prediction-error-driven replay
  mechanism with the new headline, n=10.

This is a longer Path B than the brainstorm framed (the "lightweight
contingency" framing assumed pair #4 was implementation-ready, which
it isn't post-smoke-falsification).

### Net for Tier 0.3

The pair #4 pivot was **explicitly designed**, not silently drifted —
but it's **already been smoke-falsified** in a way the STATUS
2026-05-21 banner partially acknowledges but the pair #4 design note
itself does not.

**Path B in the brainstorm's framing ("½ day to close pair #4") is
not actionable as-is.** A Path B' that:
1. Acknowledges pair #4 trajectory-c_i is falsified,
2. Reframes as surprise/PE-driven replay (UPER/ReaPER/SuRe),
3. Updates the Phase 5 spec headline explicitly,
…would be ~1 week, not ½ day.

---

## Combined verdict — what the Tier 0 trio changed about the
## strategic decision

Both Path A and Path B have changed materially:

### Path A (was: "lower-D redesign, 2-3 weeks")

Now: **probably does not graduate Phase 5, but is good for
strengthening the closure paper.** Tier 0.1's optimistic table
assumed signal stays constant as N shrinks, but report 053's
empirical scaling (signal scaled 50×, noise scaled 95× from N=12 to
N=1064, SNR roughly invariant) suggests Path A's capacity-
proportional trajectory lands at SNR ~0.40-0.55 — still sub-floor by
~2×. The user can either argue (with new evidence) that report 053's
scaling does not extrapolate to the new operating point, OR re-frame
Path A as a deliberate scale-down probe to confirm the substrate-
saturation finding generalizes (making the closure paper much
stronger). Without a counter-argument to report 053's scaling, Path A
should NOT be framed as a graduation rescue.

### Path B (was: "close + pivot to pair #4, 4 days")

Now: **not actionable as designed.** Pair #4 trajectory-c_i has
already failed its own pre-committed smoke gate. Path B' (close +
pivot to surprise/PE-driven replay) is the actionable form, costing
~1 week and requiring a Phase 5 spec update.

### Path C — Varner log-prior softmax bias (NOT "first-priority" — see walk-back below)

> **WALK-BACK (Codex audit 2026-05-23):** the prior recommendation here
> ranked Path C as "first-priority after Tier 0" on the basis of
> S = 40 passing the "Varner threshold S > 0.30." **That ranking is
> withdrawn.** The S computed by [scripts/fisher_separation_diagnostic.py](../../scripts/fisher_separation_diagnostic.py)
> is global pattern-pair separability, not the Varner per-cue
> functional-subset/background PCA Fisher index — so the threshold
> Varner specifies does not apply to that number.

What we still know about Path C after the walk-back:

- It is a substrate-pure mechanism change: add a per-pattern
  `log_prior_bias` tensor parallel to the existing `prior_bias` in
  [experiments/40_phase5_branching.py:603](../../experiments/40_phase5_branching.py:603).
- It does not require a Phase 4 retrain; runs on the locally-cached
  seed-17 snapshot.
- ~½ day implementation + tests (including a `log_prior_gain=0`
  non-regression test, load-bearing for bit-identical baseline), then
  ½ day Colab spike, then 1 day write-up.
- The Varner threshold prediction is **untested** on the Phase 5
  substrate. Running the spike directly *is* the test.

What we no longer have evidence for: that the spike "is predicted to
work" by the substrate geometry. The honest framing is **exploratory
spike**, not **evidence-supported next move**.

If the user wants a Varner-correct prediction before committing to
the spike, the diagnostic to build is: per-cue, partition the schema
store into "designated for role-prior branch" (top-K_main by FHRR
unbind similarity with the cue's role-binding) vs "background" (the
rest), do PCA on the substrate, compute
`S = (c̄_within − c̄_between) / [½(σ_within + σ_between)]` in PCA
space, and apply Varner's 0.30 threshold. That is ~1 day of work in
its own right.

### Recommended next session

1. **Decision:** the user picks the sequence informed by Tier 0 +
   both walk-backs. Recommended ordering:

   - **Step 1 — Path C first (Varner log-prior spike, 1-2 days):**
     dominates Path A as an opening move under any reasonable
     expected-value calc. Cheap, substrate-pure, no Phase 4 retrain.
     If it works, K-branch failure closes without further substrate
     or mechanism work. If it fails, you've learned that mechanism-
     level prior-bias is insufficient — making Path A's "scale-down
     probe" or Path B's "close + pivot" frame much stronger.
   - **Step 2 — Path B' OR Path A (after Path C result):**
     - If Path C closed the K-branch failure: skip both, write up
       and move to Phase 6 planning.
     - If the user wants honest closure now: **Path B'** (close +
       pivot to surprise/PE-driven replay per UPER/ReaPER/SuRe,
       ~1 week, requires Phase 5 spec update). Phase 4 already
       validated the Δms_w3 metric class; PE-driven replay reads
       from the metric class the substrate exposes variance in.
     - If the user wants the closure paper to be as strong as
       possible: **Path A as scale-down probe** (NOT as graduation
       rescue, ~2-3 weeks). Confirms substrate-saturation
       generalizes across scales. Architectural finding becomes the
       seventh instance. Don't expect Phase 5 graduation; do expect
       publishable scale-spanning confirmation.

2. **Independent of path choice:** update [phase-5-unified-design.md:256-281](notes/emergent-codebook/phase-5-unified-design.md:256)
   to reflect whichever direction is chosen. This closes the
   discipline gap Tier 0.3 surfaced. **The agent did NOT silently
   make this edit — the user should approve the spec change
   explicitly.** Draft language for each path is in the
   per-path sections above.

The original brainstorm's claim — "make sure we are not missing
anything before we move forward" — is answered: **yes, four things
(Tier 0.1, 0.3, the Codex follow-up audit, and the SNR-invariance
follow-up discussion), each of which changes the strategic decision
tree non-trivially**. Tier 0.2's diagnostic returned real numbers
(μ_cross = 0.25, σ_cross = 0.117, S_global = 40 across three seeds)
but the Varner-threshold interpretation those numbers were originally
given is **withdrawn** per Codex audit — the script measured global
pattern-pair separability, not the per-cue functional-subset/
background PCA Fisher index Varner defines. The "basin collapse
falsified" reading is similarly downgraded to "global pattern-pair
separability rules out the strongest form of basin collapse, but the
relevant per-cue basin-discrimination question remains open."

Net path ranking after BOTH walk-backs:

- **Path C first** (1-2 day Varner spike): cheapest and dominates
  Path A as an opening move; no longer "empirically supported" but
  remains the cheapest exploratory probe.
- **Path B'** (~1 week, close + PE-driven replay): the honest version
  of the SNR-invariance insight — substrate exposes variance in the
  Δms_w3 metric class, not the K-branch energy-margin class. Phase 5
  closure framing reads naturally from this.
- **Path A** (2-3 weeks): no longer framed as a graduation rescue.
  Useful as a deliberate scale-down probe to strengthen the closure
  paper, if the user wants the substrate-saturation finding to be
  unimpeachably scale-spanning.

The user's choice is now substantially better-informed (the
SNR-invariance walk-back surfaces a load-bearing empirical scaling
that the Tier 0 framing assumed away) but still genuinely open. The
deeper question Path A's walk-back raises — whether the K-branch
mechanism is wrong-shaped for sharp-basin substrates as a class —
points toward Path B' being the most honest direction, but Path C
first is still the right opening move because it is so cheap to
falsify.

---

## Files produced

- [scripts/verify_magnitude_floor.py](../../scripts/verify_magnitude_floor.py)
- [scripts/fisher_separation_diagnostic.py](../../scripts/fisher_separation_diagnostic.py)
- This document
