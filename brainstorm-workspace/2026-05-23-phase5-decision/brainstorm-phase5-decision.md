# Brainstorm: Phase 5 Decision Point — What Are We Missing?
> Generated 2026-05-23, post-report-058 (cross-seed cue-regime sweep).
> Context drawn from: phase-5-unified-design.md, phase-{2,3,4}-* design notes,
> reports 026 / 038 / 041–058, notes/notes/ synthesis dated 2026-05-09 through
> 2026-05-21, and the 2026-05-20 prior brainstorm.
> Research: 5 parallel agents, ~50 web searches + ~30 paper fetches, 2024–2026 primary lit.

## Project Understanding

Phase 5 (HAM × energy-guided structural branching wrapped around a post-death
Phase 4 substrate) has reached a substrate-saturation finding at D=4096 with
six independent instances. The headline ΔE (role-prior vs content-prior
energy margin) sits 4.2× below the pre-committed magnitude floor (5.5e-3).
Three of four strategic options are foreclosed empirically — option 2
(sub-floor advance) by the magnitude gate, option 3 (basin-membership/R@K
reformulation) by report 058's ΔE/basin-hit anti-correlation, option 4
(basin-shape priors) by the absence of exploitable basin geometry. The
user's working assumption is that only option 1 (lower-D redesign) remains
plausible, with a "Path B" contingency to close Phase 5 and pivot to pair
#4 (metastability/replay-prioritization).

The deeper cross-phase picture: Phase 4 graduated on D1 (Δms_w3 = −0.79, CI
disjoint) using a **binary mass-death** mechanism that collapses substrate
d_eff from ~40 to ~5 of 4096 dims. The A+B+A1' continuous-form replacement
was designed for Phase 5 to preserve d_eff, but the A1' substrate still ends
up with 1064 atoms tied at |E_i| ≪ ε = 0.05 — sharp self-retrieving basins
that, as report 052 noted, foreclose ANY mechanism whose signal is softmax-
derived per-atom variance. **The architectural finding is real: clean-basin
substrates at D=4096 are incompatible with energy-margin prior-traversal.**

The user explicitly asked: look at previous phases (not just Phase 5) to
make sure we are not missing anything before committing to either path.
The research surfaced **at least three findings that should change the
decision tree before the next session ends**.

---

## TOP-LEVEL FINDINGS (the things we appear to be missing)

### 🔴 FINDING 1 — The "lower-D rescues the floor" assumption is mathematically suspect

The STATUS.md framing of option 1 reads: *"at D=512 the noise floor would be
~8× higher; at D=1024 it would be ~4× higher."* The dimensionality-scaling
research agent **re-derived the floor formula** from
`phase-5-unified-design.md` (`(1/β)·log(1 + (N−1)·exp(−β·(1−1/√D)))`) and
found:

| D | floor (β=10, N=1064) |
| - | - |
| 512  | ~7.3e-3 |
| 1024 | ~6.3e-3 |
| 2048 | ~5.8e-3 |
| 4096 | ~5.5e-3 (current) |
| 8192 | ~5.3e-3 |

That's **~1.3× variation across a 16× range of D, not 8×.** Mechanism:
`(1 − 1/√D)` is already 0.96–0.99 for any D > 256, so β·(1−1/√D) ≈ β dominates
the exponent. **The D lever in the floor formula is weak.** By contrast, at
fixed D=4096:

| β | floor |
| - | - |
| 10 | 5.5e-3 (current) |
| 15 | ~2.7e-5 (200× lower) |
| 20 | ~1.4e-7 |

**This is the most important pre-commit-to-anything finding.** Either:
1. The formula in the design spec is missing a term and the agent's
   derivation is wrong, OR
2. Option 1 (lower-D redesign) is not the right axis at all — the actual
   sensitivity is in β, which audit constraint #10 forbids retuning.

The **first action before this brainstorm closes** should be a 30-minute
walk-through of the formula derivation in
`notes/emergent-codebook/phase-5-unified-design.md` lines ~256–281 against
the agent's re-derivation. If the agent is right, the "lower-D redesign"
path needs re-justification on grounds other than the floor formula (e.g.,
Plate unbind-amplitude noise `σ = √(N/D)` which goes the *opposite* way —
lower D makes the unbind noise WORSE, not better).

> **Why this matters:** if lower-D doesn't actually rescue the floor as the
> STATUS framing implies, then Path A (D-sweep + Phase 4 retrain over 2–3
> weeks) is investing weeks against a weak lever. Either β-rederivation
> changes things, or the path forward is operator-level (Finding 2) or
> substrate-mechanism-level (Finding 3), not dimension-level.

See [research/dimensionality-scaling.md](research/dimensionality-scaling.md)
for the full derivation table and the Plate-vs-floor reconciliation.

---

### 🔴 FINDING 2 — Input-Driven Plasticity (Betteti 2025) is a substrate-pure mechanism the project's notes don't reference

**Betteti et al., "Input-driven dynamics for robust memory retrieval in
Hopfield networks," Science Advances 2025 (arXiv:2411.05849).** Defines
`W(u) = Σ_μ α_μ(u) · ξ_μ ξ_μ^T` — the synaptic matrix itself becomes
input-dependent, with closed-form coefficients (no training). Proves
analytically that different inputs reorder basin depths — which is exactly
the substrate-pure mechanism Phase 5 has been searching for in the form of
"prior-traversal that doesn't require sharp basins to be soft."

Where this lands relative to Phase 5:
- Anti-homunculus compatible: `α_μ(u)` is a closed-form local function of
  the input, not a controller reading a metric and arbitrating.
- Substrate-pure: lives in the substrate energy, not in a readout.
- Doesn't undo Phase 4 sharp-basin engineering: basins stay sharp; their
  *relative ordering* is what becomes input-dependent.
- Independent of D: the mechanism is dimension-agnostic.

**Concrete idea:** treat the role-prior or content-prior as the input `u`
that modulates `α_μ`. The K=4 branches then settle on substrates with
*different* basin orderings, which is the property the substrate-saturation
finding says is currently absent.

The basin-engineering agent's recommendation here is **measure the Fisher
separation index** of the surviving atoms first (single ~30-line
computation). If Fisher S > 0.3 the K-branch failure is a softmax-bias
issue and a one-line change to settling logits (`b_k = log r_k` per Varner
2026, arXiv:2603.20115) fixes it. If S < 0.2 the basin geometry truly
collapsed and IDP / LSR / Dynamic Manifold Hopfield surgery is needed.

> **Why this matters:** if either Varner's log-prior softmax bias OR Betteti's
> IDP turns out to be the right shape on this substrate, Phase 5 has a
> path forward that doesn't require lower-D redesign OR pivot. Both are
> small experiments (~1 day each), substrate-pure, and well-precedented.

See [research/basin-engineering.md](research/basin-engineering.md) for full
treatment of HEN, Energy Transformer, Sparsemax-Hopfield, LSR
(arXiv:2506.10801), Dynamic Manifold Hopfield (arXiv:2506.01303), Dead
Neurons (arXiv:2410.13866), and Langevin Stochastic Attention
(arXiv:2603.06875).

---

### 🔴 FINDING 3 — The Phase 5 K-branch premise has a one-day discriminating experiment that nobody has run

The substrate-saturation finding rests on six instances, all on **FHRR
convolution-binding**. The binding-operators research agent argues this is
not just a Neuro-AI accident — the `1/√D ≈ 0.0156` noise floor is the
**Plate/Frady bound for dense complex-phasor binding** and is operator-family-
limited. Most modern VSA work (GHRR, HLB, MCR) preserves the same `1/√D`
scaling.

But two operators move the floor:
- **Permutation binding** (Recchia 2015, Kanerva) — unary, no crosstalk by
  construction. Already flagged in the 2026-05-20 brainstorm as
  Tier-2/Phase-6.
- **Tensor Product Representation** (TPR) — zero crosstalk at orthogonal
  roles. At D=64 the TPR has dim D²=64²=4096, same memory footprint as
  current FHRR.

**The agent's strongest concrete proposal:** run TPR at D=64, role-vocab
matched to the schema-store cardinality, on the existing Phase 5 cues.
**Zero crosstalk by construction.** If the ΔE gap closes → operator
crosstalk is the binding limit and Phase 5 needs operator surgery, not
substrate redesign. If the ΔE gap stays sub-floor → the saturation is
**operator-independent** and we have a much stronger architectural finding
(publishable per Finding 5 below).

> **Why this matters:** this single 1-day experiment discriminates between
> "Phase 5's failure is operator-level" and "Phase 5's failure is
> substrate-architecture-level." The user's current strategic options can
> only be properly ranked once that question is answered. AND — surprise
> finding from the agent — no published VSA paper benchmarks
> prior-traversal SNR (they all benchmark *retrieval* SNR). This regime
> is *unmapped* in the literature, which means a clean experiment here is
> contributory.

See [research/binding-operators.md](research/binding-operators.md) for
GHRR (arXiv:2405.09689 — non-commutative binding gives intrinsic role/
content asymmetry), HLB (arXiv:2410.22669 — stable norm under iterative
settling), and the SNR-vs-D table for all 9 operator families.

---

### 🟡 FINDING 4 — Pair #4 may be optimizing the wrong metric, and trajectory-c_i has a 4-year-old name in the RL literature

Two findings about Path B (pair #4 pivot):

**4a.** Pair #4 is currently scoped to graduate on `Δ meta-stable-rate at
W=3` — a Phase 4 substrate metric. But the Phase 5 design spec headline at
[phase-5-unified-design.md:256-281](notes/emergent-codebook/phase-5-unified-design.md)
is ΔE. Per CLAUDE.md's "experiment preamble requirement," a Phase 5
graduation experiment must cite the Phase 5 design-spec line for its
headline. **Graduating Phase 5 on a Phase 4 metric is a discipline drift
worth flagging before the pair #4 implementation lands.**

The metastability-replay agent's most parsimonious proposal: collapse
`m_i` into a per-atom version of the design-spec ΔE
(`ΔE_i = E_i^{content} − E_i^{role}`). This (a) keeps the headline on the
Phase 5 design spec; (b) matches the SuRe surprise-driven precedent
exactly (arXiv:2511.22367); (c) is derivable from existing instrumentation
because the K-branch run already computes per-atom branch energies.

**4b.** The trajectory-c_i reformulation (`max_t w_i(t) − w_i(final)`,
designed 2026-05-20) is **structurally identical to "hindsight regret"** in
Liu et al. ReMERN/ReMERT (NeurIPS 2021, arXiv:2105.07253). Re-naming gives
the signal a 4-year RL provenance and connects pair #4 to a much larger
literature than the Saighi neuromorphic note alone. Concretely:
hindsight-regret-modulated replay priority has clean precedent; the design
note can cite it instead of inventing the wheel.

> **Why this matters:** pair #4 is the cheapest path to close Phase 5, but
> it has a metric-discipline issue and was missing a literature anchor.
> Both are addressable with edits to the existing design note before any
> implementation work, but if pair #4 ships with the W=3 headline, it
> sets a precedent for future phase pivots to graduate on prior-phase
> metrics — exactly the discipline drift CLAUDE.md was tightened to
> prevent.

The agent also flagged biological-replay consensus is **prediction-error
weighted, not metastability-weighted** (Yang/Buzsáki Science 2024;
van der Meer/Bendor 2025 TINS review). If metastability ends up being
inert on this substrate, the cleaner pivot is to surprise/PE-weighted
replay (UPER arXiv:2506.09270, ReaPER arXiv:2506.18482).

See [research/metastability-replay.md](research/metastability-replay.md).

---

### 🟡 FINDING 5 — "Architectural finding as primary contribution" has a publication norm now, and the substrate-saturation result fits the shape

The FEP/closure agent surfaced direct precedents for the closure framing:

- **NeurIPS 2025 launched a Position Paper Track** specifically for
  contributions of the shape "we tried this hard, here is the principle"
- **"Embracing Negative Results in ML"** (Karl et al., arXiv:2406.03980)
  was an **ICML 2024 oral**
- **"Modern Hopfield Networks Require Chain-of-Thought to Solve NC^1-Hard
  Problems"** (arXiv:2412.05562) is the direct templating analog: "this
  substrate provably can't do this without this extension"
- **"Practical Lessons on VSAs"** (Carzaniga et al., NeSy 2025) — four
  lessons, three of them negative results

Three framing modes ordered by prestige: **theorem-style** ("here is the
bound"), **mechanism-explanation** ("here is why"), **empirical-lessons**
("here is what we learned"). The recommended framing for Neuro-AI's
six-instance substrate-saturation finding is **mechanism-explanation**,
with the geometric mechanism being:

> *Sharp self-retrieving basins (necessary for capacity at high N) collapse
> the intermediate-state geometry required for role-binding-prior
> traversal. The two requirements pull in opposite directions on the
> Hopfield substrate at moderate D.*

**Surprise finding:** arXiv:2510.17593 ("Paradoxical increase of capacity
due to spurious overlaps in attractor networks," 2025) reframes a classical
defect as a *useful capacity-amplifying mechanism*. **This is the exact
rhetorical move available to Neuro-AI** — frame the substrate-saturation
finding as the principle behind why a *different* mechanism class
(diagnostic-actuator dynamics, e.g. pair #4) is the right Phase 5 shape
rather than the energy-margin K-branch the design spec originally
proposed.

> **Why this matters:** if Path B (close + pivot) is chosen, the closure
> note should be drafted to publication-shape from the start. The
> precedents above tell us this is a publishable result, not just
> archival documentation.

See [research/fep-and-closure.md](research/fep-and-closure.md) — also
covers Sub-angle A: Active Inference / Expected Free Energy as a Phase 5
reformulation candidate (arXiv:2504.14898 EFE Planning as Variational
Inference, arXiv:2505.19867 Deep Active Inference, Sophisticated Inference
arXiv:2006.04120). EFE-scored branches would discriminate via
KL-divergence on the variational posterior, not via raw energy gap.

---

## Concrete Ideas and Approaches

Organized by cost-to-information, with the new findings woven in.

### Tier 0 — Do BEFORE choosing Path A or Path B (1 hour total)

#### Idea 0.1: Verify the magnitude-floor derivation
**What:** Read `phase-5-unified-design.md` ~lines 256–281 and reconcile
against the dimensionality-agent's re-derivation in
[research/dimensionality-scaling.md](research/dimensionality-scaling.md).
Confirm whether D-sensitivity is ~1.3× or ~8× across 16× D range.
**Why:** the entire Path A premise rests on this. 30 min of math.
**How:** rederive symbolically; compare both forms; if discrepancy, find
the missing term or the bug.

#### Idea 0.2: Compute the Fisher separation index of A1' surviving atoms
**What:** ~30-line computation. Single Python cell on the existing seed-17
snapshot. Per Varner 2026, S > 0.3 ⇒ basins well-separated and softmax
log-prior bias fixes K-branch; S < 0.2 ⇒ basin geometry truly collapsed.
**Why:** the cheapest signal for "is K-branch fixable with a one-line
softmax bias?" — Finding 2.
**How:** `S = (μ_pos − μ_neg)² / (σ_pos² + σ_neg²)` over within-vs-between
pattern inner products.

#### Idea 0.3: Audit pair #4's headline metric against the design spec
**What:** Confirm whether pair #4's "Δ meta-stable-rate at W=3" headline
is on the Phase 5 design-spec headline (ΔE per phase-5-unified-design.md:
256-281) or is silently a Phase 4 metric. If the latter, edit the design
note before implementing.
**Why:** discipline drift; Finding 4a.
**How:** read the design note, the Phase 5 spec line 256-281, and decide
whether to (a) graduate on ΔE_via_metastability or (b) acknowledge pair #4
as Phase 5 closure + Phase 4 D1 strengthening rather than a Phase 5
graduation.

---

### Tier 1 — Cheap discriminating experiments (~1 day each)

#### Idea 1.1: TPR at D=64 (zero-crosstalk operator test)
**What:** Reimplement role-binding via Tensor Product Representation
(role ⊗ filler matrix; D_role = D_filler = 64; representation dim 4096
matches current memory footprint). Re-run Phase 5 K-branch experiment on
existing schema store, same K=4, β=10, γ=0.5.
**Why:** discriminates "operator crosstalk" vs "substrate-architecture"
as the binding limit. Finding 3.
**Anti-homunculus:** operator change, not arbitration; PASS by inspection.
**Source:** [research/binding-operators.md](research/binding-operators.md)
§TPR section.

#### Idea 1.2: Log-prior softmax bias on K-branch settling (Varner 2026)
**What:** Add `b_k = log r_k` to the per-branch settling logits where r_k
is the schema's prior weight. One-line change. Verify with same seed-17
n=200 cue set.
**Why:** if Fisher index from Idea 0.2 is > 0.3, this is the predicted
fix. Cheap. arXiv:2603.20115 is the paper.
**Anti-homunculus:** local bias in the energy expression, not a
controller; PASS.

#### Idea 1.3: EFE-proxy diagnostic on K=4 branches
**What:** Compute a 1-day Expected-Free-Energy proxy per branch
(`EFE_k = -⟨log p_k(q*)⟩ + KL[q* || prior]`) and re-weight bundling by
EFE instead of by `-E/τ`. Does it stratify the branches?
**Why:** if energy-margin doesn't stratify but EFE does, Phase 5 is
score-function-level saturated, not substrate-level. arXiv:2504.14898
sketches the EFE-as-VI formulation.
**Source:** [research/fep-and-closure.md](research/fep-and-closure.md)
§Sub-angle A.

#### Idea 1.4: Oracle-priority control for pair #4
**What:** Run pair #4 with `m_i` replaced by a synthetic oracle priority
(e.g., set `m_i = 1` for the 5 highest-eff-strength atoms, 0 for the
rest). Does the headline move at all? If not, the issue is downstream of
priority weighting and Path B doesn't help.
**Why:** cheap falsifier for the whole Path B premise. Metastability-
agent's recommendation.
**Source:** [research/metastability-replay.md](research/metastability-replay.md).

---

### Tier 2 — Substrate-level moves (1–3 days)

#### Idea 2.1: Input-Driven Plasticity Hopfield (Betteti 2025)
**What:** Replace `W = Σ ξ_μ ξ_μ^T` with `W(u) = Σ α_μ(u) ξ_μ ξ_μ^T`
where u is the prior (role-prior or content-prior). Closed-form `α_μ`.
**Why:** substrate-pure mechanism for prior-traversal without softening
basins. Finding 2. arXiv:2411.05849, Sci. Adv. 2025.
**Anti-homunculus:** `α_μ(u)` is a local function of input; PASS.

#### Idea 2.2: Pair #4 with corrected design (Finding 4a + 4b)
**What:** Implement pair #4 with (a) headline = ΔE_per_atom (not W=3),
(b) signal renamed to "hindsight regret" with Liu 2021 citation, (c)
oracle-priority sanity check (Idea 1.4) before n=10.
**Why:** Path B is still viable, but make it pay off Phase 5's actual
design spec, not pivot the headline silently.

#### Idea 2.3: HLB iterative-settling stability (Alam 2024)
**What:** Replace FHRR circular convolution with Holographic Linear
Binding (arXiv:2410.22669). Same `1/√D` retrieval-step noise floor BUT
stable norm `||·||₂ = √D` across iterations — addresses the regime where
Hopfield settling compounds noise step-by-step.
**Why:** Hopfield-loop noise compounding may be a hidden contributor to
the saturation. HLB is a drop-in operator change at unchanged D.

---

### Tier 3 — Reframes / bigger redesigns (3+ days)

#### Idea 3.1: Predictive Coding Associative Memory substrate
**What:** Migrate substrate from Modern Hopfield to predictive-coding
associative memory (Salvatori NeurIPS 2021; BayesPCN NeurIPS 2022; Tang
arXiv:2509.01987 Sept 2025). PC-AM has explicit latents → EFE scoring is
natural; empirically exceeds Modern Hopfield capacity and degrades
gracefully.
**Why:** if both substrate-pure interventions (Idea 2.1, 2.3) fail, the
substrate family itself may be the constraint. PC-AM is the
next-most-precedented option.
**Risk:** larger rewrite; touches Phase 3 and Phase 4 too.

#### Idea 3.2: Dynamic Manifold Hopfield (Li 2025)
**What:** Learned context-dependent manifolds give 64% accuracy at 2N
patterns in N neurons (vs Modern Hopfield's 13%). Capacity + prior-
discriminable basins jointly achievable.
**Why:** highest-payoff substrate redesign with published validation.
arXiv:2506.01303.
**Risk:** requires training (not substrate-pure in the "no learned
parameters" sense); careful anti-homunculus audit needed.

#### Idea 3.3: Close Phase 5 + position as architectural finding (Finding 5)
**What:** Close Phase 5 as graduation-unattained; write the
substrate-saturation finding as a mechanism-explanation paper. Pair #4
ships as Phase 5'/closure-and-pivot rather than as Phase 5 graduation.
Target NeurIPS 2025 Position Paper Track or similar.
**Why:** the finding is genuine, novel, and publishable. arXiv:2510.17593
demonstrates the rhetorical move (defect-as-feature). The
substrate-saturation principle becomes load-bearing for whatever Phase 5'
ends up being.
**Risk:** locks in the "we tried X, it doesn't work" narrative before
exhausting Tier 0/1 cheap discriminators. Should NOT be done before
Ideas 0.1, 0.2, 1.1.

---

## Cross-Cutting Themes

**Theme A: Operator vs substrate vs metric.** The substrate-saturation
finding has been interpreted as a *substrate* property, but the six
instances span all three layers: 1, 2, 3 are substrate-side; 4 is
mechanism-side (branching utility); 5 is measurement-side (magnitude-to-
noise); 6 is measurement-vs-substrate (basin-hit decouples from ΔE).
**Discriminating which layer the binding constraint actually lives at
should be done before committing weeks to lower-D substrate retraining.**
Ideas 0.1–0.2 and 1.1–1.3 each isolate one layer.

**Theme B: "Diagnostic as actuator" is well-precedented and pair #4 is on
the right path.** UPER, ReaPER, SuRe (2024–2025) all do signal-modulates-
rate continuous priority with anti-homunculus discipline. Pair #4 belongs
in this literature. The metric-discipline issue (Finding 4a) and the
"hindsight regret" lit anchor (Finding 4b) are improvements, not
refutations.

**Theme C: The K-branch mechanism's "branches don't stratify" may be
β-saturation more than basin-collapse.** Langevin Stochastic Attention
(arXiv:2603.06875) shows single-chain diversity at β=200 exceeding
multi-chain diversity at β=2000. The current β=10 may already be past the
β-saturation knee for this substrate, making K=4 chains all converge to
the same attractor for reasons that are score-function-related, not
basin-related. EFE-proxy (Idea 1.3) tests this.

**Theme D: There's a unifying cross-phase pattern — every phase's binding
constraint has been substrate-pure-metric vs readout-metric mismatch.**
Phase 3's top1 regression is a Hebbian-codebook-reshaping property
detected via readout (R@10, top1) on a substrate that wasn't designed for
those readouts. Phase 4's D1 graduation succeeded by adopting a substrate-
pure metric (Δms_w3). Phase 5's ΔE was substrate-pure but turned out to
collapse on a sharp-basin substrate. **The architectural carry-forward is
that substrate-pure metrics are necessary but not sufficient — they must
also measure something the substrate exposes at non-noise magnitude.**
This is the kind of principle the closure paper (Idea 3.3) would state
crisply.

---

## Challenges and Counterarguments

**On Finding 1 (the floor formula):** the agent's re-derivation might
itself be wrong; the design-spec formula might have a term the agent
missed. **Run Idea 0.1 before treating Finding 1 as binding.** If the
agent is right, the strategic decision tree changes; if the spec is
right, the agent's recommendations downgrade.

**On Finding 2 (IDP / log-prior bias):** these fixes target the K-branch
mechanism, not the underlying substrate-saturation finding. If the
ΔE-near-noise-floor result is a deeper property than "K-branch doesn't
stratify," IDP and log-prior bias may close *one* of the six instances
(2 and 4) without addressing the rest. **The Fisher-index measurement
(Idea 0.2) is the only cheap way to know.**

**On Finding 3 (TPR test):** TPR at D=64 changes the role × filler space
size; the schema-store at 1064 atoms may overflow D²=4096 capacity under
TPR's stricter capacity formula (D_role × D_filler). May need D=128 for
capacity headroom, which is 16384 dims — larger memory than current FHRR.

**On Finding 4 (pair #4 metric drift):** the W=3 metric IS substrate-pure
and was the original Phase 4 graduation gate. Arguing for ΔE-per-atom
adds a constraint to a not-yet-shipped mechanism. The right answer
depends on whether pair #4 is best understood as Phase 5 graduation
(must hit Phase 5 spec) or Phase 5'/closure-pivot (free to repurpose the
W=3 success).

**On Finding 5 (publication as closure):** writing the paper takes weeks
that could be spent on Tier 1 experiments. The closure framing is
strongest AFTER Tier 1 has ruled out the operator-level and substrate-
manipulation fixes. Premature closure narrows the contribution.

**On all findings:** none of these refute the substrate-saturation
finding. They suggest cheap experiments that might *change the diagnosis*
of WHY it happens, which would let the user choose between fixing it
(Idea 2.1 / 2.3 / 3.1 / 3.2) and documenting it (Idea 3.3) with better
information.

---

## Rabbit Holes Worth Following

1. **GHRR's non-commutative role/content asymmetry**
   (Yeung/Zou/Imani 2024, arXiv:2405.09689). Under-cited in VSA lit.
   May give intrinsic role-prior/content-prior asymmetry "for free"
   without operator surgery. Worth a 2-paragraph read before deciding
   between FHRR and TPR.

2. **Dead Neurons (Fanaskov & Oseledets ICLR 2025, arXiv:2410.13866).**
   "Alarmingly on-the-nose" for Phase 4's mass-death and d_eff=5
   pathology. Proposes a modified Lyapunov function preserving
   steady-state structure without flat directions. Could be the principled
   replacement for the A+B+A1' machinery.

3. **Active Inference Tree Search (Maisto et al. Neurocomputing 2024)**
   for if Phase 5 reformulates as variational. The closest published
   analog to K_main + 1 surprise-branch with a principled scoring rule.

4. **"Paradoxical increase of capacity due to spurious overlaps"**
   (arXiv:2510.17593, 2025). Direct template for reframing
   substrate-saturation as a useful principle rather than a failure.

5. **PAM/co-occurrence predictor for the role codebook** (deferred from
   2026-05-20 brainstorm). The metastability-replay agent reinforces that
   prediction-error / surprise is the cleaner signal class. If pair #4
   stalls, PAM-style predictor distance is the next move in the same
   shape.

6. **A direct read of `phase-5-unified-design.md` lines 256-281** by the
   user, to confirm the floor formula and the headline definition match
   the agent's re-derivation. 5 minutes. Pre-condition to acting on
   anything in this brainstorm.

---

## Recommended Next Session Sequence

Given that the user explicitly asked "make sure we are not missing
anything before we move forward," the strongest sequence is:

1. **Tier 0 trio** (≤2 hours): verify the floor formula (0.1), compute
   Fisher separation (0.2), audit pair #4 headline (0.3).
2. **Decision gate based on Tier 0 results:**
   - If floor formula confirms D-sensitivity is weak: drop Path A as
     framed; reframe option 1 as "operator redesign + substrate
     redesign" rather than just D-sweep.
   - If Fisher S > 0.3: **Tier 1 Idea 1.2 (log-prior softmax bias) is
     the new top recommendation** — one-line fix; if it works, the
     whole strategic decision evaporates.
   - If Fisher S < 0.2: substrate geometry truly collapsed; go to
     Tier 2 (IDP / pair #4 with corrected design) before Tier 3.
3. **Tier 1 spike** of whichever experiment Tier 0 picks (one day,
   single seed).
4. **Strategic decision** with much better information than this session
   started with.

The headline of this brainstorm: **don't commit to Path A or Path B
before running the Tier 0 trio.** Each of those three checks is under
2 hours and they collectively change the answer to "are we missing
anything?"

---

## Sources

### Top-line papers (one or two will be load-bearing)
- Betteti et al. 2025 — Input-Driven Plasticity Hopfield —
  arXiv:2411.05849 — Sci. Adv. 2025 — https://arxiv.org/abs/2411.05849
- Varner 2026 — Pattern Multiplicity — arXiv:2603.20115 —
  https://arxiv.org/abs/2603.20115
- Liu et al. 2021 — ReMERN/ReMERT (hindsight regret) —
  arXiv:2105.07253 — NeurIPS 2021 — https://arxiv.org/abs/2105.07253
- Alswaidan & Varner 2026 — Langevin Stochastic Attention —
  arXiv:2603.06875 — https://arxiv.org/abs/2603.06875
- Hoover et al. 2025 — LSR / Epanechnikov Energy — arXiv:2506.10801 —
  https://arxiv.org/abs/2506.10801
- Li et al. 2025 — Dynamic Manifold Hopfield — arXiv:2506.01303 —
  https://arxiv.org/abs/2506.01303
- Fanaskov & Oseledets 2025 — Dead Neurons — arXiv:2410.13866 —
  ICLR 2025 — https://arxiv.org/abs/2410.13866

### Operator literature
- Recchia et al. 2015 — Permutation > convolution at D=2048
- Plate 1995 — HRR foundations
- Yeung/Zou/Imani 2024 — GHRR matrix-phase — arXiv:2405.09689
- Alam et al. 2024 — HLB stable-norm iterative settling —
  arXiv:2410.22669 — NeurIPS 2024
- Frady/Kleyko/Sommer 2021 — Sparse VSA

### Pair #4 / replay grounding
- UPER — arXiv:2506.09270 — RLC 2025
- ReaPER — arXiv:2506.18482 — 2025
- SuRe surprise-weighted — arXiv:2511.22367 — late 2025
- Aljundi MIR 2019 + 2024 follow-ups
- Yang/Buzsáki Science 2024 SWR experience-selection
- van der Meer/Bendor 2025 TINS review (biological replay)

### FEP / closure
- Friston 2021 — Sophisticated Inference — arXiv:2006.04120
- Maisto et al. 2024 — Active Inference Tree Search — Neurocomputing
- EFE Planning as Variational Inference — arXiv:2504.14898 (Apr 2025)
- Deep Active Inference for Long-Horizon — arXiv:2505.19867 (May 2025)
- Salvatori et al. NeurIPS 2021 — Predictive Coding Associative Memory
- BayesPCN — NeurIPS 2022
- Tang et al. — arXiv:2509.01987 (Sept 2025) — PC-AM recent
- Karl et al. 2024 — Embracing Negative Results in ML — arXiv:2406.03980 — ICML 2024 oral
- Hu/Lin/Song/Liu — Modern Hopfield Networks Require CoT for NC^1 — arXiv:2412.05562 (Dec 2024)
- Carzaniga et al. NeSy 2025 — Practical Lessons on VSAs — PMLR 284:218-236
- "Paradoxical increase of capacity" — arXiv:2510.17593 (2025)
- Pinnacle Sharpness — arXiv:2511.13053 (2025)
- Capacity-under-data-manifold — arXiv:2503.09518

### Full source lists per angle
- [research/binding-operators.md](research/binding-operators.md)
- [research/basin-engineering.md](research/basin-engineering.md)
- [research/dimensionality-scaling.md](research/dimensionality-scaling.md)
- [research/metastability-replay.md](research/metastability-replay.md)
- [research/fep-and-closure.md](research/fep-and-closure.md)

### Project context summaries
- [context/phase5-current.md](context/phase5-current.md)
- [context/prior-phases.md](context/prior-phases.md)
- [context/research-base.md](context/research-base.md)
