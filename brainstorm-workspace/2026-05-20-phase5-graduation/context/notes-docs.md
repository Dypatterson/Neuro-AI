# Phase 5 Graduation Brainstorm — Notes & Docs Context

Compiled 2026-05-20 to seed a brainstorm session on the four-step debugging
journey (A+B → A1 → A1' → β) ending in the empirical failure of β
(role-fidelity-weighted prior, `f_i` zero-variance at D=4096). The goal
of this document is to surface implicit assumptions the four design notes
share, so the brainstorm can search outside the A1''/β'/re-scope triangle.

Files scanned:
- `/Users/dypatterson/Desktop/Neuro-AI/docs/PROJECT_PLAN.md`
- `/Users/dypatterson/Desktop/Neuro-AI/STATUS.md`
- `/Users/dypatterson/Desktop/Neuro-AI/notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md`
- `/Users/dypatterson/Desktop/Neuro-AI/notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md`  (A+B)
- `/Users/dypatterson/Desktop/Neuro-AI/notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md`  (A1)
- `/Users/dypatterson/Desktop/Neuro-AI/notes/notes/2026-05-20-r-inst-measure-dynamic-form.md`  (A1')
- `/Users/dypatterson/Desktop/Neuro-AI/notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md`  (β)
- `/Users/dypatterson/Desktop/Neuro-AI/notes/emergent-codebook/phase-5-unified-design.md`
- `/Users/dypatterson/Desktop/Neuro-AI/notes/emergent-codebook/phase-5-checklist.md`

---

## 1. The anti-homunculus rule — exact binding language

The rule is the binding constraint on every proposed fix. It has been
upgraded three times during the four-step journey; each upgrade names a new
"failure mode" that the prior framing missed.

### Canonical form (2026-05-09, foundational paper synthesis)

> Every proposed addition to the architecture must either be a local
> geometric dynamic or be expressible as a measurement of one — never
> an arbitration over them. If a candidate mechanism cannot pass that
> test, it gets rejected or reframed.

Concretely (`CLAUDE.md`):

- No supervisor module decides which subsystem wins.
- No `if X then do Y` rule that reads a metric and triggers a response.
- "Apparent decisions" are local geometry, energy, settling, tension, or
  consolidation dynamics.

### The "two readings of the same arrows" reframe (2026-05-09)

This is the technical move that lets diagnostic-actuator pairs cross
the threshold without becoming controllers:

> An actuator is a slow-timescale dynamic that some diagnostic happens
> to be a fast-timescale snapshot of. Diagnostic and actuator are the
> same physical process viewed at different temporal resolutions.

Standard reading: "condition X causes response Y" → requires a thing
that detects X and triggers Y → controller in disguise.

Non-controller reading: X and Y are the **same physical event at
different timescales**. The "diagnostic" measures where the process is
right now; the "actuator" is the process running.

This grammar is load-bearing — it is the *only* language the project
permits for promoting any of the five 2026-05-09 diagnostic-actuator
pairs into mechanism.

### Three failure modes the journey has named

The 2026-05-09 note named one; the 2026-05-20 journey extended to three:

1. **Mode 1 (2026-05-09):** A controller reads a metric and triggers a
   discrete action. *Closed by A+B (continuous coverage-weighted
   reinforcement + repulsion energy term).*
2. **Mode 2 (A1 design note):** An implementer hard-codes a constant
   where a measurement belongs. *Closed by A1 (substrate-derived
   `r_ema` init via existing geometric helper).*
3. **Mode 3 (A1' design note):** An implementer picks the **wrong
   reduction** of a measurement, encoding an architectural claim the
   substrate's geometry doesn't support. *Closed by A1' (max-over-others
   reduction instead of mean-RMS).*

> **Brainstorm tension:** the journey has been *generating* new failure
> modes faster than it closes them. Each step closes one and reveals
> the next layer. β's empirical failure is what mode 4 would look
> like if it were named — it isn't yet.

### What the rule explicitly does *not* require

The 2026-05-09 framing of "local geometric dynamic" doesn't distinguish
smooth from piecewise-smooth (A1' note §"Note on piecewise-smoothness").
It also does not require derivability from first principles — only
that the value at any moment is a measurement or evolution of the
substrate, not an implementer's choice.

---

## 2. The five diagnostic-actuator pairs — closure ledger

The 2026-05-09 note named these as the "next major thresholds." Status
after the four-step journey:

| Pair | Status | Closing mechanism |
|---|---|---|
| high drift ~ replay pressure | **Open** | (no design note) |
| high spread ~ reduced consolidation | **Closed by A+B** | -α·log(d_eff) repulsion as substrate energy term |
| bimodality ~ splitting pressure | **Attempted by β; empirically degenerate** | per-schema continuous role-fidelity-weighted prior — but `f_i` has zero variance at D=4096 |
| metastability ~ replay prioritization | **Open** | (no design note) |
| low cap-coverage ~ restructuring pressure | **Open** | (no design note) |

> **Brainstorm tension:** Three of five pairs are entirely unattempted.
> The journey has been concentrating on one pair (spread/consolidation)
> via A+B and its corrections, and one (bimodality/splitting) via β.
> The drift/replay-pressure, metastability/replay-prioritization, and
> cap-coverage/restructuring pairs are all sitting open with no
> attempted design — they may be cheaper to close, and any of them
> might shift the substrate in a way that *changes* the f_i geometry
> downstream.

---

## 3. The four 2026-05-20 design notes — load-bearing implicit assumptions

These are the implicit success criteria the brainstorm should explicitly
question.

### Shared assumptions across all four notes

1. **The substrate is the right object to fix.** Every note assumes the
   problem lives at the substrate / consolidation / measurement layer.
   *None* of the notes considers fixing the cue, the readout, or the
   binding scheme itself. The cue regime is a *test distribution* per γ
   (path 3); the binding/role-vector scheme is a fixed inheritance.

2. **Phase 5 is wrapping, not modifying, the Phase 4 substrate**
   (`phase-5-unified-design.md` §"Integration with Phase 4 substrate").
   This is *load-bearing*: it preserves Phase 4 D1 graduation as a
   non-regression constraint and means *any* Phase 5 fix that touches
   Phase 4 must clear D1. **Implicit assumption:** the Phase 4 substrate
   is sound; Phase 5 can wrap it.

3. **D=4096 is fixed.** Every note treats D as a constant. The β empirical
   failure surfaced that at D=4096 the FHRR unbind crosstalk noise
   structurally dominates any pairwise role-fidelity signal
   (`f_i ≈ 1 − 1/√D ≈ 0.984` independent of pattern content). *No design
   note questions whether D=4096 is the right substrate scale.*

4. **FHRR is the right algebra.** Bind/unbind via complex
   multiplication/conjugation is unchallenged in every note. The
   role-fidelity measure `mean(1 - |G_jk|)` collapsing to a substrate
   constant is treated as a property *of the measurement*, not a
   property of FHRR's crosstalk floor at this D.

5. **The headline must be substrate-pure (energy-based, not readout).**
   This was binding from the 2026-05-16 discipline note. The β note
   *almost* changes the headline (q-sweep monotonicity + fidelity-
   weighted ΔE) but is careful to keep it substrate-pure. *The
   discipline says: do not switch back to readout headlines if the
   substrate-pure one keeps failing — that would be H1 / cherry-pick.*

6. **Anti-homunculus filter is local, atom-by-atom.** Every note
   evaluates the anti-homunculus check per-atom: is each atom's
   update / init / measurement local to its own context? *Implicit
   assumption:* the filter cannot be passed by a mechanism that is
   global-but-non-arbitrary (e.g., a substrate-wide gradient flow
   whose result is locally read). A+B does have a global term
   (`-α log d_eff`) but is *cast* as a per-atom force.

### Per-note implicit assumptions

#### A+B (death-dynamic note)

- **Assumes the binary death mechanism is what produced the d_eff
  collapse.** True at the geometric level (report 044 shows the
  collapse coincides with the death event). But d_eff might also be
  collapsible from upstream causes — Hebbian-codebook reshaping under
  drift (STATUS blocker #6′) is a separately-named substrate-shaping
  force.
- **Assumes `r_i` (redundancy) as the right per-atom geometric
  invariant.** The formal definition uses projection magnitude onto
  the complement substrate's column-span. This is the FHRR-natural
  reading, but it presumes redundancy *is* the thing throttling
  reinforcement. Alternative invariants (k-NN d_eff per atom, local
  basin curvature, retrieval entropy at i) are not considered.
- **Assumes `H_anti = -α log d_eff` as the right anti-collapse
  energy.** The functional form is "log of participation ratio" —
  motivated by it being differentiable and bounded. Alternative
  forms (a quadratic penalty on spectral concentration, entropy of
  eigenvalue distribution, KL to a target spectrum) are not
  considered.
- **Pre-commits α and λ from theoretical considerations** (the H4
  binding). This is anti-tuning discipline but also locks the design
  to whatever the first calibration produces.

#### A1 (r_ema init note)

- **Assumes new atoms should *measure* their redundancy at add-time.**
  The design rejects W1/W2/W3 (controllers in disguise) but never
  considers: should new atoms even *enter* with full strength? What
  if the discovery channel itself is the wrong mechanism on a
  low-d_eff substrate? *The discovery channel is treated as a
  constraint, not a candidate variable.*
- **Treats `add_pattern` as an event in the discovery channel's
  dynamics, not a wall-clock trigger.** This is the anti-homunculus
  cover for measuring at add-time. But that framing depends on the
  discovery channel itself being a clean dynamic, which has not been
  audited at the same depth.

#### A1' (r_inst measure note)

- **Assumes `max_{j≠i} |G_ij|` is the correct stronger reduction.**
  The argument is: max-saturates-at-1-for-any-duplicate. But max
  loses partial-redundancy signal (an atom 50% similar to 100
  neighbors gets the same score as 50% similar to 1 neighbor).
  Smoothed p-norm (Candidate β) and streaming low-rank SVD
  (Candidate γ) are listed as fallbacks but considered higher cost.
- **Assumes the measurement layer is the binding failure.** The note
  *notices* (and STATUS confirms) that after A1' the top-K selector
  picks from 7 FP-identical discovery atoms tied at strength 0.029
  — i.e., A1' migrated the failure to the **selector layer**, not
  solved it. The note's framing keeps the selector layer out of
  scope; β was then proposed to displace the selector.

#### β (cue-regime / role-prior note)

- **Assumes per-schema role-fidelity is a measurable substrate
  property at D=4096.** The empirical failure (`f_i = 0.9858, std =
  0.0000` across 1064 atoms) falsifies this for the chosen
  operationalization `f_i = mean_{r1,r2} (1 - |G(unbind(s, r1),
  unbind(s, r2))|)`. STATUS-5 already calls out the substrate-
  encoding-level structural cause: FHRR unbind crosstalk noise
  dominates the signal.
- **Assumes the prior shape — `prior = Σ_i (cue·s_i)^p · f_i^q · s_i`
  — is the right place to insert role-fidelity weighting.** The
  weighting is multiplicative on the existing similarity prior. If
  `f_i` is uniform, the weight cancels; that is by construction.
- **Assumes the cue regime is the *only* readout-coupled axis worth
  averaging over** (γ companion). Other distributions (over D, over
  W, over the role-set, over schema-store size) are not considered.
- **Reframes Phase 5's headline** by introducing the q-sweep + fidelity-
  weighted ΔE. The note acknowledges this is a discipline cost (the
  2026-05-09 headline principle warns against changing headlines
  mid-flight) but defends it via substrate-purity.

> **Cross-note implicit assumption:** the four notes' candidate space
> is generated by *peeling layers of A's lifecycle* (substrate
> energy → init condition → measurement reduction → downstream prior).
> Each layer is assumed to be the right next layer to fix. **No note
> considers a mechanism that *isn't* in A's lifecycle** — e.g.,
> changing what gets retrieved in the first place, changing the
> discovery channel itself, changing the role-binding algebra.

---

## 4. Phase 5 checklist — pre-committed falsification criteria

The checklist binds graduation. Status as of late 2026-05-20:

### Headline (A1) — *still binds*

> Δ final-state energy E_A − E_B > 0 with 95% CI disjoint from zero,
> n_seeds ≥ 10.

Current: partial at n=5, ΔE = −1.5e-4, CI includes 0; LOSO never
excludes zero. **Has not been re-attempted** since A1'/β; the n=5/n=10
graduation runs are the gates A+B+A1+A1'+β are supposed to clear.

### Section B — Required controls

| Item | Status | Falsification implication |
|---|---|---|
| B1 random-schema branches | partial | not paired against role_K4 at n=5; cannot rule out artifact |
| B2 K=1 single-branch | **⚠️ FIRES at n=5** | K1 ΔE = +1.3e-4 (5/5 positive, CI excludes 0) ≥ K4 ΔE = −1.5e-4 → **branching is gratuitous (even destructive) on post-death substrates** |
| B3 No-prior (γ=0) | ✅ | sanity passed |
| B4 No-schema-store | open | not run |

> **B2 has fired and is not yet retracted.** Single-branch retrieval
> with role-prior is *better than* K-branch with role-prior on the
> 6-12-atom post-death substrate. This is a major design-assumption
> falsification: branching may not be the right structural matcher
> at all on the current substrate scale.

### Sections C–G — gating rules

- **C (schema-source robustness):** must pass C1 AND at least one of
  {C2, C4}. Only C1 has been run.
- **D (seed robustness):** n ≥ 10 with LOSO CI excludes zero for
  every leave-one-out subset. Failed at n=5.
- **G1 (substrate-pure tiebreaker):** Δ meta-stable rate at W=3 under
  role-prior must not regress. Not yet measured.

### Section H — explicit anti-homunculus prohibitions (binding)

| # | Prohibition |
|---|---|
| H1 | Schema-source ablation is diagnostic-only. **No adaptive switching.** |
| H2 | Atom-splitting diagnostic logged; split *action* deferred. |
| H3 | Branch selection is energy-based only. **No metric-based selection.** |
| H4 | No "if ΔE fails on C1 but passes on C2, declare graduation" — graduation is structural, not best-of-N over conditions. |

> **H1 + H4 are very tight.** They forbid the most obvious "engineering
> recoveries" — try a different schema source, try a different cue
> regime, pick the configuration that passes. They are also what makes
> the bimodal-ΔE-across-seeds problem actually hard: you can't pick
> the seeds that worked.

### Falsifications that *have* fired

1. **Decision 5 spike (per-pattern vs global pull):** global pull
   was empirically falsified; per-pattern wins on real W=4 substrate.
   This is *the only Phase 5 decision so far that has cleanly cleared
   a pre-committed falsifier.*
2. **B2 (K=1 vs K-branch):** at n=5 on post-death substrate, branching
   is destructive. Has not been retracted by subsequent runs.
3. **A1 partial fix:** criteria #1 and #2 FAIL but mechanism
   progressed; identified A1' as the next layer.
4. **A1' partial fix:** criteria #1 and #2 still FAIL but mechanism
   correct at substrate level; failure migrated to selector layer.
5. **β at n=1:** uniform `f_i = 0.9858` across all 1064 atoms; the
   role-fidelity measure has structurally-zero variance at D=4096.

### Pre-committed falsifiers that *have not yet been tested*

- A+B mechanism-validity at **n=5** (only n=1 pilot has run).
- A1+A1' top-K diversity criterion #1 at **n=5**.
- A1+A1'+β at **n=5** Colab confirmation of uniform f_i.
- The actual Phase 5 graduation headline (A1, ΔE > 0 CI-disjoint, n≥10)
  against the A+B+A1+A1' substrate.

---

## 5. Tensions, unresolved decisions, threads the user has been circling

### Tension 1 — Substrate scale vs effect magnitude

The user has *already noted* (Decision 5 closure rationale, phase-5-unified-design.md:519-531):

> The per-pattern ΔE is microscopic (+2.6e-5) because at β=10 with
> only 12 attractors, Hopfield settling is highly decisive — most cues
> land at the global minimum regardless of prior. The 98% directionality
> is the substrate-bounded signal of structural retrieval working.
> **Larger N_schemas should yield larger magnitude. This is a
> substrate-capacity observation, not a Phase 5 mechanism limitation.**

This is a major thread that hasn't been pulled. The substrate is 12
post-death atoms (or up to 1064 with discovery channel). β=10 produces
near-decisive settling at this attractor count. Yet:

- A+B is designed to *preserve* d_eff at 25–50.
- A1+A1' fixes the discovery channel's strength dominance but doesn't
  change the *count* of effective attractors.
- β assumes 1064 distinct schemas exist but `f_i = 0.9858` shows they
  are not distinguishable at D=4096.

> **Hidden thread:** the substrate may not have enough *distinct
> meaningful attractors* for any role-vs-content discrimination to
> emerge as a substantial energy gap. The fixes have all been to the
> mechanism; none has been to the substrate's *attractor density*.

### Tension 2 — D=4096 vs FHRR crosstalk floor

The β failure surfaces a substrate-encoding fact: at D=4096, FHRR's
unbind crosstalk gives `1 - 1/√D ≈ 0.984` as the floor for similarity
between unbound fillers. This is *independent* of pattern content.

The four notes treat D as fixed. PROJECT_PLAN.md treats D=4096 as the
Phase 1 scale target. But the crosstalk floor *is* a property of
(D, W, codebook), and the substrate has been tuned at one corner of
that space.

> **Implicit thread the brainstorm should pull:** is the design fighting
> a wall that's a property of (FHRR, D=4096, W=4) jointly? Increasing D
> reduces the crosstalk floor (`1 - 1/√D → 1`); changing W changes the
> binding tensor depth; switching encoders away from FHRR removes the
> crosstalk-floor constraint entirely.

### Tension 3 — Headline metric stability vs reformulation

The 2026-05-09 headline principle is binding:

> Pick one metric as the headline signal that defines whether the
> phase crossed its viability threshold; treat the others as
> conditional drill-downs.

The β note proposes a successor headline (q-sweep monotonicity +
fidelity-weighted ΔE). The 2026-05-09 note explicitly warns against
this. The user's STATUS-5 entry ends with the open question:

> Recommended next decision (user's call): (1) stop chasing K-branch
> state_divergence and re-scope Phase 5's headline metric; (2)
> reformulate f_i (β': codebook-correlation instead of pairwise-
> distance); (3) investigate lower-dimension substrates (research
> direction); (4) defer to Colab n=5 to confirm the uniform-f_i
> finding generalizes.

Option (1) is "concede the current headline." Option (2) is β'.
Option (3) directly opens the D-axis. Option (4) is a falsification-
discipline move. **The user is circling because there is no clean
path that doesn't either (a) violate H1/H4 cherry-pick discipline,
(b) reformulate the headline mid-flight, or (c) admit Phase 5 doesn't
graduate at this substrate.**

### Tension 4 — B2 firing means branching may not be the mechanism

B2 (K=1 single-branch ≥ K-branch on post-death) is a falsifier that
*has fired and has not been retracted*. The entire Phase 5 architecture
is "energy-guided structural branching." If K=1 is at least as good
as K=4 on the actual substrate, the design's central claim — that
branching is the structural matcher — is under empirical pressure.

> **The four notes do not address B2.** A+B fixes d_eff so branches
> *could* separate; A1+A1' fixes which schemas seed the branches; β
> fixes the prior weighting. None of them addresses "why K=4 isn't
> beating K=1 in the first place." The implicit assumption is that
> once the substrate is fixed, K=4 will recover its advantage.

### Tension 5 — The successor failure mode that hasn't been named

The journey has named three failure modes (controller-triggers,
implementer-constant, wrong-reduction). β's failure is structurally
different from all three:

- The measurement `f_i` is local-per-atom (PASS mode 1).
- It is computed from substrate geometry (PASS mode 2).
- The reduction is sensible (mean of pairwise unbind-distances)
  (PASS mode 3).

But the measurement has *no signal* at D=4096 — the substrate
encoding produces a uniform value. This is a fourth mode:

> **Mode 4 (proposed):** a measurement that is locally correct but
> structurally constant — the encoding scheme dominates over any
> per-atom content variation, so the measurement carries no
> discriminative information.

The brainstorm should consider: are there other measurements in the
architecture also operating in this zero-signal regime, and just
not caught yet?

### Tension 6 — STATUS open questions that have not been resolved

From the active blockers and pre-phase commitments still open:

- **Seed-23 idiosyncratic geometry** (blocker #5, three runs identified)
  — never diagnosed.
- **θ′(β) calibration spike** — pre-Phase-3 commitment, never done.
- **Consolidation-geometry regime classifier** — pre-Phase-3, not built.
- **Brainstorm idea 5 (frequency-weighted Benna-Fusi α)** — "key
  experiment for the architecture's compression → abstraction claim,"
  never built (report 040 closed the related freq-α schema-source
  decision but the *abstraction-emergence* idea is separately listed).
- **Top1 regression is Phase 3 / Hebbian-codebook-reshaping, not
  Phase 4** (blocker #6′) — A1+A1' fix the discovery channel but
  not the underlying Hebbian-reshaping effect on top1.

> **Implicit thread:** Phase 5 may be trying to graduate on top of
> Phase 3/4 substrate properties that have known unresolved issues.
> The four 2026-05-20 notes assume Phase 4 is graduated; technically
> it is (on D1), but D1 graduation under binary death is what produced
> the d_eff collapse A+B is now redesigning. The "Phase 4 substrate
> is sound" assumption is in some sense already broken.

---

## 6. Threads the brainstorm could pull (outside A1''/β'/re-scope)

Surfacing what the design notes have *not* questioned:

1. **D as a free parameter.** The β failure mode is structurally
   D-bound. What happens at D=8192? D=2048? Is there a D where f_i
   has measurable variance and the substrate still supports W=4
   binding?

2. **FHRR vs alternative encoders.** The crosstalk floor is an FHRR
   property. HRR, BSC, sparse binary codes, or learned encoders
   (HEN-style, per the 2026-05-09 paper synthesis) would change
   the crosstalk geometry.

3. **The retrieval mechanism itself.** B2 firing says K-branching may
   be the wrong shape. What if structural retrieval emerges from a
   single coupled settle with a *different energy term* rather than
   from competing branches?

4. **The cue construction.** Every fix has been on the substrate side.
   The cue is generated by `_build_role_binding_cues` with fixed
   `binding_noise_std=0.05, content_distortion=0.6`. γ averages over
   a distribution of these, but the cue *shape* (cue ⊛ position⁻¹
   decomposition) is fixed. A different cue construction might
   produce role-prior advantage at the current D.

5. **The three open diagnostic-actuator pairs.** Drift/replay-pressure,
   metastability/replay-prioritization, cap-coverage/restructuring
   have no design notes. Any one of them, designed as a continuous
   local dynamic, might shift the substrate enough to give β a
   measurable signal — or to obviate it.

6. **The branching premise.** Phase 5's headline is "role-prior
   branches settle lower-energy than content-prior branches." Could
   structural retrieval emerge instead as a *property of the settled
   state*, not as a *comparison between branches*? E.g., a structural-
   completeness term in the energy itself.

7. **Substrate scale (attractor count).** Decision 5 closure noted
   the magnitude is bounded by N_schemas=12. A+B+A1+A1' preserves
   ~1064 atoms but their role-fidelity is uniform. There's a
   thread about *how many distinct meaningful attractors* the
   substrate actually has, separate from how many *atoms* it has.

8. **The freq-weighted α "compression → abstraction" idea**
   (brainstorm idea 5, STATUS pre-phase commitment) was bypassed by
   the report-040 finding that mass death already produces the under-
   capacity slow store. But that's the *quantity* finding; the
   *quality* claim ("compression produces abstraction") was never
   tested. Re-opening this would change what schemas *are*.

9. **The role-binding scheme itself.** W=4 positions, bind = complex
   multiplication, unbind = bind with conjugate. The role-fidelity
   signal at D=4096 is bounded by FHRR's unbind noise. Alternative
   binding schemes (sparse block bindings, learned positional
   transforms, hyperdimensional VSAs other than FHRR) are not
   considered in any note.

10. **Whether Phase 5 should graduate on the current substrate at
    all.** STATUS option (1) is "re-scope Phase 5's headline." The
    cleaner version is: declare Phase 5 substrate-fragile, document
    the binding limit at D=4096, and move to Phase 6 (predictive
    world model) which may not depend on this specific structural-
    retrieval mechanism. The PROJECT_PLAN's Phase 6 exit criteria
    talk about latent rollouts, not structural matching per se.

---

## 7. One-line summary

The four-step debugging journey has correctly closed three named
failure modes at the substrate construction, init, and measurement
layers, and β's empirical failure surfaces a fourth (mode 4: locally-
correct measurement that is structurally constant under the encoding
scheme). The journey has been *peeling A's lifecycle*; the brainstorm
should consider mechanisms outside A's lifecycle entirely — including
questioning D=4096, FHRR, the branching premise (B2 has fired and is
not retracted), the cue construction, the three unattempted
diagnostic-actuator pairs, and whether the Phase 5 headline as
written is achievable on this substrate at all.
