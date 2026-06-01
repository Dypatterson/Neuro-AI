# Phase-3 Consolidation-Write — Dorrell Range-Shaped Structure-Writing Objective (HARDENED design spec)

> Status: **DESIGN-ONLY, NOT APPROVED-TO-BUILD.** This is the hardened spec for the
> deferred Dorrell *training-objective* lane (Report 068:120-128, "that work is deferred";
> distinct from the storage-side `range_postsettle` proxy that Report 111 closed — see
> §Lane provenance). It folds in a five-lens grill (break-it, anti-homunculus,
> done-gate+fence, terminology, plus a primary-opened paper card). **Three FATAL
> findings against the original "dual-reader" candidate forced a structural collapse;
> the design below is the post-collapse single-reader form.** Do not write code until
> the OPEN DECISIONS (§13) are resolved and the experiment preamble (§11) is filled.
>
> Citation target for the experiment preamble: this file. Headline source of truth:
> [phase-3-consolidation-write-design.md §Headline R1](phase-3-consolidation-write-design.md) (lines 55-64)
> and §Required controls (lines 118-137).

---

## 0. TL;DR — what changed under the grill, and what this is now

The candidate was a **dual-reader** design: (R-write) the graduated FHRR+MHN
heteroassociative memory carrying the project's Selectivity-Δ headline, fed
range-shaped cues; and (R-AE) a bolted-on nonneg autoencoder on the *same* rebound
phasors, carrying the Dorrell modularization signal — "compared, not merged."

The break-it grill **REFUTED that framing** with three fatal structural defects, all
verified against the opened primary (arxiv:2410.06232 v4) and the code:

1. **FATAL-1 — the theorem's precondition is violated by the FHRR input.** Dorrell's
   modularity is defined as "each row of `W_in` has at most one non-zero entry" *with
   respect to the ground-truth source coordinates*, where the encoder reads
   `z = ReLU(W_in·s + b)` from sources `s` directly, or from an **invertible linear
   mixture** `x = A·s` (paper §B.4, and even there only *local* optimality). The
   candidate's input was `x = [Re(q); Im(q)]` with `q = bind(pos[r], cb[a]) =
   exp(i(θ_r + φ_a))`, so each coordinate is `cos(θ_r[d]+φ_a[d])` — a **trigonometric,
   non-invertible** function of the two factors, **not** a linear mixture `A·s`. The
   modularity *guarantee* does not transfer; a ReLU on this manifold is the cargo-cult
   port the grounding warned about.
2. **FATAL-3 — the headline reader could not be moved by the mechanism.** R-write's
   write rule was unchanged; Dorrell only reshaped *which cues it saw*. So
   modularization (in z) was **causally disconnected** from the Selectivity-Δ headline,
   and the most plausible cause of any lift was a value-codebook **coverage/data-volume**
   artifact (no coverage-matched control). The R-AE drill-down sat on a disconnected
   reader and so **could not adjudicate** the confound — a violation of the project's
   headline-vs-drill-down discipline.
3. **OVERARCHING-FATAL — decoupling bought anti-homunculus safety at the cost of
   evidential coherence.** A "Stage-1 GO" would be four loosely-coupled readings that do
   not *jointly* license "rectangular replay drives Dorrell modularization that improves
   the memory."

**The hardened design therefore collapses to a SINGLE theorem-faithful reader** on
**linear source inputs**, where (i) the precondition holds, (ii) modularity is defined
on axes not handed to it by construction (with anchors), and (iii) modularization is
causally upstream of the adjudicated metric. It is reported, honestly, as a **modest
Phase-5-modularization PROBE — "does Dorrell reproduce on a VSA-flavored toy, and does
rectangularizing the replayed support drive it?"** — fenced hard from the FHRR
substrate, the consolidation write, and the Selectivity-Δ memorization headline.

This is **de-risking by downgrade**: we keep exactly the part of the bet the theorem
can actually license, and we drop the part the grill proved it cannot.

---

## 1. Lane provenance (terminology grill — fold #T-lane / #DG-lane)

- This exercises the **Report 068 deferred TRAINING-objective lane** — the nonneg
  energy-efficient AE whose *optimum* modularizes (STATUS.md:19 "Stage-1 candidate under
  Selectivity-Δ"; phase-3-consolidation-write-design.md:163-164).
- It is **NOT** the storage-side `range_postsettle` novelty-without-retrieval proxy that
  **Report 111 closed**. A reader must not mistake this for re-opening the 111 lane. The
  range-shaped *sampler* (Report 068, KL 2.08→0.013, coverage 32→256, n=3) is shared
  data-shaping machinery; the *training objective* on top of it is the open part.
- **Phase ownership (terminology grill #T-phase):** the range-shaped sampler + a
  structure-writing objective are an **ACTIVE Phase-3 consolidation-write candidate**
  (spec:154-164), not a Phase-5 bet. The *only* piece touching the paused
  Phase-5 "modularization" concept is the **latent-code factorization** measured here,
  and even that is latent-code factorization of a standalone probe — **NOT** the paused
  Phase-5 codebook bind-vs-bundle / atom-splitting deliverable
  (experimental-progression.md §"Phase 5"), which needs learned role atoms that do not exist
  until Phase 5. The design **does not reopen Phase 5 or Phase 5′.**

---

## 2. What modularizes, and on what (the precondition, made explicit)

At Phase 3 there are **no learned role atoms**: roles are FIXED substrate-position
vectors (`build_position_vectors`, `encoding.py:18-26`; `bind(positions[index],
codebook[token_id])` at `encoding.py:38` ⇒ **role ≡ position**), and the codebook holds
**only filler/value atoms**. So "the Phase-3 codebook modularizing into role-cells vs
content-cells" is **not well-posed and is NOT claimed.**

**The single reader operates on LINEAR SOURCE INPUTS, not on bound phasors** (FATAL-1
fix, route (b)):

- Sources `s = [onehot_role (R dims); onehot_atom (N dims)]`, `s ≥ 0` (onehots are
  naturally nonnegative — a clean match to the theorem's source domain).
- Input to the AE is either `x = s` directly, or `x = A·s` for a **precommitted random
  invertible** `A ∈ R^{(R+N)×(R+N)}` (so the modularity theorem's "invertible linear
  mixture" precondition, §B.4, holds *by construction*).
- **What modularizes** is the latent code `z = ReLU(W_enc·x + b)`: Dorrell predicts the
  `K` latent neurons partition into a **role-selective** group (reads the role-onehot
  block, invariant to atom) and a **content-selective** group (reads the atom-onehot
  block, invariant to role), **iff** the support of `s` is range-rectangular and `z ≥ 0`
  + energy penalty are present. Modularity = "each row of `W_enc` has ≤ 1 nonzero entry"
  w.r.t. the `(role, atom)` source axes (Thm 2.1).

**Stage-(−1) PRECONDITION CHECK (FATAL-1 fix, mandatory, ~5 min):** before any modularity
claim, assert `x` is an invertible-linear function of the source onehots (identity, or
`A` with `cond(A)` recorded and finite). If a future variant feeds bound phasors, the
theorem claim is **abandoned** and the run is relabeled an empirical curiosity. This
check is a hard gate, not a convention.

**Honest scope statement (must appear in any report):** a PASS proves *Dorrell
reproduces on a VSA-flavored linear-source toy and that rectangularizing the source
support drives it* — it proves **nothing** about whether the FHRR substrate, the
consolidation write, or the value codebook modularize, and **nothing** about held-out
prediction (the theorem is about the *optimum's factorization*, not generalization —
paper has no held-out eval anywhere).

---

## 3. Mechanism (single reader, end-to-end)

1. **DATA — skewed source buffer.** A synthetic buffer of `TrajectoryTrace`s with
   `encoder_terms = (role ∈ 0..R-1, atom ∈ 0..N-1)`, deliberately **skewed** so the
   `(role, atom)` joint is non-rectangular (each role natively draws a contiguous atom
   block). This is the falsifier-(i) raw arm.
2. **RECTANGULARIZE — the data-shaping engine.** `RangeShapedReplaySampler.sample_pairs(n)`
   draws `(role, atom)` from `p(role)·p(atom)` factored marginals
   (`range_shaped_replay.py:128-133`), turning the skewed joint rectangular (KL
   2.08→0.013, coverage 32→256 per Report 068). The **FIXED** Laplace atom prior
   (`atom_smoothing_alpha` + `atom_count`, lines 89-91) reaches atoms with zero buffer
   weight so the rectangle covers the full `R×N` grid.
3. **MATERIALIZE the source matrix.** For each sampled `(role, atom)`, build
   `s = [onehot_role; onehot_atom]` (and `x = A·s` if the mixture arm is on). **No FHRR
   bind, no rebind-on-the-fly, no `encode_window_with_provenance`** — those were the
   FATAL-1 / attack-(4) hazards and are removed from the reader path. (The sampler's
   rebind synthesizers `synthesize_single_binding_trace`/`synthesize_window_preserving_trace`
   are **not used** by this reader.)
4. **TRAIN — offline batch nonneg AE.** `z = ReLU(W_enc·x + b_enc)`, `x̂ = W_dec·z + b_dec`;
   minimize `recon-MSE(x, x̂) + β·mean(||z||²) + λ·(||W_enc||² + ||W_dec||²)`. Offline
   batch gradient descent on the frozen rectangularized dataset. `K ≥ R + N + headroom`.
5. **READ — modularization measurement.** Linear probes `role ← z`, `atom ← z`, plus the
   **module-disjointness** score and the **rectangularization-Δ** (see §4). All
   measurements are post-hoc; no metric gates training or arm selection.

The mechanism never inspects a downstream metric to branch. Sampler, mixture-on/off,
β/λ grid, and the rectangular-vs-skew arm are **static config**; every arm runs; all
comparisons are post-hoc (anti-homunculus PASS — §7).

---

## 4. Headline metric + modularization read (single adjudicator)

**FATAL-3 + terminology #T-headline fix — the Selectivity-Δ memorization headline is
DROPPED for this bet.** The original candidate read Selectivity-Δ on R-write, a reader
the mechanism could not causally move; that headline belongs to the in-sample
memorization program (Report 056, judged in-sample) and is **not** this bet's
adjudicator. There is no R-write reader in the hardened design.

**SINGLE GRADUATION ADJUDICATOR — the rectangularization-Δ of module-disjointness:**

```
Δ_mod = disjointness(z | range-shaped support) − disjointness(z | joint-preserving support)
```

measured on the SAME atoms/seeds/objective, with a CI. **Only the Δ attributable to
making the support rectangular is a Dorrell signal**; raw disjointness alone is not (it
can arise from a generic bottleneck on near-orthogonal inputs — attack-2). The headline
is `Δ_mod > 0` with a bootstrap CI strictly above 0, **and** the range-shaped arm's
absolute disjointness above its shuffled-axis floor (§5 anchor).

**Disjointness, precommitted operationalization (done-gate #DG-disjoint fix — no tunable
knob):** for each latent neuron `k`, train logistic probes and read its role-loading
`a_k = |w_role,k|` and atom-loading `b_k = |w_atom,k|`; normalize each to `[0,1]` across
the layer; call neuron `k` **shared** if `min(a_k, b_k) > 0.5`; report
`disjointness = 1 − (shared fraction)`. Threshold `0.5` is **fixed a priori**, never
swept. (Threshold-free alternative offered as an OPEN DECISION, §13.)

**Drill-downs (subordinate — explain the headline, never gate it; terminology #T-gate
fix):**
- Probe accuracies `acc(role←z)`, `acc(atom←z)` — a high `Δ_mod` should coincide with
  high single-factor probe accuracy.
- The **lambda/beta-invariance** profile (§5.ii) — read OFF the fixed grid, never used to
  re-pick an operating point.
- The **linear-latent necessity** result (§5, nonnegativity falsifier).

These are **attribution checks reported alongside**, NOT GO gates. The Stage-1 GO is the
single conjunction in §10; the panel disambiguates *why* `Δ_mod` moved, it does not
provide a competing definition of success.

---

## 5. The three built-in falsifiers (from the paper) + the anchors the grill demanded

Primary-grounded (paper card, opened arxiv:2410.06232 v4):

- **(i) NON-rectangular support must FAIL to modularize.** Run the reader on the RAW
  skewed buffer joint (joint-preserving sampler, no rectangularization) on the SAME
  atoms/seeds. Predict disjointness near floor. This is the precommitted baseline
  (Report 068 head-to-head) and the *reference arm of the headline Δ_mod*.
- **(ii) λ/β-INVARIANCE.** Sweep the weight penalty `λ` AND the activity-energy penalty
  `β` across ~2 orders of magnitude (`β ∈ {0, 1e-3, 1e-2, 1e-1, 1e0}`, `λ` over a
  matched grid). The theorem says modularity does **not depend on the energy tradeoff
  parameter** (paper §2.3, verbatim: "Remarkably, they do not depend on the energy
  tradeoff parameter λ"). So `Δ_mod` and disjointness must be **approximately invariant**
  across the sweep — **except `β=0`**, the degenerate arm that must NOT modularize (with
  `β=0` the activity-energy floor is absent and `z≥0` is decorative). If modularity
  *tracks* β monotonically, it is an activity-penalty artifact → REJECT.
- **(iii) PRECOMMITTED BASELINE.** Joint-preserving vs range-shaped sampler on the
  IDENTICAL frozen buffer/seed, same objective — this *is* the headline Δ_mod (§4).

**NONNEGATIVITY-NECESSITY control (the port's own falsifier — citation honesty, break-it
secondary + done-gate #DG-appendix):** run a **linear-latent** arm (drop ReLU →
identity/linear `z`). If the linear latent modularizes *equally*, `z≥0` is decorative and
the port is **null-for-Dorrell** — report it as such, not as a success. **Citation
discipline (RESOLVED by the paper card):** the paper's nonnegativity-necessity ablation is
**§F.2.1** ("Biological Constraints are necessary for Modularity"), nested under Appendix
**F** ("What-Where Task") — **NOT Appendix G** (Appendix G is "Nonlinear Autoencoders").
The prior gate card (2026-05-30-gate-cards.md:66) was wrong on the letter *and* the title;
the corrected card resolves it. **Two further honesty constraints carried from the
primary read:** (a) F.2.1 ablates positivity AND energy-efficiency **jointly** — it is NOT
a clean isolation of `z≥0` alone; (b) the paper contains **no analytic "signed-latents
break modularity" theorem**. So our linear-latent arm is the **design's own** necessity
test, motivated by the paper's structural premise, **NOT** "the paper's ablation" — a
report may cite F.2.1 for the joint What-Where empirical finding but must not claim the
paper isolates `z≥0` alone.

**Two anchors the break-it grill (attack-2) demanded, because raw disjointness/probe
accuracy can be a bottleneck-generic artifact on near-orthogonal inputs:**
- **Shuffled-axis anchor:** re-run with the `(role, atom)` source labels RANDOMLY
  PERMUTED relative to inputs. Disjointness *w.r.t. the shuffled axes* must collapse. If
  it doesn't, disjointness is measuring bottleneck-generic structure, not factor
  alignment. (This is the linear-source analogue of the project's role-pairing shuffle,
  spec control 3.)
- **Random-source anchor:** run on inputs with no factor structure (single random
  source vector per sample). Disjointness must be undefined/floor.

The headline Δ_mod (§4) already builds the rectangularization anchor in; the shuffled-axis
and random-source anchors guard against the "any bottleneck factors near-orthogonal
inputs" confound.

---

## 6. Toy where modularization is well-posed AND measurable

Modularization is well-posed because the two generative factors are **known and
independently controllable** by construction, and the theorem's precondition holds on
the linear source representation.

- **Sources:** `R = 4` roles, `N = 16` atoms ⇒ `R + N = 20`-dim source onehot, `R×N = 64`
  joint cells.
- **Skewed buffer:** each role natively draws a contiguous 4-atom block (non-rectangular
  joint). Rectangularize via `RangeShapedReplaySampler` + FIXED Laplace prior → uniform
  over the 64-cell grid.
- **AE:** `K = 32` latents (`≥ R + N + headroom`), offline batch GD.
- **Upper-bound arm:** a perfectly modular ground-truth AE (block-diagonal `W_enc`) bounds
  the achievable disjointness so a null is "didn't modularize" vs "no recoverable
  structure in principle."

**Toy-validity inversion (terminology grill #T-toy — must be stated loudly in any
report):** the design panel proved a clean 4-role *rule-toy* **cannot** reproduce the
**consolidation-WRITE** signature (memory `neuro_ai_design_panel_clean_toy_cant_reproduce`;
spec §G-D Forks 1/2/3). **That ruling does not bind this toy**, because this toy tests a
**different object** — representation-factorization of a learned nonneg latent on
linear-source inputs, NOT a role→target rule — for which a clean controllable-factors toy
is **exactly right**. The report must draw this distinction explicitly or it will (rightly)
draw the "clean toys were already ruled out" objection. Note: because the dual-reader
collapsed, there is **no longer a masked-encoding/topic-corpus grounding** in this design —
the bet is **synthetic-only** through both stages, which is honest for a "does the theorem
reproduce on a VSA toy" probe and is the reason it graduates nothing (§9).

---

## 7. Anti-homunculus check (PASS — fold the AH grill, 0 fatal)

The AH grill returned **PASS overall, 0 fatal**, verified in code. The collapse to a
single reader on linear sources only *strengthens* this (the FHRR rebind path, the only
non-trivial surface, is removed). Component-by-component:

1. **Factored-marginal sampler** (`range_shaped_replay.py:100-141`): `torch.multinomial`
   over `p(role)`, `p(atom)` computed once in `_rebuild_index` from the frozen
   store — a product-of-marginals draw, the definitional rectangularize op. No
   runtime-metric branch (docstring lines 3-7). **PASS, no change.**
2. **Laplace atom prior** (lines 56-57, 68, 89-91): set once as a constructor arg, frozen,
   consumed only to add a flat count. **Structurally cannot become a thermostat in code.**
   **PASS-with-precommit:** bind `atom_smoothing_alpha` + `atom_count` in static config
   before any arm runs; record in the preamble; forbid re-selecting `α` post-hoc on
   `Δ_mod`/disjointness.
3. **The nonneg AE** (`z = ReLU(W_enc·x+b)`, UNBUILT): the "decision" of which neuron
   reads role vs atom lives in the **energy-minimizing geometry of the AE optimum under
   rectangular support** — a LOCAL optimization dynamic, the one the filter permits, not
   an arbitration. The ReLU is a STATIC architectural constraint. **PASS-with-precommit:**
   (a) `β=0` is a precommitted degenerate arm that MUST NOT modularize; (b) "offline batch
   GD on the frozen dataset" is a **hard architectural fence**, never an online/error-driven
   update path. *(AH-grill soft spot, addressed: the optimum here is a generic AE optimum
   on a linear-mixture manifold — the §B.4 setting where the theorem at least has
   *local*-optimality footing; we do NOT over-claim it is the global Dorrell optimum, and
   the linear-latent + β-sweep falsifiers distinguish a real energy dynamic from a
   decorative ReLU.)*
4. **Objective** (`recon-MSE + β·||z||² + λ·||W||²`): a static loss; β/λ swept as
   precommitted static grid across all arms, never adaptively set. **PASS-with-precommit.**
5. **Readout**: post-hoc logistic probes + disjointness on `z`; no controller; all arms
   run; arm selection is static config, comparison post-hoc. **PASS, no change.**
6. **Regression guard (highest residual, cross-cutting):** the BANNED shortcut is wiring
   the AE into the online `ReconstructionLearner._consolidate` contrastive loop
   (`src/energy_memory/phase2/reconstruction_learner.py:136-162` — *citation corrected per
   done-gate #DG-cite: file is `phase2/reconstruction_learner.py` not `phase34`; the
   `_consolidate` pull/push renorm is 136-162; the `sims.argmax` thermostat is at line
   106 in `train`, separate from the renorm block*). That loop reads `predicted_id !=
   target_id` (line 143) to pick push targets and renormalizes atoms to **unit magnitude**
   (lines 148-158) — an online error-driven update AND the **opposite of an energy code**.
   **Add a doc-fence / structural assertion that neither the sampler nor the AE may be
   imported or called from `reconstruction_learner._consolidate`.** The frozen-buffer raise
   in `heteroassociative_write` (`hetero_write.py:156-159`) enforces the equivalent on the
   R-write side; the AE side needs its own discipline since R-write is no longer used here.

**Where the "decision" lives:** in the local energy-minimizing geometry of the nonneg AE
optimum under rectangular support — a local geometric dynamic, exactly what the
anti-homunculus filter permits.

---

## 8. Fence bright-line (PASS — fold the done-gate+fence grill, 0 fatal)

The fence grill returned **0 fatal, FENCE-CLEAN.** Bright line (binding): no read path may
feed scores into `_energy_from_scores`/`mhn_energy` → `min`/`argmin`-over-branches → ΔE.

- This reader's modularization read terminates in **logistic-probe accuracies + a
  disjointness count over latent neurons** — no energy scalar, no min-over-branches, no
  ΔE. Fence trivially held.
- The Phase-5′ `min_branch` aggregator (`m1_role_energy.py settle_weighted_mhn`,
  argmin-over-branches) **stays OUT and de-arbitrated**; the Phase-5′ ΔE re-run **stays
  fenced** (STATUS.md:18-20, spec:176-177).
- **Latent fence-creep surface (done-gate #DG-fence, carried forward as a guard even
  though the post-GO R-write integration is removed):** *if a future revision ever
  re-introduces an R-write reader and integrates into `UnifiedReplayMemory.run_replay_cycle`
  (`replay_loop.py:627`)*, that cycle carries a `resolve_threshold` score-gate
  (`replay_loop.py:691`). That gate is sanctioned for **which cues get replayed**
  (data-shaping) but must NEVER filter **which cues get scored** for a headline. Recorded
  here so the surface is not re-discovered later.

---

## 9. Done-gates (fold done-gate grill #DG-done — was MISSING from the candidate)

Each stage is "done" per CLAUDE.md §"What done looks like" (all five items):
1. Headline (`Δ_mod`) reported with a bootstrap CI; β/λ-invariance profile with CIs.
2. Control on the same data: joint-preserving baseline + shuffled-axis + random-source
   anchors + linear-latent necessity + `β=0` degenerate arm + upper-bound arm.
3. Drill-downs (probe accuracies, invariance profile) explain the headline.
4. **A numbered report under `reports/`** (Stage-0 = nonneg-recoding admissibility;
   Stage-1 = modularization go/no-go). *(Candidate omitted this — now mandatory.)*
5. **STATUS.md Recent-updates (≤5 lines) + Phase-3 checklist + the Dorrell-bet memory card
   updated BEFORE ending any session that changes status.** *(Candidate omitted this.)*

**Graduates NOTHING (over-investment guard):** even a clean Stage-1 GO licenses only (a)
deciding whether the modularization probe is worth carrying further, and (b) any future
substrate-faithful attempt — it is **not** a Phase-3 graduation, **not** a memorization
result, and **not** a generalization result. Treat both stages strictly as gates.

---

## 10. The FIRST cheapest-kill go/no-go (two-stage; skeptic's kill is Stage 0)

**Precommit numeric thresholds in the report preamble (done-gate #DG-precommit) so a
marginal result cannot be talked across the gate post-hoc.**

### STAGE −1 — PRECONDITION CHECK (FATAL-1 gate, ~5 min, no training)
Assert `x` is identity or `A·s` with finite `cond(A)` recorded. If a variant feeds bound
phasors, **abandon the theorem claim** and relabel. This gates Stage 0.

### STAGE 0 — ADMISSIBILITY / NECESSITY KILL (cheapest, <1 day, D-free, R=4, N=16, 3 seeds, ~60 LOC new)
Fit the minimal honest nonneg AE on range-shaped **linear-source** data. Measure:
- **(M1) Can `z≥0` host a recoverable factor code at all?** Probe `role←z` and `atom←z`;
  require `acc(role←z)` and `acc(atom←z)` both above their shuffled-axis floors with a CI.
- **(M2) β-invariance of disjointness** across the 2-OOM β sweep.

**KILL / NO-GO (drop Dorrell on this toy):**
- M1 probes collapse to shuffled-axis floor (the nonneg latent cannot host a factor code
  at all), OR
- the **linear-latent** arm modularizes *identically* to the ReLU arm (z≥0 decorative —
  cargo-cult), OR
- disjointness *tracks* β (activity-penalty artifact, not the theorem).

*(The skeptic's sharpest worry — "z≥0 kills the read" — is structurally defused: there is
no live unbind read in this design; the latent is a parallel diagnostic readout judged by
signed reconstruction, never feeding cleanup. Stage 0 tests only whether `z` can host a
recoverable factor code, a weaker, survivable bar.)*

### STAGE 1 — MODULARIZATION GO/NO-GO (only on Stage-0 survive, ~half-day, R=4, N=16, n=5, full 64-cell grid)
**GO iff the single conjunction holds (one adjudicator, drill-downs reported alongside):**
- **(a) HEADLINE:** `Δ_mod = disjointness(range-shaped) − disjointness(joint-preserving)
  > 0` with a bootstrap CI strictly > 0, AND the range-shaped arm's absolute disjointness
  above its shuffled-axis floor; AND
- **(b) ATTRIBUTION (reported, used to interpret a GO, not as a co-equal gate):** β/λ
  invariance holds (β=0 excepted), the linear-latent arm does NOT modularize, the
  random-source anchor is at floor.

**NO-GO** if the range-shaped arm shows no `Δ_mod`, or the linear-latent arm passes
equally, or disjointness tracks β.

A Stage-1 GO **licenses nothing operational** beyond a decision on whether to invest
further (§9). No Phase-4/wikitext spend at any point in this lane — the bet is
synthetic-only and graduates nothing (DON'T-SCALE-A-NULL, verified clean by the
done-gate grill).

---

## 11. Experiment preamble (MANDATORY before the first run — done-gate #DG-preamble)

Fill and record before Stage 0:

> **Active phase:** 3 (consolidation-write sub-program)
> **Headline metric per [this file §4]:** rectangularization-Δ of module-disjointness
> `Δ_mod` on the linear-source toy — NOT the spec:55-64 Selectivity-Δ (that is the
> memorization headline, on a different reader, dropped for this bet). **This experiment
> is a DRILL-DOWN/PROBE, not a graduation experiment** — it does not measure the design-spec
> headline, so per CLAUDE.md it is labeled a drill-down by definition.
> **Required controls per [this file §5] + [spec:118-137]:** joint-preserving baseline,
> shuffled-axis anchor, random-source anchor, linear-latent necessity, β=0 degenerate arm,
> upper-bound arm.
> **Last verified result:** Report 068 (sampler-level rectangularization, n=3); the
> training objective is DEFERRED/OPEN (STATUS.md:19).
> **Why this experiment now:** addresses STATUS blocker (1) range-lane correction — the
> Dorrell training-objective lane is OPEN and the user opted in; a local kill-gate before
> any spend.

Because the headline measured here is **not** the design-spec Selectivity-Δ, every report
must state in its first paragraph that it is a **Phase-5-modularization PROBE / drill-down**,
not a graduation experiment.

---

## 12. LOC / integration plan (single-reader, FHRR-rebind path removed)

Reusing what is built; the nonneg AE is the only substantial new code.

| Component | LOC | Source |
|---|---|---|
| Nonneg AE module (`relu` bottleneck, signed decoder, `recon-MSE + β·||z||² + λ·||W||²`, offline batch GD) | ~45 new | new file under `phase4/` or `phase5/` (OPEN DECISION §13) |
| Linear-source materializer (`(role,atom) → [onehot; onehot]`, optional `A·s`) | ~15 new | new helper |
| Disjointness + probe read (logistic probes, fixed-0.5 disjointness, bootstrap `Δ_mod`) | ~30 new | reuse `phase2/metrics.py` Wilson/bootstrap |
| Skewed-buffer + rectangularize harness | ~10 | reuse `RangeShapedReplaySampler` (`range_shaped_replay.py`), FIXED Laplace prior (lines 89-91) |
| Doc-fence assertion (AE not importable from `reconstruction_learner._consolidate`) | ~3 | new |
| **TOTAL** | **~100 new LOC** | |

**Removed vs the candidate:** the rebind synthesizers (`synthesize_single_binding_trace`,
`synthesize_window_preserving_trace`), the FHRR real-lift adapter, the
`observe(cue=)/consolidate_hetero/recall_hetero` R-write path, and all
`encode_window_with_provenance` bundling — none are on the single-reader path. This also
**eliminates attack-(4)** (the window-preserving bundle reproduces the 065/066 bundled-key
null by construction) entirely, since no bundled cue is ever fed to a reader.

**No Phase-4 `run_replay_cycle` integration** in this lane (synthetic-only; graduates
nothing). The `resolve_threshold` fence-creep surface (§8) is therefore not exercised, and
recorded only as a guard for any future revision.

---

## 13. OPEN DECISIONS (user must resolve before building)

1. **Mixture arm: identity `x = s` vs invertible `x = A·s`?** Identity is the cleanest
   test of Cor 2.3 (range-independence of the source support). `A·s` exercises the §B.4
   linear-mixture (local-optimality) setting and is a slightly stronger "the theorem
   transfers through an invertible mixture" claim, but only *local* optimality is
   guaranteed there. **Recommend: run identity first (Stage 0), add `A·s` as a Stage-1
   robustness arm.** User confirms.
2. **Disjointness operationalization: fixed-0.5 threshold vs threshold-free?** §4 uses the
   precommitted fixed-0.5 `min(a_k,b_k)>0.5` rule. The done-gate grill offered a
   threshold-free alternative (per-neuron `|cos(role-weight-vec, atom-weight-vec)|` or
   mixing-matrix off-diagonal mass). **Recommend threshold-free** (no knob to leak), but it
   is a different number; user picks one and it is fixed a priori.
3. **β/λ grid values.** `β ∈ {0, 1e-3, 1e-2, 1e-1, 1e0}` proposed; confirm the matched λ
   grid and the headroom `K` (proposed `K=32`).
4. **Where the new module lives** — `phase4/` (alongside the sampler) vs `phase5/`
   (modularization is Phase-5-flavored). Affects the doc-fence import path. **Recommend
   `phase5/`** to signal it does not touch the active Phase-3 write path; user confirms.
5. **Is the downgraded "does Dorrell reproduce on a VSA toy" probe worth the ~100 LOC + 1.5
   days at all?** The break-it grill's honest verdict is that this is the *only* salvageable
   form, and it is **modest** — it cannot say anything about the FHRR substrate. The user
   opted into the Dorrell bet expecting a substrate-faithful or generalization-axis result;
   this delivers neither. **This is the load-bearing decision:** proceed with the modest
   probe, or decline the lane. (See §First-killtest framing.)
6. **Manifest upgrade for arxiv:2410.06232** (currently `link_only`, `authors:"unknown"`,
   truncated title). The paper card resolved authors/title/appendix from the opened
   primary; upgrading the manifest is a small chore but not a build blocker.

---

## 14. STOP-RULE (fold done-gate #DG-stoprule — was un-operationalized)

The candidate named the real-text-held-out stop-rule but attached **no metric and no
threshold**, so it could not fire. **In this design the stop-rule is structurally
inert** through both stages — the reader is **synthetic linear sources only**, never real
text, and never produces a held-out recall number. There is therefore nothing to trigger
it. **Recorded explicitly so a future scale-up cannot silently bypass it:** *if any future
revision ever feeds this lane real-text data and measures held-out recall, attach a
precommitted held-out `top_index_hits` recall vs `1/|value codebook|` chance with a Wilson
CI; if the CI-lower clears chance, HALT and surface it as a tension with
memorization-is-target (Report 056) — do not bank it into a Dorrell result.* For the
present synthetic-only design, the trigger cannot fire by construction.

---

## 15. De-risked vs still speculative (honesty ledger)

**De-risked (verified in code / primary this session):**
- The data-shaping engine (factored-marginal sampler + FIXED Laplace prior) is built and
  validated (Report 068; `range_shaped_replay.py:89-91,128-133`).
- Anti-homunculus PASS and the fence hold by construction (both grills: 0 fatal;
  `hetero_write.py:65` argmax-count terminus verified; `reconstruction_learner.py:136-162`
  confirmed as the BANNED online unit-renorm path to fence the AE away from).
- The paper card is primary-grounded: Thm 2.1/2.2, Cor 2.3, the verbatim λ-independence
  sentence, and the F.2.1 (NOT G) nonnegativity ablation are confirmed.
- The linear-source single reader satisfies the theorem's precondition (Stage −1 check),
  closing FATAL-1.

**Still speculative (genuinely unknown until Stage 0/1 run):**
- Whether a ReLU-bottleneck AE on this *specific* 4×16 linear-source toy actually
  modularizes at the optimum — the theorem guarantees it *iff* the support is
  range-rectangular, but our materialized rectangle is an engineered uniform grid, and
  whether GD reaches the modular optimum (vs a mixed local min) is empirical.
- Whether `Δ_mod` is cleanly positive (rectangular > skewed) with separable CIs at n=5.
- Whether the linear-latent arm fails (nonnegativity load-bearing) as the theory predicts,
  or passes (port null-for-Dorrell).

**Honestly conceded (the bet shrank):** even a clean GO is a "Dorrell reproduces on a VSA
toy and rectangularization drives it" result. It says **nothing** about the FHRR
substrate, the consolidation write, the value codebook, held-out generalization, or the
memorization program. The substrate-faithful and generalization-axis claims the user
originally opted into are **not** deliverable through a theorem-faithful port — that was
the break-it grill's core finding, and this design accepts it rather than papering over
it.