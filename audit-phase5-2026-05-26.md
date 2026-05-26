# Audit: Phase 5 Ideation — 2026-05-26

Reconnaissance only. Bullets through Phase 4, structured idea cards in Phase 5.
Branch storyline collapses to **one chain** (`codex/phase5-prime-bundle-first-scene-memory`
tip, 75 commits ahead of `main`); the predecessor branches are checkpoints on
the same chain. `product/grounded-ideation-spike-002` ignored per user
direction.

---

## TL;DR — 3 highest-leverage moves

These are the three moves with the largest expected ratio of (Phase-5
unblocking) to (cost + principle risk). They are **proposals, not decisions**.
Reasoning behind each lives in §5.

1. **Pivot the Phase 5 headline off `E_min(branch)`.** The current `ΔE`
   headline (design spec line 256–281) compares the **min final-state energy**
   across branches. Reports 103–109 show this readout is structurally
   degenerate: every successfully-stored scene-MHN basin sits at `E = -1.0` by
   construction, so the headline is sampling a saturated tail with no
   headroom. Strict-discriminator probes show max mean `ΔE = 0.00217` against
   a floor of `5.5e-3`. The bridge is not broken; the **scoring convention
   has no room to move**. A floor set on a saturating quantity is a
   measurement-design bug, not a substrate failure. (Requires user-approved
   objective change, which STATUS line 8 already names as the only forward
   path.)

2. **Backfill Phase 1's missing D=4096 bind/unbind fidelity benchmark before
   the next bridge attempt.** This is a one-spike, one-test gap (current
   tests stop at D=256/512). Every Phase 5 mechanism — bundle-first,
   range-shaped replay, ΔE bridge — chains 16 role-binds at D=4096. Silent
   roundtrip error here would mimic every failure mode the bridge work has
   been chasing. Cheap to close, gates everything downstream.

3. **Operationalize the "metric ↔ actuator identity" rule on the
   diagnostic stack that already exists on paper.** NC1, separability,
   bimodality, cap-coverage live in `notes/notes/2026-05-09-…md`,
   `phase-3-deep-dive.md`, `consolidation-geometry-diagnostic.md`, but
   the live consolidation loop computes none of them. Phase 5 inherits a
   Phase 3 codebook that **may be collapsing to points along high-variance
   directions** with no instrument to detect it. Lift these diagnostics into
   *substrate-side pressures* (the slow-timescale dynamic the diagnostic
   snapshots) per the 2026-05-09 design directive. This is the principled
   anti-homunculus move and it closes the largest measurement gap in Phases
   3–4.

---

## §1. Core principles (Phase 1.a)

Confirmed verbatim by user; for full quotes see chat. Working set used in §5:

- **CP1.** No controller — apparent decisions are local geometry, energy,
  settling, tension, consolidation dynamics. (`PROJECT_PLAN.md:14-18`)
- **CP2.** Memory is the self. (`PROJECT_PLAN.md:19-20`)
- **CP3.** Contextual completion over token prediction. (`PROJECT_PLAN.md:21-22`)
- **CP4.** Continuous learning — replay consolidates and reshapes the
  landscape. (`PROJECT_PLAN.md:23-26`)
- **CP5.** Latent reasoning in vectors/energy before language.
  (`PROJECT_PLAN.md:27-28`)
- **CP6.** Energy efficiency / compact latent ops / MPS / local-first.
  (`PROJECT_PLAN.md:29-32`)
- **CP7.** Anti-homunculus filter (hard rule): every addition is a local
  geometric dynamic or a measurement of one, never an arbitration over them.
  (`2026-05-09:93`)
- **CP8.** Diagnostic↔actuator identity: an actuator is a slow-timescale
  dynamic that a diagnostic is a fast-timescale snapshot of.
  (`2026-05-09:150-154`)
- **CP9.** Headline metric over panel-of-metrics; drill-downs explain the
  headline, not replace it. (CLAUDE.md + `2026-05-09:75-87`)
- **CP10.** Substrate ops fixed up front; codebooks grow from experience;
  bind-vs-bundle is **discovered, not declared**.
  (`literature-and-principles.md` design principles 1–3)

Non-negotiable rules (`PROJECT_PLAN.md:272-279`) repeated here because §3 and §5
test them: no subsystem-winner module; no ad-hoc if/then routing; no
vector-DB+summaries collapse; no LLM-as-identity; **no removal of pure-Python
reference**; no new mechanism without a passed control.

### Where the code suggests drift from the principles

- **CP3 (contextual completion)** is real in the bundle-first design but
  Phase 2 still grades by `Recall@1`/`accuracy`. Reports 016–018 silently
  swapped the headline mid-Phase-2 and the Phase 2 baseline was never
  re-established in the new metric space. The "completion" framing exists in
  notes; the running CSVs do not yet measure it.
- **CP9 (headline)** has drifted twice on the Phase 5 line: design spec ΔE
  (line 282–297) → STATUS banners briefly described K-branch
  `state_divergence` as the headline (CLAUDE.md `2026-05-09:75` rule was
  itself created in response to this) → Phase 4 D1 metastability headline
  silently replacing the original R@K + cap-coverage exit criterion in the
  Phase 4 checklist.
- **CP10 (bind-vs-bundle discovered)** is currently *declared*: bundle-first
  scene memory hard-codes the scene as the storage unit. This is a
  pragmatic, well-motivated declaration (Report 067) — but it is a declaration,
  not an emergent dynamic, and §4-P12 below flags this as the deepest
  principle tension on the current chain.

---

## §2. Branch landscape (Phase 1.b)

| Branch | Δ vs main | Role |
|---|---|---|
| `codex/phase5-prime-bundle-first-scene-memory` | +75/0 | tip of phase 5′ chain |
| `codex/phase5-prime-broader-followup-scope` | +69/0 | predecessor (same chain) |
| `phase5-m1-role-energy-stack` | +69/0 | identical SHA to above |
| `codex/phase5-prime-nonsynthetic-native-preflight` | +62/0 | earlier predecessor |
| `main` | — | merged phase5-session-close on 2026-05-24 |
| `product/grounded-ideation-spike-002` | +7/-23 | side quest, ignored |

Only one storyline. Phase 2 = one branch-survey subagent's scope.

---

## §3. Phase-5′ chain survey (subagent return, condensed)

**Built (still load-bearing in `src/`)**
- `src/energy_memory/phase5/bundle_first_scene_memory.py` (Report 099)
- `src/energy_memory/phase5/natural_source_protocol.py` (Report 094)
- `src/energy_memory/phase5/bridge_readouts.py` — `cue_conditioned_scene_energy_v1` (Report 105)
- `src/energy_memory/phase4/range_shaped_replay.py` — `RangeShapedReplaySampler` (Report 069)
- Cleaned natural-source gate (Report 093, candidate ≈ 0.9512, controls ≤ 0.0010)

**Closed paths (mechanism / report / why)**
- M1 retrieval stack (P1+D3+P3) — Report 064 — null, `ΔE ≈ -0.32`, `hit_role = 0`
- Raw scene-MHN bridge `raw_scene_energy_v0` — Reports 101–104 — degenerate
  saturation at `E = -1.0`
- Cue-conditioned bridge `cue_conditioned_scene_energy_v1` — Reports
  105–107 — non-saturated but min-branch attractor collapse
- Strict discriminator (probes 29, 110, 220, 331) — Reports 108–109 — max
  mean `ΔE = 0.00217 < 5.5e-3` floor; role does not separate from random
- Range-shaped downstream lane (`range_postsettle` and `range_presettle`) —
  Reports 110–111 — novelty without retrieval; post-settle stores
  near-duplicates, pre-settle preserves d_eff but top1 does not move

**Dead-but-revivable ideas**
- **Range-presettle insertion** (Report 074 positive control) — synthesized
  queries are novel pre-settle; the problem is post-settle Hopfield cleanup
  erasing the novelty. Revivable if the **insertion path** is changed
  (Phase 4 gap P8 below), not the sampler.
- **Passive replay-observed context source** (Reports 082–086) — degraded
  but **real** signal vs controls; failed on role-universe coverage and
  learned-token geometry. Revivable with pre-filtered role-universe
  matching.
- **GHRR matrix-binding** (Report 066) — null on single-role MQAR on Haar
  keys; **untested on learned (non-Haar) keys and multi-role**. Not actually
  falsified in the regime the project cares about.

**Reusable infra still untapped**
- `m1_role_energy.py` — defined; not exercised post-Report 064
- `ham_with_layer2.py`, `ham_aggregator.py`, `role_fidelity.py` — pre-committed;
  no headline-scope reports route through them
- `RangeShapedReplaySampler` optional atom-support smoothing — never tuned
- Experiment 43 (`range_shaped_replay_gate.py`) — sampler-algorithm-validated;
  no downstream integration experiment after Report 111

**Principle tensions surfaced by the chain**
- T1. Cleaned-protocol `freq ≤ 32` cap (Report 093) is an off-line
  controller-style filter applied to recover clean controls. CP7 risk.
- T2. Scene is the storage unit by declaration; Report 075 used random
  unique scene IDs as anchors and that *worked* — Report 076 replaced them
  with substrate-derived bundles, but the storage structure still
  pre-selects scene identity. CP10 tension.
- T3. ΔE headline floor (5.5e-3) was set when "queries with clear
  structural separation would exist at n≥10" was the assumption; at the
  cleaned-discriminator-strict-query n=1 setting it is below noise floor.
  CP9 (headline integrity) needs re-grounding.
- T4. `bridge_readouts.py` shape — picking which energy reads out the
  decision — risks importing arbitration unless the readout is a property
  of the dynamics, not a chosen scoring function. CP7 watch.

---

## §4. Per-phase gap analysis

Severity: **B** = blocker for Phase 5; **F** = friction; **N** = nice-to-fix.

### §4.1 Phase 1 — Scaled Substrate Validation
- **[B] D=4096 bind/unbind roundtrip never tested.** All FHRR tests stop at
  D=256/512 (`tests/test_fhrr.py`, `test_torch_hopfield_optional.py`).
  Phase 5 bundle-first chains 16 binds at D=4096. → unblocks: every Phase 5
  bridge attempt that *might* be failing because the substrate's algebra
  has silent accumulated error at the production dimension.
- **[B] Bundling capacity / recovery never quantified at D=4096.** Exit
  criterion in the project plan; no test exists. Context bundling
  (Reports 076–081) lives at D=4096. → unblocks: silent
  context-source degradation that has been chased through 6+ reports.
- **[F] `alpha_anti` (dimensionality-preserving repulsion) wired but
  unbenchmarked.** Defaults 0.0; one test suite (`test_phase5_ab_death_dynamic.py`)
  at D=512. → unblocks: confidence in d_eff stability during
  Phase 4 consolidation at production scale.
- **[F] Single-vector bind/unbind MPS hotspot (Report 014, ~100× overhead)
  not addressed.** May hit Phase 5 hot paths invisibly. → CP6 / latency.
- **[F] Pure-Python reference not exercised in Phase 5 path.** No parity
  test between pure-Python and Torch bind/unbind at matched D / noise.
  → CP-non-negotiable violation by attrition.

### §4.2 Phase 2 — Static Contextual-Completion Baseline
- **[B] No per-query energy readout shipped.** Phase 2 computes
  `memory.energy()` per cell but aggregates and discards per-sample values.
  Phase 5 needs `E(cue)` and `E(final)` per probe to compute paired ΔE.
  → unblocks: the entire bridge work. Bridge readouts in
  `bridge_readouts.py` have **no Phase 2 baseline to validate against**.
- **[B] Energy conventions never validated against per-query behavior.**
  Phase 2 noted `mean_energy` saturating at `-1.0` at high β but never
  treated this as a measurement-design finding. Reports 103–104 then
  rediscovered the saturation as a Phase 5 blocker. → CP9 + bridge.
- **[F] Phase 2 baseline was unstable across seeds (0.205 single-seed →
  0.089 [0.060, 0.125] multi-seed) and was never re-established under the
  post-Report-018 metric pivot.** → headline integrity (CP9).
- **[F] cap_coverage / metastable_rate defined in Phase 2 but **never
  ported** as drill-downs into Phase 5.** Phase 5 uses scene_tix/content_tix/
  candidate top1 only. → CP9 drift; loss of falsification capacity.
- **[N] L ∈ {64, 256, 1024} and β ∈ {3, 10, 30, 100} swept long after L=64,
  β=10 became the operating envelope.** Matrix bloat; no impact on Phase 5.

### §4.3 Phase 3 — Growing Codebook

**Per-knob status (the 5 zero-default `ConsolidationConfig` knobs)**
| Knob | Report | Verdict | Phase 5 inherits |
|---|---|---|---|
| `alpha_freq_lambda` | 040 | empirically null | noise (no value) |
| `coverage_lambda` | 045, 046, 054 | integrated | clean signal at λ=1.0 |
| `retrieval_weight_epsilon`/`_tau` | 046, 054 | integrated, **inert** on A1′ substrate (uniform bias across 1063/1064 atoms cancels ΔE to zero) | noise |
| `metastability_obs_rate` | — | **never exercised**; default 0.0 leaves replay-prioritization signal unfired | nothing |
| `inhibition_gain` | 034 | integrated, seed-1 only, never 5-seed verified | partial signal not in A+B path |

- **[B] NC1 / separability pair never computed in the live loop.** Spec'd
  in `phase-3-deep-dive.md:197-204` and `2026-05-09:111-117`. Phase 5
  cannot tell if structural retrieval failure is "atoms collapsed" vs
  "atoms diffuse" vs "atoms polysemous." → unblocks every bridge debugging
  cycle that currently can't localize a root cause.
- **[B] Bimodality tracking never wired.** Spec'd; no implementation;
  Phase 5 splitting decisions are blind. → CP10 (bind-vs-bundle discovered).
- **[B] Phase 3 exit criterion not met.** Report 017: learned codebook
  Recall@1 = 0.104 [0.071, 0.141] vs random 0.089 [0.060, 0.125] —
  CI-overlapping. Phase 5 inherits a codebook that doesn't
  measurably beat random at the operating envelope. → unblocks: every "is
  it the substrate or is it the readout?" question Phase 5 has asked.
- **[F] Regime classifier (d̄, d_eff) runs post-hoc only.** Live
  consolidation has no regime label; θ′(β) uncalibrated.
- **[F] Genuine shuffled-token control never run.** Random baselines in
  Report 017 are artifact reuse.
- **[CP7 violation, latent]** `phase-3-deep-dive.md:234-245` enumerates
  failure-mode fixes in **if/then** form ("if NC1→0, increase repulsion
  strength; if chaotic, decrease learning rate"). These are controller-style
  interventions waiting to be added if Phase 3 instability appears at
  scale. Pre-empt by re-expressing each as a slow-timescale dynamic per
  CP8 *before* they are coded.

### §4.4 Phase 4 — Replay and Consolidation
- **[B] `TrajectoryTrace.encoder_terms` schema exists but is never
  populated at capture time.** Phase 5 (Reports 087–093) was forced to
  reconstruct provenance from synthetic sources. → unblocks: every
  "non-synthetic native provenance" lane.
- **[B] Insertion path silently erases novelty.** Range-shaped replay
  generates novel `(role, atom)` provenance (Report 068 KL 2.08→0.013) and
  novel pre-settle vectors (Report 074), but post-settle Hopfield
  cleanup collapses them into existing basins (Report 110–111
  `stored_near = 1.000`). Sampler is sound; **the consolidation step is
  the bottleneck**. → unblocks: every replay-based Phase 5 intervention.
- **[F] Replay scheduler is fixed-cadence (`_retrieval_count % K == 0`).**
  Design says "later replaced with tension-driven" — never done. Not a
  controller; the opposite — *under-engineered*, no local-tension signal.
  CP7 + CP8 directly motivate the fix.
- **[F] Pattern death mechanism never fires at canonical config.** Report
  036: >99.9% of atoms never reinforced; death waits on reinforcement.
  Stale atoms accumulate invisibly.
- **[F] Headline drift inside Phase 4.** Original headline = R@K +
  cap-coverage; checklist pivoted to D1 metastability silently. Exit
  criterion E3 (entropy drops with replay) failed and was deferred without
  diagnosis. CP9.
- **[N] `tag_overlap`, `suppression_decay`, `recovery`** wired
  default-off; not ablated; adds tuning surface.

---

## §5. Phase 5 problem inventory

### Explicit (in design docs / STATUS / checklist)
- **P1. ΔE bridge readout.** Translate scene-MHN basin success into
  `ΔE = E_content-prior − E_role-prior` with mean ≥ 5.5e-3 and CI strictly
  above zero. Current best (strict discriminator, 4 probes): 0.00217.
  Causes:
  - C1.a. Raw scene energy saturates at `-1.0` by construction.
  - C1.b. Cue-conditioned energy avoids saturation but exhibits min-branch
    attractor collapse (118/120 branches → target scene).
  - C1.c. Floor was set on a saturating quantity (the headline scoring
    convention itself is the issue, not the substrate).
- **P2. M2 training-time intervention.** EqProp + role-shuffled-negatives +
  DSM warm-start on existing FHRR. Never run. Heaviest open path
  (~2–3 weeks).
- **P3. Range-shaped replay downstream.** Closed for current mechanism
  (Reports 110–111). Needs fresh precommit; not scale-up.
- **P4. n≥10 verified ΔE headline.** Nothing has reached this bar on any
  path.
- **P5. Bundle-first matrix at K_roles=16, full N×noise grid.** A2/A3 in
  `phase-5-prime-checklist.md` partial; full matrix open.

### Inferred (surfaced by §3–§4)
- **P6. (Substrate) D=4096 bind/unbind fidelity unverified.** §4.1 [B].
- **P7. (Energy convention) Per-query energy readout never operationalized
  in Phase 2.** Phase 5 bridge work is computing a quantity that has no
  Phase 2 ground-truth distribution. §4.2 [B].
- **P8. (Codebook diagnosis) Phase 3 atoms carry no live NC1 /
  separability / bimodality / regime labels.** Phase 5 cannot localize
  "structural retrieval fails" between substrate, codebook, replay,
  readout. §4.3 [B]. *Inferred but well-supported.*
- **P9. (Insertion path) Post-settle Hopfield cleanup erases novelty
  before Phase 5 can use it.** §4.4 [B].
- **P10. (Provenance plumbing) `TrajectoryTrace.encoder_terms` never
  populated; bundle-first scene assembly is forced onto synthetic
  sources.** §4.4 [B].
- **P11. (Headline / CP9 drift) Phase 5 ΔE floor sits on a saturating
  quantity, has no Phase 2 precedent in the same metric space, and the
  Phase 4 headline silently pivoted from R@K to D1.** §3 T3; §4.2; §4.4.
- **P12. (CP10 drift) Scene as storage unit is *declared*, not
  *discovered* from co-occurrence statistics.** §3 T2. Deepest tension —
  bundle-first works (Report 093), but it works on a structure that the
  project's design principles say should emerge, not be hard-coded.
- **P13. (CP7 watch) Cleaning-protocol `freq ≤ 32` cap and the broader
  "choose a bridge readout" pattern in `bridge_readouts.py` are
  borderline arbitration**. §3 T1, T4.

---

## §6. Idea generation

For each problem: 2–4 candidate approaches, **at least one boring, at least
one weird**, each traced to a Core Principle (CP#). "Depends on" lists
Phase 3 gaps (P6–P13) that must be closed first.

Rough ordering: P11 (headline) and P6 (substrate fidelity) gate everything
else, so they go first.

### Ideas for P11 — headline integrity (and by extension P1 cause C1.c)

- **I-11.a (boring).** Reset the headline to a **non-saturating energy
  gap**: the post-settle residual `||q* − Π(q*)||²` where `Π` is the
  scene-MHN projection. The current `E = -1.0` is "the projection error is
  zero." Use the *unprojected* residual energy of the *unbound content
  estimate* against the content-MHN as the readout, since that is where
  there is actual headroom. CP9 (headline ↔ measurable headroom).
  Depends on: nothing new; uses existing machinery.
- **I-11.b (boring).** Pair the headline against a same-substrate
  random-prior **null distribution**, not a fixed floor. Floor `5.5e-3`
  was an *a priori* guess; replace it with the upper CI of a deranged-role
  null on the same probes. CP9 + non-negotiable rule #6
  (no mechanism without control). Depends on: nothing.
- **I-11.c (weird).** **Make the headline a basin-shape statistic, not an
  energy.** Use NC1-style within-basin variability of the **role-prior
  branches' final states** vs **content-prior branches' final states** on
  the same cue. If role priors land in a tighter basin than content priors,
  that *is* the structural-retrieval signal — and unlike `E_min`, it is
  not saturating. CP3 (meaning lives in the retrieval neighborhood) +
  CP9. Depends on: **P8** (NC1 instrument must exist live).
- **I-11.d (weird, philosophical).** Reframe Phase 5 graduation away from
  "ΔE on held-out cues" entirely and toward **"does the basin under a
  role-prior contain the right content atoms at a higher density than
  under a content-prior?"** — a density / probability-mass question on
  the settled state, not an energy scalar. CP3 + CP5 (latent reasoning).
  Depends on: **P8**.

### Ideas for P6 — D=4096 substrate fidelity

- **I-6.a (boring).** Add bind/unbind roundtrip + bundling capacity tests
  at D=4096 with K_roles ∈ {1, 4, 16}. Single PR, ≤1 day. CP10 +
  non-negotiable rule "do not remove pure-Python reference" (add the
  parity test the rule has been implicitly relying on).
- **I-6.b (boring).** Parity test between pure-Python and Torch FHRR at
  matched D / noise / batch. CP6, CP10. Re-arms the reference backend.
- **I-6.c (weird).** Treat substrate noise as an **explicit term in the
  headline** rather than a hidden floor. Measure `ΔE` modulo a substrate
  noise envelope computed from a randomized-key control on the same
  cues — i.e., the noise floor *is* part of the metric. CP9. Depends on: nothing.

### Ideas for P1 — bridge readout (after P11 reset)

- **I-1.a (boring).** Replace `min(E_branch)` with **soft-min over branches
  with weights from the energy distribution itself** (logsumexp-style).
  Removes min-branch attractor collapse without adding arbitration.
  CP7 (the readout is a local dynamic over the branches, not an
  arbitration). Depends on: nothing.
- **I-1.b (boring).** Use **per-probe paired Welch on `mean(E_branch)`**
  instead of `min(E_branch)`. Mean is non-saturating because branches
  differ in margin even when each individually saturates against its own
  scene. CP9. Depends on: nothing.
- **I-1.c (weird).** Replace the energy readout with a **basin-residence
  time** under low-temperature settling: how long does the cue-conditioned
  trajectory stay within ε of the role-prior branch's basin vs the
  content-prior branch's basin? Time is non-saturating and is naturally
  a local dynamic. CP7 + CP8. Depends on: **P6**, **P9**.
- **I-1.d (weird, lateral).** Reframe the bridge as a **two-stream
  competition** that runs on the same substrate without an external
  ΔE comparator: store both role-prior and content-prior priors in one
  composite landscape and let the settling pick. The "decision" then is
  geometry, not a metric subtraction. CP1, CP7 (anti-homunculus exact
  hit). Depends on: anti-homunculus review (review block at the top of
  CLAUDE.md says "any new mechanism that reads a metric and acts on it"
  triggers it).

### Ideas for P8 — Phase 3 codebook diagnostic stack

- **I-8.a (boring).** Wire NC1 + inter-basin separability + bimodality
  (Hartigan dip / GMM-BIC) as **passive logs** in the live consolidation
  loop. One PR, mostly already specified in
  `consolidation-geometry-diagnostic.md`. CP9. Depends on: nothing.
- **I-8.b (canonical move from CP8).** Re-express each diagnostic as the
  fast-timescale snapshot of a slow-timescale dynamic per
  `2026-05-09:140-148`. Example: NC1 → bounded within-basin variability
  as a soft anti-collapse term in consolidation; separability →
  inter-basin repulsion already partially wired via `alpha_anti`. This is
  the operationalization the 2026-05-09 note prescribed but was never
  done. CP7, CP8. Depends on: I-8.a.
- **I-8.c (weird).** Promote the regime classifier (d̄, d_eff) to a
  **per-atom property** computed online (cheap with EMA over recent
  consolidation events). Atoms then carry their own regime label,
  Phase 5 retrieval can stratify by it, and the choice of regime is not
  a controller's — it is the atom's local geometry. CP1 + CP10.
  Depends on: I-8.a; also surfaces empirical θ′(β) calibration as
  worthwhile.

### Ideas for P9 — insertion-path novelty erasure

- **I-9.a (boring).** Add a **pre-settle insertion mode** to
  `UnifiedReplayMemory` and run the F9 lane again. Reports 110–111 closed
  *post-settle* as not viable; pre-settle was only a positive control in
  Report 074. CP4 (replay reshapes the landscape — but only if it can
  insert off-basin patterns). Depends on: anti-homunculus review (no
  if/then about which mode to use; it has to be a substrate-level choice).
- **I-9.b (weird).** Replace "post-settle Hopfield clean → insert" with
  **store the *trajectory* (the q→q* path) rather than the endpoint q*.**
  Replay then re-runs the path; novelty lives in the path geometry, not in
  endpoint coordinates that collapse to existing basins. CP4 + CP5 + the
  whole point of `TrajectoryTrace`. Depends on: **P10**.
- **I-9.c (weird).** Treat post-settle near-duplicates as **evidence of a
  basin needing to split**, not as a storage failure. Plug into P8's
  bimodality signal: persistent near-duplicate insertions are a
  splitting-tension on the receiving basin. CP10 (bind-vs-bundle
  discovered). Depends on: **P8**.

### Ideas for P10 — provenance plumbing

- **I-10.a (boring).** Populate `TrajectoryTrace.encoder_terms` at capture
  time during settling — schema exists, capture point missing. One PR.
  CP2 (memory is the self — it has to know what it stored).
- **I-10.b (weird).** Stop carrying `(role, atom)` tuples as separate
  schema and instead **store the actual bound hypervectors** as the
  trace, letting Phase 5 re-derive role/filler via algebraic unbind from
  the trace itself. Fewer schemas to maintain, harder to corrupt. CP10.
  Depends on: **P6**.

### Ideas for P2 — M2 training-time intervention

- **I-2.a (boring).** Run M2 as currently specified (EqProp +
  role-shuffled-negatives + DSM warm-start) once P6, P8, P11 are closed,
  so a null result can be **localized** rather than absorbed into
  substrate/codebook ambiguity. CP4. Depends on: **P6, P8, P11**.
- **I-2.b (weird).** Treat training-time intervention not as an explicit
  EqProp pass but as a **consolidation-pressure modulation**: per-atom
  coverage_lambda / repulsion already drives codebook geometry; modulate
  *those* by role-shuffled-negative loss residuals during replay
  ("retraining without a separate training mode"). Looks like CP4
  continuous learning by construction. Depends on: **P8, P9**.

### Ideas for P12 — scene-as-declared (CP10 drift)

- **I-12.a (boring, accept the drift).** Document scene-as-storage-unit
  as a **deliberate Phase 5 simplification** to be revisited in Phase 6,
  with explicit user-approved deviation from CP10. Cheapest move; lets
  current work continue. CP9 (be honest about scope).
- **I-12.b (weird).** Replace declared scenes with **emergent attractor
  scenes** from Phase 4 trajectories: cluster `TrajectoryTrace` final
  states and let scene identity be the cluster label that emerges from
  consolidation. CP4 + CP10 + the whole point of CP-non-negotiable rule #1
  (no module decides which subsystem wins; the scene is whatever
  consolidation makes stable). Depends on: **P8, P9, P10**. Heavy.
- **I-12.c (weird, lateral).** Drop "scene" as a unit entirely; do
  bundle-first over **windows of trajectory** instead — the bundle is the
  natural temporal granularity the system already produces. CP3 (the
  question is "what fills this unresolved gap"; the bundle of recent
  trajectory *is* the gap-filler context). Depends on: **P10**.

### Ideas for P13 — anti-homunculus watch

- **I-13.a (boring).** Run the `anti-homunculus-reviewer` agent on
  `bridge_readouts.py` and on the cleaning protocol's `freq ≤ 32` cap
  before any further work. CLAUDE.md's review block already mandates this
  before any "metric-reading and acting" mechanism lands; the chain has
  slipped past it. CP7. Depends on: nothing.
- **I-13.b (weird).** Reframe the `freq ≤ 32` cleaning cap as a
  **basin-mass cap** that is a property of the codebook (atoms with
  basin mass above the cap are saturating attractors and naturally
  dominate any bridge readout; the cap is *measurable*, not chosen).
  CP7, CP8. Depends on: **P8**.

---

## §7. Cross-cutting observations

- **The chain has been doing readout debugging for 6+ reports while a
  saturating energy is the actual problem.** Reports 100–111 are a
  textbook case of CP9 drift — the headline metric *itself* was outside
  the questioning frame. STATUS line 8 ("next work needs a fresh Phase 5′
  precommit or a user-approved objective change") finally names this,
  but the user-approved objective change has not been spelled out
  anywhere. TL;DR #1 is the proposal that this should be the next
  precommit.
- **Three "novelty without retrieval" lanes (Reports 074, 086, 111) all
  share the same root cause: post-settle insertion erases off-basin
  vectors.** This is a single Phase 4 fix (P9), not three separate
  Phase 5 problems.
- **Every "Phase 5 inherits noise" finding traces back to a Phase 3
  diagnostic that was specified in 2026-05-09 / phase-3-deep-dive but
  never implemented (NC1, separability, bimodality, regime classifier,
  θ′(β) calibration).** The 2026-05-09 directive — "the session that
  crosses this threshold safely needs to do two things" — never happened.
  TL;DR #3.

---

## §8. What I am not asserting / what could be wrong

- The TL;DR #1 reframing of the headline is **a proposed objective
  change** and requires user approval per STATUS line 8 and CLAUDE.md
  preamble rules. The spec headline at `phase-5-unified-design.md:282-297`
  is still the binding spec until that approval lands.
- The phase-by-phase severity tags are my reading of the gap analyses;
  reasonable people could re-rate. The blockers I'm most confident in
  are P6 (D=4096 substrate fidelity) and P9 (post-settle insertion) —
  these have direct test/report evidence. P8 (live NC1/separability) is
  a strong inference but I have not run the diagnostic to confirm
  collapse is happening on the current codebook.
- I have not independently re-run any experiment. All "verdict" entries
  in §4.3 are read from reports surfaced by grep; if any of those reports
  have been quietly superseded, the verdicts move.
- The product/grounded-ideation-spike-002 branch was excluded by user
  direction; if it contains an idea worth reviving for P12 (emergent
  scene), it would belong in §6 P12.b/P12.c and I cannot see it.

---

## §9. Post-write verification (2026-05-26)

Three load-bearing claims were spot-checked by independent subagents
after the audit was written. Results:

- **§4.1 [B] D=4096 fidelity gap — TRUE.** `tests/test_fhrr.py:7-14` and
  `tests/test_torch_fhrr_optional.py:5-16` use D=256; one bundle test at
  D=512. No bundling-capacity test at D=4096. Production runs at D=4096
  (`experiments/44_phase5_prime_bundle_first.py:933`,
  `src/energy_memory/substrate/torch_fhrr.py:27`). Caveat:
  `compute_role_fidelity` runs at D=4096 as a metric, not a
  substrate-roundtrip assertion.

- **§4.4 [B] / §7 post-settle insertion erasure — TRUE with refinement.**
  Insertion at `src/energy_memory/phase4/replay_loop.py:631-650` stores
  `trace.final_state`; pre-settle vs post-settle is hard-coded, no
  config knob. **The collapse happens during re-settling, not during
  insertion** (Report 074: `query_near=0.269` → `final_near=0.977`). And
  Report 111 shows pre-settle insertion preserves novelty but does **not
  move retrieval**. So fixing the insertion path is **necessary but not
  sufficient** — the audit's framing was correct in mechanism but
  understated the depth: the Hopfield dynamics themselves are doing the
  erasure, and the off-basin novelty is too weak to survive even when
  stored pre-settle. P9 cost estimate raises accordingly.

- **§5 P1.C1.a / TL;DR #1 saturation claim — PARTIALLY TRUE; conflates
  two failure modes.** `raw_scene_energy_v0` does saturate at `-1.0` by
  construction (`memory/torch_hopfield.py:188-201`); Report 104
  unambiguously diagnoses this. **However**,
  `cue_conditioned_scene_energy_v1`
  (`src/energy_memory/phase5/bridge_readouts.py:17-36`,
  `E = -Re(<q_scene, cue>)/D`) is **already non-saturating** — it scores
  against the cue, not against stored patterns. Report 107 finds
  `ΔE = 0` on this readout anyway, because of a **second, distinct
  failure mode**: min-branch attractor collapse (118/120 branches and
  36/36 min selections land on the probe's target scene regardless of
  condition). Implication: TL;DR #1 must be **two moves**, not one — a
  non-saturating *readout* AND a non-min *aggregator*. Pivoting the
  readout alone leaves the min-branch collapse intact.

  Audit-level correction: §5 P1's `C1.a` (saturation) and `C1.b`
  (min-branch collapse) are co-equal, not cause-and-effect. The TL;DR
  recommendation as written would solve `C1.a` only. Treat them as two
  blockers and address both.
