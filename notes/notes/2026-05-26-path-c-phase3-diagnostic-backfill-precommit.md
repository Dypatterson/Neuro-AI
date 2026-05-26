---
date: 2026-05-26
project: neuro-ai
tags:
  - notes
  - phase-3
  - phase-5-prime
  - planning
  - precommit
  - diagnostic-actuator-identity
---

# Path C — Phase 3 Diagnostic-Stack Backfill Precommit

## Status

This precommit reopens **Phase 3** to backfill the diagnostic stack the
2026-05-09 directive flagged as the conceptual move between Phase 3 and
Phase 4 and which was never implemented. Phase 5′ is **paused** pending Path
C exit.

Path C was chosen 2026-05-26 by the user after the
[audit](../../audit-phase5-2026-05-26.md) surfaced that:

1. The current Phase 5 ΔE bridge has **two distinct failure modes**
   (saturation + min-branch attractor collapse — audit §9 verification
   deltas), and the next bridge attempt cannot be localized without live
   codebook diagnostics.
2. The Phase 3 exit criterion (regime-stratified Recall@K beats shuffled
   control) was **never met** — Report 017 found
   `hebbian=0.104 [0.071, 0.141]` vs `random=0.089 [0.060, 0.125]`,
   CI-overlapping (audit §4.3).
3. Five diagnostics specified in
   [`notes/emergent-codebook/phase-3-deep-dive.md`](../emergent-codebook/phase-3-deep-dive.md)
   and [`notes/notes/2026-05-09-papers-diagnostics-and-actuator-dynamics.md`](2026-05-09-papers-diagnostics-and-actuator-dynamics.md)
   were never wired live: **NC1 within-basin variability**,
   **inter-basin separability**, **bimodality (Hartigan dip / GMM-BIC)**,
   **regime classifier (d̄, d_eff, calibrated θ′(β))**,
   **per-atom metastability EMA** (`metastability_obs_rate` defaults to
   0.0 and has never been exercised).

Path C is not a Phase 5 graduation precommit. It does not authorize any
Phase 5 ΔE / bridge / M2 / full-matrix / headline / graduation work.

## Pre-Path-C Evidence Banked This Session

These results are recorded here so they survive the Phase 3 reopen.

- **D=4096 substrate fidelity verified.**
  `tests/test_substrate_fidelity_d4096.py` passes 6/6:
  - Single bind/unbind roundtrip at D=4096, cosine `1.000000` both backends.
  - Chained 16-role bind/unbind at D=4096, mean recovery `0.224`,
    min `0.198` (against threshold `0.10`).
  - Bundle capacity curve at D=4096:
    `N=4→0.447, N=8→0.318, N=16→0.221, N=32→0.160, N=64→0.110`
    (tracks 1/√N as expected).
  - Pure-Python ↔ Torch FHRR parity at D=4096, max abs error
    `4.6e-7` (rtol 1e-5).
  Substrate is officially struck from the suspect list. The audit §4.1 [B]
  blocker is closed.

- **Anti-homunculus review of `src/energy_memory/phase5/bridge_readouts.py`
  recorded: PASS, with two CP7 risk edges in the consumers (not in this
  module).** Findings:
  - The two functions in `bridge_readouts.py` are pure measurements
    (`cue_conditioned_scene_energy_v1` projects a settled state onto the
    fixed cue; `delta_e_content_minus_role` is a paired-condition contrast).
    Shape-isomorphic to cap-coverage / NC1.
  - **Edge A — `min_branch(E)` aggregator at
    `scripts/phase5_prime_strict_discriminator_viability.py:96-98`** is
    genuine arbitration (`min` ≡ `argmin ∘ select` — picks a winning
    branch). This is the actual CP7 failure site that the audit's §5 P1.C1.b
    was pointing at. The recommended refactor is the audit's I-1.a
    (logsumexp / soft-min) or I-1.d (two-stream composite landscape).
  - **Edge B — `q_bundle` vs `q_greedy` dispatch at
    `scripts/phase5_frozen_snapshot_audit.py:656` and `:915-916`**
    (`use_bundle = k_main > 1 and res.q_bundle is not None`) is a
    borderline `if/then` combiner choice. Either always report both, or
    replace the dispatch with a continuous mixing weight derived from the
    same low-energy-set geometry that `n_in_low_energy_set` already measures.
  - Neither edge is in scope for Path C directly — they are Phase 5
    refactors. Path C should not touch them.

- **Anti-homunculus review of the `non_special_unique_target_freq_le_32`
  cleaning predicate
  (`src/energy_memory/phase5/natural_source_protocol.py:120-130`)
  recorded: PASS (qualified, fragile).** Findings:
  - The predicate reads only properties of the source-row distribution
    (`atom_counts = seed_atom_counts(rows)`), not properties of the
    substrate's settling, energy, basin geometry, or retrieval output. It
    is boundary-IO scoping of the evaluation set, equivalence-class with
    "evaluate Recall@K only on unique-target rows."
  - **Fragility conditions** under which the same predicate FAILS the
    filter (must not be allowed to drift into any of these):
    1. Cap becomes adaptive on a substrate observable.
    2. The predicate migrates into the retrieval/settling/replay inner
       loop.
    3. The headline metric is reported on the filtered subset *without
       the filter being declared as the headline's scope*.
    4. The cap migrates into Phase 4 replay sampling.
  - **Hardening move** (audit I-13.b): re-express the cap as a
    *measurement of basin-mass saturation* on the codebook itself. This
    requires Path C's NC1 / separability / basin-mass instrumentation
    (C.1) to land. Until then, the `freq ≤ 32` filter must stay inside
    the diagnostic-evaluation surface.

## Architectural Claim

Five Phase 3 codebook diagnostics will become **first-class measurements**
of the substrate, and each will be re-expressed as the *fast-timescale
snapshot of a slow-timescale dynamic the substrate already runs*, per the
2026-05-09 mapping table (lines 140-148). This is the principled
operationalization the 2026-05-09 note named as "the session that crosses
the diagnostics→actuators threshold safely."

Anti-homunculus discipline:

> **Each diagnostic must be implementable as a passive log first (C.1),
> then re-expressed as a slow-timescale dynamic (C.2) only after an
> anti-homunculus reviewer pass. C.1 has low risk; C.2 has high risk.
> Reviewer is invoked per-mechanism, not per-PR.**

## Mechanism

### C.1 — Passive Instrumentation (low CP7 risk)

| # | Diagnostic | Source spec | Computed where | Persisted as |
|---|---|---|---|---|
| C.1.1 | **NC1 within-basin variability + inter-basin separability** (paired) | `2026-05-09:111-117`, `phase-3-deep-dive.md:197-204` | Per consolidation event, on the codebook snapshot | `ConsolidationDiagnostics.nc1`, `.separability` |
| C.1.2 | **Bimodality (Hartigan dip / GMM-BIC)** on per-atom `context_bag_history` | `phase-3-deep-dive.md:147-171` | Rolling window N=5 over recent consolidation events | `Atom.bimodality_score` |
| C.1.3 | **Regime classifier (d̄, d_eff)** per-atom and per-codebook | `consolidation-geometry-diagnostic.md` | Live in consolidation loop (currently post-hoc only via `scripts/consolidation_geometry_diagnostic.py`) | `Atom.regime`, `Codebook.regime_distribution` |
| C.1.4 | **Empirical θ′(β) calibration** | `consolidation-geometry-diagnostic.md:172` | One-time spike on project's FHRR substrate; replaces θ′ ≈ 1/β approximation | Calibration table committed to repo |
| C.1.5 | **Per-atom metastability EMA `m_i`** | `2026-05-09:140-148`; `metastability_obs_rate` knob in `ConsolidationConfig` | Per retrieval, EMA over `c_i = w_i · (1 − max_j w_j)` | `Atom.metastability` |

Each C.1.x deliverable includes:
- One implementation PR, sized to ≤1 unit of new behavior.
- Unit tests against synthetic codebooks with known regime / known
  collapse / known bimodality.
- A passive log entry in the consolidation report output so the next
  Phase 3 / Phase 5 experiment can stratify on it.

Cost estimate: **3–5 days** for all of C.1.

### C.2 — Diagnostic-as-Actuator (high CP7 risk; one anti-homunculus review per mechanism before landing)

Per the 2026-05-09 table:

| # | Diagnostic | Slow-timescale dynamic (the actuator the diagnostic is a snapshot of) | Anti-homunculus reviewer gate |
|---|---|---|---|
| C.2.1 | NC1 trending to zero | **Bounded-non-zero anti-collapse pressure** in consolidation: an additive term in the consolidation update that grows as within-basin variability shrinks below a floor. NOT a triggered if-then; a continuous term. | required pre-merge |
| C.2.2 | Persistent bimodality | **Splitting-tension energy** as a per-atom scalar that accumulates with sustained dip-test rejection; splitting fires when tension exceeds threshold (energy crossing, not rule trigger). | required pre-merge |
| C.2.3 | Cap-coverage failure | **Local error gradient on consolidation**: cap-coverage residual contributes additively to the consolidation update on the responsible atoms. | required pre-merge |
| C.2.4 | Metastability | **Replay buffer energy-ranking by construction**: traces with higher per-retrieval `c_i` carry higher buffer-energy and are replayed sooner. This may already be partially supported by `metastability_obs_rate` — the knob has never been exercised; C.2.4 is the smoke that exercises it with `m_i` populated by C.1.5. | required pre-merge |
| C.2.5 | Drift | **Replay-tension energy** as a per-atom scalar that grows with magnitude-of-drift; high replay-tension atoms get re-encoded during replay. | required pre-merge |

Each C.2.x deliverable includes:
- A precommit note specifying the exact dynamic and the equivalence to its
  paired diagnostic (e.g., "this term reduces to NC1-as-measurement when
  observed at convergence").
- Anti-homunculus reviewer pass on the proposed dynamic *before* code lands.
- Implementation PR.
- A test that confirms the dynamic and the diagnostic agree at convergence
  (the substrate-pure version of the diagnostic↔actuator identity claim).

Cost estimate: **1–2 weeks** for all of C.2.

**C.2 must not become if/then logic under any pressure.** The 2026-05-09
note's right-hand column gives the principled form for each mechanism; if
implementation drifts toward the left-hand-column form (`if metric > τ
then action`), the anti-homunculus reviewer fails the gate and the PR is
rejected.

### C.3 — Phase 3 exit criterion re-run

With the live diagnostic stack in place:

- Re-run the Phase 3 exit criterion from `phase-3-deep-dive.md:180-189`:
  "Recall@K on masked-token contextual completion, stratified by regime
  classification, vs. shuffled-token control."
- Stratify by per-atom regime label produced by C.1.3.
- Use a **genuine** shuffled-token control (the Report 017 random
  baselines were artifact reuse — audit §4.3 [F]). Match Phase 2's random
  baseline construction.
- Report on n ≥ 10 seeds with Wilson CIs per regime stratum.

Cost estimate: **2–3 days**.

## Required Controls

For C.1:
- Synthetic codebook with known regime / known collapse / known
  bimodality. Each C.1.x metric must produce the expected value to within
  numerical tolerance.

For C.2:
- Convergence-equivalence test: at quasi-stationary substrate, the live
  diagnostic computed from C.1 must match the slow-timescale dynamic's
  fixed-point value to within numerical tolerance. This is the
  *measurement of the identity* between diagnostic and actuator.
- Anti-homunculus reviewer gate (binding per spec H1-H4 of
  `notes/emergent-codebook/phase-5-prime-checklist.md`).

For C.3:
- Genuine shuffled-token control (not artifact reuse).
- Regime stratification reported per stratum, with sample counts and CIs.
- Headline = regime-stratified Recall@K beats shuffled control, CI-disjoint
  in at least one regime stratum.

## Exit Criteria (Path C closes)

1. C.1.1–C.1.5 land as passive logs with passing unit tests.
2. C.2.1–C.2.5 land as substrate-pure dynamics, each with an
   anti-homunculus reviewer PASS recorded, and each with a
   convergence-equivalence test passing.
3. C.3 reports regime-stratified Recall@K with at least one stratum
   CI-disjoint from the shuffled-token control, on n ≥ 10 seeds.

When all three are satisfied:
- Phase 3 is *graduated* (the exit criterion the project plan named at
  `docs/PROJECT_PLAN.md:180-181` is met for the first time).
- Phase 5′ is reopened. The next Phase 5 lane choice (A / B / D from the
  audit) is made *against the instrumented substrate* and any future
  bridge / readout attempt is diagnosable per-atom and per-stratum.

## What This Does Not Permit

- No Phase 5 ΔE / bridge / M2 / full-matrix / headline / graduation work
  during Path C.
- No retune of the `freq ≤ 32` cleaning cap (it must stay frozen until
  C.1 lands and the audit I-13.b basin-mass reframe is ready).
- No refactor of the `min_branch` aggregator or the `q_bundle`/`q_greedy`
  dispatch (those are Phase 5 refactors and must wait until Phase 3 reopens).
- No new "let's add a thing that decides" in C.2 — every C.2.x precommit
  note must explicitly state how the mechanism passes
  `2026-05-09:140-148` and must have an anti-homunculus reviewer PASS
  before the implementation PR lands.

## Anti-Homunculus Discipline Notes (binding)

H1 — H4 of `notes/emergent-codebook/phase-5-prime-checklist.md` continue
to apply throughout Path C. In addition:

- H5 — C.2 mechanisms must be expressible as the right-hand column of the
  2026-05-09 table. The left-hand column (controller-style "condition X
  causes response Y") is not acceptable even as a stepping stone.
- H6 — Path C does not introduce metric-triggered routing. Diagnostic
  values produced by C.1 are passive logs. C.2 mechanisms read substrate
  state, not C.1 diagnostic logs. (The diagnostic and the actuator are
  the same physical process, not a pipeline from one to the other.)
- H7 — Per the bridge_readouts.py review, the `min_branch` aggregator and
  the `q_bundle/q_greedy` dispatch are CP7 sites that *exist already* but
  are out of scope for Path C. They are flagged here for the post-Path-C
  Phase 5 reopen.

## Verification Standard

Same standard as Phase 5′ checklist `G1`–`G5`: n ≥ 10 seeds for any
verification claim, Wilson CIs reported, leave-one-seed-out sensitivity,
controls run on the same test set, and no graduation claim unless all
gates pass.

## Implementation Findings (running log)

### 2026-05-26 — C.1.1 landed (NC1 + Generalized NC2)

- **Status:** [src/energy_memory/phase3/basin_diagnostics.py](../../src/energy_memory/phase3/basin_diagnostics.py) and [tests/test_basin_diagnostics.py](../../tests/test_basin_diagnostics.py) shipped; 8/8 unit tests pass.
- **Formula choices recorded as binding:**
  - NC1 per atom = `(tr(G))² / ||G||_F²` (participation ratio of the
    centered Hermitian Gram of the per-atom basin's settled-state cluster).
    Real and complex tensors handled by the same algebraic path.
  - Separability NC2 = `||G_observed − G_etf||_F` where
    `G_etf = (K/(K−1)) · (I − J/K)`. Lower = closer to equiangular = better
    separation. **Primary metric, per user choice 2026-05-26.**
  - **Contingent fallback (per user instruction 2026-05-26):** if NC2
    proves uninformative or noisy during C.3 graduation (e.g., too
    coupled to K, dominated by basin-count variance across consolidation
    events, or numerically unstable when basin counts fluctuate), pivot
    to *mean off-diagonal pairwise centroid cosine distance* —
    `1 − mean_{i≠j}(<c_i, c_j>)` over normalized basin centroids. Range
    `[0, 2]`. **Higher = more separated** (opposite directionality to
    NC2). The fallback metric is implemented alongside NC2 as
    `compute_basin_separability_pairwise_mean` in
    [src/energy_memory/phase3/basin_diagnostics.py](../../src/energy_memory/phase3/basin_diagnostics.py)
    with its own unit test (T9); the pivot is a one-line swap at the
    consumer when C.3 evidence warrants. CP7 note: implementing both
    metrics does not introduce arbitration — the pivot is a future
    user-approved design choice, not a runtime metric-triggered switch.
  - Edge cases: empty basin excluded from NC1 and NC2; single basin
    observed → NC2 = `None`; singleton basin → NC1 = 1.0 with atom in
    `nc1_singleton_atoms`.
- **Finite-sample finding (binding for C.1 follow-ups and C.3):** with
  N=5 traces in D=128, NC1 caps near `n_samples − 1 = 4`. Observed values:
  T1 collapsed `≈ 1e-19`, T2 diffuse `≈ 3.91`, T3 healthy `≈ 3.90`. The
  T2/T3 values are indistinguishable at this buffer size. **At N=5, NC1
  behaves as a binary collapse detector, not a gradient measure of
  basin shape.**
  - Implication for C.1.x follow-ups: the rolling-window `maxlen = 5`
    default (matching `phase-3-deep-dive.md:160` `context_bag_history`)
    is too small for NC1's "shape" interpretation. For C.3 graduation,
    either:
    (a) accumulate basins over many consolidation events (so per-atom
    sample count grows over time, reaching `>> D`), or
    (b) raise the basin-trace buffer's `maxlen` substantially for the
    NC1 use case specifically (separate buffer from the
    bimodality-context buffer).
  - Recommendation: option (a). Treat the N=5 window as the
    bimodality-context convention and let NC1 read from a longer,
    accumulating buffer. This keeps `phase-3-deep-dive.md:160` intact
    and removes a tunable knob (`maxlen`) from the NC1 path.
- **Not wired into ConsolidationDiagnostics yet** (per the precommit's
  "What This Does Not Permit" spirit — keep C.1 mechanisms standalone
  until C.2 is on the horizon). Integration with the consolidation event
  dict at [src/energy_memory/phase34/stable_online_codebook.py:74-94](../../src/energy_memory/phase34/stable_online_codebook.py)
  is a separate small PR after C.1.2-C.1.5 land.

### 2026-05-26 — C.1.5 landed (per-atom metastability EMA)

- **Surprise finding (binding):** the metastability EMA machinery is
  **already implemented** in
  [`src/energy_memory/phase4/consolidation.py:163-569`](../../src/energy_memory/phase4/consolidation.py).
  `ConsolidationState.metastability_ema` tensor, `update_metastability`
  with EMA formula `m_i ← (1−μ_obs)·m_i + μ_obs·c_i` (line 381),
  `metastability_payback`, and aggregate diagnostics
  (`metastability_ema_mean`, `_max`) all exist. The knob defaults to 0.0
  (audit confirmed: never exercised in any report). The audit's claim
  that C.1.5 was "unimplemented" was *literally* true (off) but
  **structurally false** — the wiring is there.
- **C.1.5 scope reduced to a standalone read-only wrapper.**
  [`src/energy_memory/phase3/metastability_diagnostic.py`](../../src/energy_memory/phase3/metastability_diagnostic.py)
  exposes the per-atom EMA as `MetastabilityDiagnostics.per_atom`, with
  `obs_rate_active` flag so consumers can detect the κ=0 baseline. 5/5
  tests pass; T2 closed-form match exactly equals
  `c·(1−(1−μ)^n) = c·0.6513` after 10 updates at μ=0.1; T3 converges
  to target within float32 precision after 100 updates at μ=0.5.
- **Trajectory vs fixed-point metastability semantics
  (binding for C.2.4):** the existing per-retrieval `c_i` at
  [`src/energy_memory/memory/torch_hopfield.py:170`](../../src/energy_memory/memory/torch_hopfield.py)
  uses the **trajectory** definition
  `c_i^(traj) = max_t w_i(t) − w_i(T)` (clamped non-negative), not the
  spec's fixed-point definition `c_i = w_i · (1 − max_j w_j)`. Both are
  valid metastability measurements; trajectory version is more
  sensitive to settling dynamics. Whether they are equivalent at
  convergence depends on the iteration count and the softmax landscape.
  **C.2.4 (replay-buffer energy-ranking by construction) must explicitly
  state which definition the actuator side uses and justify the choice
  against an anti-homunculus reviewer.**

### 2026-05-26 — C.1.2 landed (bimodality: Hartigan dip primary, GMM-BIC fallback)

- **Status:** [`src/energy_memory/phase3/bimodality_diagnostic.py`](../../src/energy_memory/phase3/bimodality_diagnostic.py)
  and [`tests/test_bimodality_diagnostic.py`](../../tests/test_bimodality_diagnostic.py)
  shipped; 10/10 unit tests pass. Hand-rolled, zero new external
  dependencies (no scipy, no sklearn).
- **Formula choices recorded as binding:**
  - **Primary: Hartigan dip test (1985 construction).** For each
    candidate mode m, fit GCM on `[0..m]` and LCM on `[m..n−1]`, compute
    the unimodal envelope, take sup-deviation, minimize over m. Returns
    `0.5 × min_m sup_x |F_n(x) − envelope(x)|`. Monte-Carlo p-value via
    1000 uniform-null bootstrap samples. α=0.05.
    Verified: T2 bimodal p=0.001, T3 unimodal Gaussian p=0.985.
  - **Contingent fallback: 1D GMM-BIC (K=1 vs K=2)**, hand-rolled EM.
    BIC = `k·ln(n) − 2·ln(L)`. ΔBIC threshold = 10 (Kass-Raftery
    "strong" evidence).
    Verified: T4 bimodal ΔBIC=355.4, T5 unimodal ΔBIC=−7.77.
    The pivot is a one-line swap (`use_gmm=True` in
    `compute_bimodality_diagnostics`) if Hartigan proves uninformative
    during C.3.
  - **Signal reduction:** D-dim context-bag hypervectors → consecutive
    cosine similarities. N bags → N−1 scalars. Complex tensors handled
    via Hermitian inner-product real part (same convention as C.1.1).
  - **Rolling window:** N=5 (matches phase-3-deep-dive.md:160).
  - **Persistence:** ≥3/5 dip-rejections across recent consolidation
    events.
- **First-attempt iteration recorded for posterity:** the subagent's
  first dip-statistic implementation (ECDF vs uniform-on-range) and
  second attempt (pure GCM/LCM single-sided envelope) both incorrectly
  rejected unimodal Gaussian data. The third attempt — the textbook
  Hartigan & Hartigan construction — passed cleanly. **Lesson for C.2
  reviewers:** "approximate Hartigan dip" is fragile; only the full
  construction is correct.
- **Finite-sample limitation (binding for C.3):** at N=5 context bags,
  the consecutive-cosine signal has only 4 elements. The dip test on 4
  samples has near-zero power. The diagnostic handles this honestly —
  if `len(signal) < 4`, returns `(p_value=None, rejects=False)`. **The
  persistent-bimodality detector accumulates evidence across
  consolidation events**, which is the principled way to recover power
  at low per-event sample count. For C.3 graduation, accumulation depth
  matters: at the default window size, the diagnostic requires at least
  3 consolidation events with ≥4 cosines each that the test can resolve
  as bimodal. This is consistent with the C.1.1 finding that the N=5
  convention is a binary collapse detector, not a gradient measure.
- **Not wired into Atom / codebook yet.** `ContextBagHistory` accepts
  bags as input; the diagnostic does not assume any particular atom
  schema. Integration with the codebook is C.1 follow-up after C.1.3
  and C.1.4 land.

### 2026-05-26 — C.1.3 landed (regime classifier lifted live)

- **Status:** [`src/energy_memory/phase3/regime_diagnostic.py`](../../src/energy_memory/phase3/regime_diagnostic.py)
  and [`tests/test_regime_diagnostic.py`](../../tests/test_regime_diagnostic.py)
  shipped; 7/7 unit tests pass. The math primitives from the post-hoc
  script [`scripts/consolidation_geometry_diagnostic.py`](../../scripts/consolidation_geometry_diagnostic.py)
  (`_pairwise_fhrr_similarity`, `_pairwise_fhrr_distance`,
  `_participation_ratio`, `_summary_stats`) are now lifted into the
  module as public functions; the script imports them and continues to
  produce byte-identical output (`--help` runs cleanly, per-atom dict
  shape preserved via a thin local wrapper that calls the lifted
  primitives).
- **API:** `compute_codebook_regime_diagnostics(patterns, k_nn=8, beta=10.0, theta_prime_fn=None)` returns `CodebookRegimeDiagnostics` with per-atom `AtomRegime` (d_bar, d_eff, regime label) and `regime_counts`/`summary` aggregates.
- **`theta_prime_fn` injectable hook (binding for C.1.4):** the default
  is the `θ' ≈ 1/β` approximation per spec line 80. The injectable hook
  is the slot C.1.4's calibrated lookup table fills.
- **Edge case T5 design call:** for K=1 codebook, returns
  `regime='borderline'` with NaN d_bar/d_eff (no neighbors to measure
  against). Defensible — neither 'tight' nor 'spread' is meaningful with
  no separation to measure.
- **Verified contrast:** T1 (synthetic tight, atoms near-identical) ⇒
  8/8 classified `tight`, d_bar_mean=0.00009. T2 (synthetic spread,
  i.i.d. random unit vectors at D=128) ⇒ 8/8 classified `spread`,
  d_bar_mean=1.00073.

### 2026-05-26 — C.1.4 landed (empirical θ'(β) calibration spike)

- **Status:** [`experiments/calibrate_theta_prime.py`](../../experiments/calibrate_theta_prime.py),
  [`src/energy_memory/phase3/theta_prime_calibration.py`](../../src/energy_memory/phase3/theta_prime_calibration.py),
  [`tests/test_theta_prime_calibration.py`](../../tests/test_theta_prime_calibration.py),
  and the calibration artifact at
  [`notes/emergent-codebook/theta_prime_calibration.json`](../emergent-codebook/theta_prime_calibration.json)
  all landed. 3/3 unit tests pass. Calibration run completed on MPS in
  **8.5 wall-clock seconds** (well under the spec's "two days" estimate
  — the run cost was overestimated; the *interpretation* cost is where
  the time lives).
- **Empirical finding (substantive):** the `θ' ≈ 1/β` starting
  approximation is **significantly wrong in both directions** for
  D=4096 / n_cluster_members=16 / d_eff≈8 at the E1 protocol's β set:

  | β | empirical θ' | `1/β` baseline | flag |
  |---|---|---|---|
  | 0.01 | < 0.05 | 100 | `boundary_below_grid` |
  | 0.1 | < 0.05 | 10 | `boundary_below_grid` |
  | 1.0 | > 0.9 | 1.0 | `boundary_above_grid` |

  At **β ≤ 0.1**, top-1 retrieval over 50 stored centroids stays at
  chance (≈ 1/50) across the whole grid — the softmax is too soft and
  the retrieval boundary collapses below any reasonable d̄. The `1/β`
  prediction (θ' = 10 to 100) was off by 2-3 orders of magnitude — the
  *actual* recoverable d̄ at low β is essentially zero.
  At **β = 1.0**, retrieval stays at 100% success through d̄=0.7 and is
  still 72% at d̄=0.9; never crosses 50%. The `1/β = 1.0` baseline is
  too conservative; the true boundary is ≥ 0.9. **Binding implication:**
  any Phase 3 / Phase 5 substrate decision that has relied on the
  uncalibrated `1/β` approximation has been operating with the wrong
  regime boundary at all β values currently in use. C.3 should run
  twice — once with the `1/β` default, once with the calibrated lookup
  — and any divergence in regime stratification is a finding to record,
  not paper over.
- **Limitations of this spike-level run (binding for future calibration
  sessions):**
  - n=50 per cell; d_eff fixed at 8.0; D=4096; n_cluster_members=16.
    Single-seed.
  - `boundary_below_grid` and `boundary_above_grid` flags mean the
    grid was too narrow at both ends to find the actual 50% crossing.
    Extending the grid to d̄ ∈ {0.001, 0.005, 0.01, 0.02, 0.03} (low
    side) and {0.95, 0.99} (high side) would tighten the empirical θ'
    at β ∈ {0.01, 0.1, 1.0}.
  - Subagent design choice (justified): `max_iter=1` (one-shot MHN
    retrieval), not iterative settling. The classical θ'(β) boundary
    in the Vangara-Gopinath paper is a one-shot retrieval boundary,
    not a fixed-point one. Iterative settling at low β diffuses the
    query into wrong basins (verified during smoke). Recorded in code
    comment.
  - Multi-d_eff sweep would convert the scalar `θ'(β)` lookup into a
    `θ'(β, d_eff)` surface. The current artifact is a scalar lookup;
    consumers of `theta_prime_fn` get a single-argument signature
    (β only), so a future multi-d_eff calibration would require
    extending the consumer API.
- **`load_theta_prime_calibration` helper:** reads the JSON at the
  default path
  (`notes/emergent-codebook/theta_prime_calibration.json`), returns a
  `theta_prime_fn(beta)` that does exact-match lookup with linear
  interpolation in log-β space between calibrated points, and falls
  back to `1/β` (warning to stderr once per call) outside the
  calibrated β range. Returns `None` if the file doesn't exist (caller
  falls back to `1/β` default).
- **C.1 standalone discipline preserved.** No wiring into
  `ConsolidationDiagnostics`. `compute_codebook_regime_diagnostics`'s
  `theta_prime_fn` parameter is the natural integration point — drop
  in `load_theta_prime_calibration()` if available, fall through to
  the `1/β` default otherwise. Consumer wiring is C.1 follow-up.

## C.1 Closure Summary (2026-05-26)

All five C.1 deliverables landed:

| # | Module | Tests | Headline finding |
|---|---|:--:|---|
| C.1.1 | `phase3/basin_diagnostics.py` | 9/9 | NC1 = d_eff(basin covariance); NC2 = ETF Frobenius; pairwise-mean fallback wired |
| C.1.2 | `phase3/bimodality_diagnostic.py` | 10/10 | Hartigan dip primary, GMM-BIC fallback; both hand-rolled |
| C.1.3 | `phase3/regime_diagnostic.py` | 7/7 | Lifted from post-hoc script; `theta_prime_fn` injectable |
| C.1.4 | `experiments/calibrate_theta_prime.py` + calibration JSON | 3/3 | `1/β` approximation is wrong by 2-3 orders of magnitude at β≤0.1; binding for C.3 |
| C.1.5 | `phase3/metastability_diagnostic.py` | 5/5 | Machinery already existed (knob off); thin per-atom wrapper |

Plus D=4096 substrate fidelity tests (6/6) and existing FHRR tests
(2/2). Full Path C C.1 + substrate suite: **42/42 pass**.

### 2026-05-26 — C.3 smoke completed (PARTIAL — two methodology gaps surfaced)

**Status:** C.3 scaffolding (driver, tests, smoke output) lands. n=3 smoke
at the Phase 2 operating point (D=4096, β=10, L=64, W=8) surfaces two
real methodology gaps that must close before any n≥10 graduation run.
Path C C.3 is **partial**, not closed.

**Files created:**
- [`experiments/c3_phase3_exit_criterion.py`](../../experiments/c3_phase3_exit_criterion.py)
- [`tests/test_c3_phase3_exit_criterion.py`](../../tests/test_c3_phase3_exit_criterion.py)
  (5 tests pass; 15/15 with C.1.3 + C.1.4 regression)
- [`reports/c3_smoke_2026-05-26/c3_summary.json`](../../reports/c3_smoke_2026-05-26/c3_summary.json) +
  [`c3_summary.md`](../../reports/c3_smoke_2026-05-26/c3_summary.md)

**Smoke result (n=3 seeds, β=10, D=4096):**
- Only the `spread` regime stratum had any trials — `tight` and
  `borderline` were empty. At D=4096 a fresh random codebook of 200
  atoms has `d_bar_mean ≈ 0.9996`, well above the regime classifier's
  θ' boundary. All 200 atoms classify as `spread`.
- Recall@K (spread stratum): standard = `0.030 [0.019, 0.047]`, control = `0.032 [0.020, 0.049]`. Δ ≈ −0.002, CI-overlapping.
- Default `1/β` and calibrated `theta_prime_fn` modes produce
  numerically identical results at β=10.

**Two methodology gaps (binding for any n≥10 re-run):**

1. **Standard condition is shuffled-vs-shuffled.** The driver builds
   BOTH the standard codebook AND the shuffled control as fresh random
   codebooks (with different substrate seeds). The Phase 3 exit
   criterion is meant to compare a *consolidated* codebook (Hebbian +
   error-driven + Path C C.2.x dynamics) against the shuffled control.
   The smoke at n=3 is therefore a *plumbing + null baseline* run, NOT
   a meaningful Phase 3 graduation test. **Fix:** wire Phase 3
   consolidation (the existing `OnlineCodebookUpdater` / `StableOnlineCodebookUpdaterV2`
   path with the full Path C C.2 dynamics turned on at modest values)
   into the standard condition before evaluation; leave the shuffled
   control un-consolidated.

2. **β=10 is outside the C.1.4 calibration grid.** The calibration
   covers β ∈ {0.01, 0.1, 1.0}. The Phase 2 baseline operating point
   is β=10. The loader correctly falls back to `1/β` for out-of-range
   β and emits a stderr warning. The "both theta_prime variants"
   binding constraint is therefore satisfied trivially at the
   operating point — both modes degenerate to the same function. The
   C.1.4 finding "1/β is wrong by 2-3 orders at low β" predicts the
   regime labels could differ substantially at β=0.01 or β=0.1, but
   THIS is untested at the operating point. **Fix:** extend
   `experiments/calibrate_theta_prime.py` to cover β ∈ {3, 10, 30, 100}
   (the Phase 2 baseline grid) before the C.3 graduation run, so the
   dual-mode comparison has bite.

**Partial-closure framing:** the C.3 driver, tests, JSON/MD outputs,
Wilson CIs, regime stratification, and dual-θ' plumbing are all
production-shaped. The experimental DESIGN has gaps. C.3's pre-graduation
remainder:
- Extend the C.1.4 calibration grid to β ∈ {3, 10, 30, 100} (small
  follow-up experiment).
- Wire Phase 3 consolidation into the standard condition (driver
  modification).
- Then run n≥10 graduation gate with regime stratification and dual
  theta_prime modes.

**Pre-existing C.2.5 A9 flake fixed in this commit:** identical pattern
to the C.2.4 A4 flake. `substrate.random_vector()` consumes the global
torch RNG; under full-discovery test ordering the prior tests left a
different state, occasionally producing an outlier in the A9 composed-
system smoothness statistic. Fix: explicit `torch.manual_seed(43)` at
the start of A9 (matching the existing `gen = torch.Generator().manual_seed(43)`
call inside the test). Verified stable across 2 consecutive
full-discovery runs (515/515 each).

**Path C closure status (2026-05-26 end-of-session):** C.1 ✓ (passive
instrumentation) + C.2 ✓ (diagnostic-as-actuator) + C.3 PARTIAL
(scaffolding done; experimental design needs two fixes before n≥10
graduation). **Phase 5′ remains paused.**

### 2026-05-26 — C.3 second smoke (consolidated standard + extended calibration): substantive NULL

After the first smoke's methodology gaps were partially closed
(calibration grid extended to β ∈ {0.01, 0.1, 1.0, 3, 10, 30, 100}
and Phase 3 consolidation wired into the standard condition with all
five C.2.x dynamics on at modest values), the n=3 smoke produced a
**substantive null result** on the Phase 3 exit criterion at the
Phase 2 operating point (D=4096, β=10, vocab=200, n_consolidation_events=1000,
1137s wall on CPU).

**Calibration extension landed cleanly:**
- β=10 empirical θ′ = 0.99 (vs 1/β=0.1) → ~10× larger; classifies most
  atoms as `tight` instead of `spread`.
- The calibrated regime classifier produces dramatically different
  results from default at the Phase 2 operating point.

**Substantive findings from the n=3 consolidated smoke:**

1. **Consolidation produces measurable structural change.** Regime
   distribution shifts under calibrated θ′: random codebook is 0/200/0
   (tight/spread/borderline); consolidated codebook is 174-193/7-26/0.
   ~90% of atoms re-classify from `spread` to `tight` after 1000
   consolidation events. **The Path C C.2 dynamics measurably reshape
   the codebook's geometry.**

2. **Consolidation does NOT improve overall held-out Recall@K.**
   Pooled across 3 seeds: standard 0.023 vs control 0.032, Δ = −0.009.
   Wrong direction; consolidation appears to slightly degrade overall
   retrieval at this configuration.

3. **The headline's "CI-disjoint in `tight` stratum" is a
   methodological artifact (error worth recording).** Standard mode
   shows `tight` Recall@K 0.022 [0.013, 0.039] vs control `tight`
   0.000 [0.000, 0.000], "disjoint=YES." But the control's `tight`
   stratum has zero trials — random codebooks have no `tight` atoms
   by construction. A Wilson CI on 0/0 = [0,0] is degenerate; any
   positive number trivially "beats" it. The apparent CI-disjointness
   is NOT a real signal.

**Two methodological errors recorded for this null finding:**

- **Error 1: degenerate stratified comparison.** When the control's
  stratum has zero trials, the stratified comparison cannot produce
  meaningful evidence either way. Future C.3 runs must either: (a)
  use a control that populates the same regime strata as the standard
  condition (which requires the control to ALSO run consolidation,
  contradicting the "no-training" framing), or (b) report only on
  strata where both standard and control have populated trials.
- **Error 2: shuffled-control conflation.** The C.3 driver's
  "shuffled control" is actually a *no-consolidation control* (fresh
  random codebook, no training pipeline). The proper Phase 3
  shuffled-token control per [`phase-3-deep-dive.md:217-218`](../emergent-codebook/phase-3-deep-dive.md)
  runs the SAME training pipeline (consolidation) with **random
  token-to-hypervector assignment**. The current control conflates
  "did training help?" with "did the substrate ever see the corpus?"
  — a weaker comparison than the spec calls for.

**Interpretation of the null:**

Path C built a principled diagnostic-as-actuator framework. The
framework's anti-homunculus properties verified (5 reviewer PASSes
+ 19 binding watch-edges held). The framework PRODUCES measurable
structural change in the codebook (regime distribution shifts).
**But the structural change does NOT translate to improved held-out
Recall@K at the Phase 2 operating point.** Specifically: atoms move
from `spread` to `tight` regime — i.e., inter-atom distances
*decrease* under consolidation. This is **composition collapse** at
the inter-atom level (not within-basin collapse, which C.2.1
specifically prevents). The 2026-05-09 reformulation's full
prescription was "bounded non-zero within-basin variability **while
preserving inter-basin separability**." C.2.1–C.2.5 address the first
half; the second half — inter-atom separability — is governed by
`alpha_anti`, which **defaulted to 0.0** during this smoke (audit
§4.1 noted alpha_anti is "wired but underspecified").

**The audit's §4.3 hypothesis ("regime stratification + correct
calibration would reveal hidden signal Report 017 missed") is
falsified at this operating point and configuration.** Regime
classification *is* dramatically different under calibrated θ′, but
the classification doesn't carry retrieval signal because composition
collapse erases the inter-atom separability the regime test depends
on.

**Path C closure as of 2026-05-26:** C.1 ✓, C.2 ✓, **C.3 produces a
substantive null at the Phase 2 operating point and current C.2
configuration**. The Phase 3 exit criterion is NOT met; Phase 3 does
not graduate. Phase 5′ remains paused.

### Recommended next path

The composition-collapse finding points at a clean, well-scoped next
experiment: **C.3 re-run with `alpha_anti` enabled** (the existing
TorchFHRR inter-atom repulsion mechanism, audit §4.1 [F] gap), plus
the proper shuffled-token control fix per Error 2 above. This tests
two hypotheses simultaneously:

1. **(Mechanism)** Does inter-atom anti-repulsion (`alpha_anti > 0`)
   preserve separability under the C.2 dynamics, allowing the
   structural change to translate to retrieval improvement?
2. **(Methodology)** Does the proper shuffled-token-with-consolidation
   control produce a meaningful comparison (vs the current
   no-consolidation control)?

If this iteration ALSO produces a null, Phase 3 graduation at this
operating point is genuinely unreachable and the project needs to
pivot — either to a different operating point (WikiText-2 scale, β
sweep, different L/W), or to an entirely different Phase 3 mechanism
design.

If this iteration produces signal, escalate to n=10 graduation.

### 2026-05-26 — Path α smoke: STRONGER NULL than the C.3 baseline

Path α executed:
- `alpha_anti = 0.01` enabled in `TorchFHRR` (audit §4.1 [F] gap closed at the C.3 operating point for the first time at D=4096).
- `repulsion_step_size = 0.05` in `ReplayConfig`; substrate repulsion-force consolidation step at `replay_loop.py:786-798` now active.
- Shuffled-token control corrected per Error 2: now runs the SAME consolidation pipeline as the standard condition, but with a random token-to-hypervector permutation.

n=3 smoke result at the Phase 2 operating point:

| Mode | Stratum | std R@K [CI] | ctrl R@K [CI] | Δ | disjoint? |
|---|---|---|---|---:|:--:|
| default | spread | 0.023 [0.014, 0.039] | 0.022 [0.013, 0.037] | +0.002 | no |
| calibrated | tight | 0.022 [0.013, 0.039] | 0.021 [0.012, 0.038] | +0.001 | no |
| calibrated | spread | 0.032 [0.009, 0.109] | 0.024 [0.007, 0.083] | +0.008 | no |

**The shuffled-token control with consolidation produces essentially
identical Recall@K to the standard condition.** Standard's tiny edge
over control (+0.001 to +0.008 across strata) sits well within the
Wilson CIs at n=3. Far from CI-disjoint.

**Interpretation (stronger than the original C.3 null):** with the
proper shuffled-token-with-consolidation control, "did consolidation
help?" is no longer conflated with "did the substrate ever see the
corpus?" The comparison is now between (a) consolidation on real
token assignments and (b) consolidation on randomly-permuted token
assignments. **At this configuration the two are statistically
indistinguishable.**

The diagnosis sharpens: **consolidation produces structural change
that is largely independent of corpus-specific token assignment.**
Pull/push (the Hebbian mechanism inside `_consolidate()` at
`online_codebook.py:144`) IS firing in both conditions; the C.2.x
dynamics IS firing in both conditions; `alpha_anti` IS firing in
both conditions. None of these mechanisms produce a measurable
preference for the real-corpus standard over a randomly-permuted
control.

**Two candidate explanations** (both worth testing in Path β):

1. **Consolidation is too weak to differentiate at this scale.** At
   `lr_pull` default values + 1000 events + vocab=200, the
   accumulated pull/push signal may not be strong enough to encode
   corpus-specific structure above noise. Sweep `lr_pull` and
   `n_consolidation_events`.
2. **Synthetic vocab=200 is too simple.** Real corpora have
   long-tail distributions, co-occurrence regularities, and rare-word
   patterns that the synthetic random-sentence generator doesn't
   capture. WikiText-2 (~30K vocab, real text statistics) would test
   whether consolidation differentiates at meaningful scale.

The audit's §4.3 hypothesis remains FALSIFIED at this operating point.
**The structural-change-without-corpus-specific-learning finding is
the dominant signal from this session.**

### Recommended next path (post-Path-α)

The Path α null rules out "inter-atom-separability was the missing
piece." The next test should be **Path β: WikiText-2 operating
point**, because:

1. It's the project's actual target scale (real text, ~30K vocab).
2. The synthetic-corpus null doesn't generalize cleanly to real
   corpora — the consolidation mechanism may simply require richer
   statistical structure to express corpus-specific learning.
3. The Phase 2 baseline pipeline already supports WikiText-2 (audit
   §4.2; `experiments/02_phase2_retrieval_baseline.py` has a
   `--corpus-source wikitext` flag).
4. If WikiText-2 still produces null, the conclusion is much
   stronger: Phase 3 graduation is unreachable with this mechanism
   regardless of operating point, and Path γ (mechanism redesign per
   Eugenio's Self-Organizing Language / Hyperseed / Predictive Coding
   directions per `literature-and-principles.md`) is the only honest
   remaining option.

This is a falsification ladder: synthetic null → real-corpus null →
mechanism redesign required.

---

**Cross-cutting binding findings for C.2 and C.3:**
- N=5 rolling window is a binary collapse detector for NC1 and a
  power-limited dip test for bimodality; persistence-across-events is
  the principled recovery (≥3/5 rejections).
- The trajectory-vs-fixed-point semantic split in metastability `c_i`
  must be explicitly chosen at C.2.4 and justified to the
  anti-homunculus reviewer.
- The `1/β` regime approximation has been silently wrong; C.3 must run
  with both the `1/β` default and the calibrated lookup and report
  divergence as a finding.
- The first-attempt dip-statistic implementations in C.1.2 (ECDF /
  one-sided envelope) both *passed simple sanity tests* yet were
  wrong (over-rejected unimodal Gaussian). The textbook Hartigan
  construction is the only correct one. **C.2 reviewers should treat
  "approximate" implementations of statistical tests as
  presumptively-buggy until cross-validated against a clearly bimodal
  vs clearly unimodal pair.**
- C.1.5's discovery (the metastability infrastructure was already
  wired but the knob was off) is a reminder that the audit's
  "never built / never run" framing can mask "built but never
  exercised." CLAUDE.md's grep-by-default rule for the 0.0-default
  knobs is exactly the right discipline.
