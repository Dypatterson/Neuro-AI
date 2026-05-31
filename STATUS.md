# Project STATUS

The bookmark. Read this first every session before doing anything. If something
in this file is wrong or stale, fix this file *first*, then do the work.

Keep it small: a fixed-shape **Current state** block (overwritten each session)
plus a rolling **Recent updates** window (≤10 entries / ≤7 days). Long-form
narrative, walk-back chains, and per-report detail live in the numbered reports
and the monthly archive at [notes/status-log/](notes/status-log/) — never inline
here. See [§Maintaining this file](#maintaining-this-file).

## Current state

*(2026-05-29. Five fixed fields — overwrite, do not append. Detail → report + archive.)*

- **Active phase:** **Phase 3 — continual learning / "Growing Codebook" (Frame B).** Reframed 2026-05-28 after the C.3 shuffled-token control was found **gauge-vacuous** (it permutes which i.i.d. atom each token-id wears → `E[Δ]=0` by construction), which makes the Path C / Path α / Γ1 null chain uninterpretable as evidence about corpus-specific learning. Anchor: [2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md](notes/notes/2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md).
- **Headline metric:** Frame B endpoint **level-DiD** on stratum-pooled Recall@K (consolidation benefit, real vs global-stream-shuffle world). The within-seed **slope-DiD** reframe `Δβ(s)=β_real−β_shuffle` is **DROPPED 2026-05-30** per the pre-committed `corr_AC_BD<0.3` rule (read-protocol [05:30](brainstorm-workspace/2026-05-29-frameb-grill/05-stage1-read-protocol.md), "apply verbatim"): the recovered Gate-0 n=10 summary gives `corr_AC_BD=+0.10` → real/shuffle lifts ~uncorrelated, so the slope's *outer* (real−shuffle) variance kill is **falsified** (differencing cancels ~nothing). The *inner-slope* σ(β) kill stays **untested (Q1 open)** — the drop honors the pre-commit, it does **not** measure σ(β). Slope-DiD [spec](notes/notes/2026-05-28-frame-b-exposure-slope-headline-design.md) shelved, never pinned. The old C.3 `ΔE`-vs-shuffled-token control stays **retired** (gauge-vacuous). [Report 116](reports/116_frameb_corr_ac_bd_slope_drop.md).
- **Active deliverable:** **STRATEGIC DECISION PENDING.** The endpoint level-DiD is **not economically rescuable** at this op point ([Report 117](reports/117_frameb_leveldid_feasibility_consolidation_variance.md), 9-agent verified). **Walk-back of the 2026-05-30 "n≈27 window-averaging" plan:** window-averaging via `window_seed_override` is **K× re-consolidation, not cheap eval**; total cells to clear zero is **monotonic in K** (min at K=1: **n≈424 ≈ 42× the n=10 run** to clear zero, ~86× for power; **n=40 reaches only t≈0.61 — clears nothing**). Report 115 §3's `K=9⇒σ=0.044`/`pool saturates` numbers were **wrong** (K=9⇒σ=0.069; test pool ~49K, compute- not pool-limited) — its no-clear-at-n=10 conclusion survives. **Root cause:** consolidation *injects* the variance (σ_A=0.157 vs frozen σ_C=0.069; the DiD differencing *adds* variance since `corr_AC_BD≈0.10`). **Landscape diagnostic DONE 2026-05-30 → landscape lever CLOSED** ([Report 118](reports/118_frameb_landscape_sweep_variance_irreducible.md); parallel fan-out, anchor σ_A(64)=0.157 reproduced exactly). Verdict **CAPACITY-WALL / σ_A-irreducible**: the frozen floor σ_C drops with L (0.069→0.033→0.023) but σ_A does **not** (0.157→0.150→0.139) — consolidation **re-injects variance independent of landscape size**; L=512 hit the β=10 capacity wall (lift +0.050→−0.006). **Three levers now closed this session** (slope / level-DiD via seeds+windows / landscape). **STRATEGIC RETHINK PENDING (user's call):** Frame B mechanism redesign, OR a β/D × L op-point exploration (the capacity wall implicates β, not just L), OR accept the effect is structurally too weak and pivot. Nugget: consolidation lift **peaks at L=256** (+0.050; real-world A−C only, NOT the DiD). Still true: `corr_AC_BD=+0.10`→**slope dropped** ([Report 116](reports/116_frameb_corr_ac_bd_slope_drop.md)); Gate-0 n=10 summary recovered on-repo (`reports/gate0_2026-05-28/`, `G0->dead`→`G0->weak`). Slope stays shelved (σ(β) likely also large — the variance is in the lift, not an intercept).
- **Blockers:** (1) Retrieval-time interventions over the existing FHRR substrate are **closed at smoke scale** (D1/D3/E1/M1 null; Reports 062–066). Training-time **M2 path remains open**; the range-shaped replay downstream lane is **closed** as novelty-without-retrieval ([Report 111](reports/111_phase5_prime_range_replay_downstream_viability.md)). (2) No path yet has an **n≥10 result with a control** — no graduation-style claim is licensed.
- **Paused:** **Phase 5′.** Pre-pause headline spec, binding once Phase 5 reopens: `ΔE = E_content-prior − E_role-prior`, magnitude floor `5.5e-3`, CI strictly > 0 ([phase-5-unified-design.md:282-297](notes/emergent-codebook/phase-5-unified-design.md)). The 2026-05-26 audit §9 flags a saturating `raw_scene_energy_v0` readout and an arbitration-shaped `min_branch` aggregator that must be fixed before any reopen ([audit-phase5-2026-05-26.md](audit-phase5-2026-05-26.md)).

## Recent updates

Newest first. Rolling window — ≤10 entries / ≤7 days, each ≤5 lines. Older
entries live **only** in [notes/status-log/2026-05.md](notes/status-log/2026-05.md).

- **2026-05-30** — **Landscape-size diagnostic DONE → landscape lever CLOSED.** Parallel fan-out (30 single-cell CUDA workers; anchor σ_A(64)=0.157 reproduced exactly). The frozen floor σ_C drops with L (0.069→0.023) but σ_A does **not** (0.157→0.139) → consolidation re-injects variance **independent of landscape size**; L=512 hit the β=10 capacity wall (lift +0.050→−0.006). Verdict **CAPACITY-WALL/irreducible**. **3 levers closed this session** (slope/level-DiD/landscape); strategic rethink pending. [Report 118](reports/118_frameb_landscape_sweep_variance_irreducible.md) · [archive](notes/status-log/2026-05.md)
- **2026-05-30** — **Level-DiD shown NOT economically rescuable; the n≈40 plan is killed.** 9-agent workflow (verified, reproduced from the recovered JSON): window-averaging via `window_seed_override` is **K× re-consolidation, not cheap eval**; total cells to clear zero is **monotonic in K** → min at K=1 = **n≈424 ≈ 42× n10** (n=40 → t=0.61, clears nothing). Report 115 §3's `K=9⇒σ=0.044`/`pool saturates` numbers were **wrong** (corrected). Root cause: **consolidation injects the variance** (σ_A=0.157 vs σ_C=0.069; DiD differencing *adds* variance). Decision pending: cheap landscape-sweep diagnostic vs mechanism reconsideration. [Report 117](reports/117_frameb_leveldid_feasibility_consolidation_variance.md) · [archive](notes/status-log/2026-05.md)
- **2026-05-30** — **Frame B branch RESOLVED: slope DROPPED.** Recovered the missing Gate-0 n=10 summary from Google Drive (artifact dependency discharged; now on-repo `reports/gate0_2026-05-28/`). `gate0_frame_a.py --variance-report` → **`corr_AC_BD=+0.10`** < pre-committed 0.3 cutoff (σ(A−C)=0.149, σ(B−D)=0.143, ~uncorrelated → the *outer* real−shuffle kill cancels ~nothing; honest-n≈750 at raw σ). Pre-commit `corr_AC_BD<0.3` fires → **drop slope, stay on level-DiD** (falsifies the outer kill; the inner-slope σ(β) kill stays **untested/Q1-open** — the drop honors the pre-commit, not a σ(β) measurement); `G0->weak`→escalate-n → **n≈27 level-DiD** is next. Drive verdict was stale `G0->dead`; current code reclassifies **`G0->weak`** (bug-fix confirmed). [Report 116](reports/116_frameb_corr_ac_bd_slope_drop.md) · [archive](notes/status-log/2026-05.md)
- **2026-05-29** — **Decision-free prereq batch landed.** [Report 115](reports/115_frameb_gate0_n10_drilldown.md) (done-gate #4 ✅, drill-down); test 4a codebook-only eval-isolation guard (`tests/test_frameb_eval_isolation.py`, 4/4); c3 `n_consolidation_events`→`n_observations` **alias** (hard rename unsafe — CLI flag + gate0 + 9 notebooks + tests; JSON key preserved); [stream-shuffle control-validity proof](notes/notes/2026-05-29-stream-shuffle-control-validity.md); draft-flag pass on 01/04/05. Suite 561 tests, same 6 pre-existing Phase-5 errors, **0 new**. Verdict firewall **deferred** (no slope classifier exists to separate). [archive](notes/status-log/2026-05.md)
- **2026-05-29** — **Audit + plan + grill workflow (39 agents).** No binding contradiction (1 *moderate*: spec §Why-a-slope attributed σ to the codebook → **annotated**; minors). **Dispositive:** within-seed corpus-window draw dominates the codebook/atom draw **~24×** (a ranking, not a %); a √K window-average **cannot clear the level-DiD off zero at n=10 by construction** (K=9 ⇒ CI≈[−0.013,+0.051]) → **n≈27 seeds** is the lever. Slope σ(β) **untested** (Q1 open); reframe conditionally-approved, PROPOSED. [digest](brainstorm-workspace/2026-05-29-frameb-grill/07-workflow-digest.md) · [review](brainstorm-workspace/2026-05-29-frameb-grill/06-stage1-review-findings.md) · [archive](notes/status-log/2026-05.md)
- **2026-05-28** — Frame B slope-DiD headline **designed** (4-design panel → synthesis, winner 29/30); within-seed log-exposure slope, byte-identity guard, self-falsifying realized-σ gate. **PROPOSED pending sign-off.** [spec](notes/notes/2026-05-28-frame-b-exposure-slope-headline-design.md) · [archive](notes/status-log/2026-05.md)
- **2026-05-28** — Variance investigation: per-seed DiD σ=`0.196` is **structural** (~5× the binomial floor) → the single-point estimand is underpowered (~590 seeds needed); CUPED re-analysis + exposure-slope estimand prescribed. [archive](notes/status-log/2026-05.md)
- **2026-05-28** — Gate 0 **ran** (n=10 wikitext): gauge **confirmed** (4a byte-identical; 4b Δ CI contains 0), verdict **`G0->weak`**; caught/fixed a verdict-classifier bug. Report 115 pending. [archive](notes/status-log/2026-05.md)
- **2026-05-28** — Gate 0 **built + tested**, ready for Colab n=10: matched-world DiD, 4a byte-identity test, arm-E per-seed gauge amendment. 547 tests pass. [archive](notes/status-log/2026-05.md)
- **2026-05-28** — Gauge finding **verified sound** (adversarial + empirical probe); C.3 per-seed CI pseudo-replication bug **fixed** (per-seed inference, +6 tests). [archive](notes/status-log/2026-05.md)
- *Older entries (2026-05-27 ← 2026-05-23): see [notes/status-log/2026-05.md](notes/status-log/2026-05.md).*

---

## Live operational policies

- **HAM regime split** (from report 022): β=30 + summed scores for retrieval; β=10 + HAM-arithmetic for replay diagnostics.
- **Online error-driven codebook updates: BANNED.** Use Hebbian for runtime, error-driven only in batch offline passes.
- **Drift sources sanctioned by design** (phase-4-unified-design.md:296-309): periodic batch retrains, online Hebbian reinforcement, or simulated synthetic perturbation.

---

## Pre-phase commitments (all resolved)

Specified as required before claiming a phase done. All three are now closed —
grep-confirmed 2026-05-29 after they had drifted as "still open" in this file.

- ~~Consolidation-geometry regime classifier (d̄, d_eff per atom)~~ **RESOLVED 2026-05-20** — built ([scripts/consolidation_geometry_diagnostic.py](scripts/consolidation_geometry_diagnostic.py)), lifted to a reusable module with tests ([regime_diagnostic.py](src/energy_memory/phase3/regime_diagnostic.py), [test](tests/test_regime_diagnostic.py)), and run twice (post-hoc over 10 snapshots in [Report 044](reports/044_consolidation_geometry_diagnostic.md); C.3 exit-criterion over live-consolidation atoms). The spec's per-atom `d̄`/`d_eff` are computed as specified; only the data referent shifted (substrate state ← live context-bag history).
- ~~Empirical θ′(β) calibration spike~~ **RESOLVED 2026-05-26** (Path C C.1.4) — E1 protocol run on the FHRR substrate ([calibrate_theta_prime.py](experiments/calibrate_theta_prime.py), [test](tests/test_theta_prime_calibration.py)) at β∈{0.01…100}; empirical θ′(β=10)≈0.99 confirms θ′≈1/β is off by 2–3 orders of magnitude at low β. Artifact: [theta_prime_calibration.json](notes/emergent-codebook/theta_prime_calibration.json).
- ~~Frequency-weighted Benna-Fusi α~~ **RESOLVED 2026-05-17** — built, tested, run at n=10×40 on Colab; production-scale null. See [Report 040](reports/040_freq_weighted_alpha_sweep.md) and [phase-5-unified-design.md:97-107](notes/emergent-codebook/phase-5-unified-design.md).

---

## Maintaining this file

- **Budget (enforced):** keep this file **< 250 lines / < 20 KB**. A pre-commit hook ([scripts/check_status_size.sh](scripts/check_status_size.sh)) fails the commit if it goes over. If an edit would exceed budget, trim the Recent-updates window **before** adding.
- **Current state** is a fixed 5-field block — **overwritten** each session, never appended. No narrative, no per-report chain; that goes to the report + archive.
- **Recent updates** is a rolling window: **≤10 entries / ≤7 days**, each **≤5 lines** (summary + report link + archive link). Adding an 11th entry means cutting the oldest — and the oldest may only be cut once its section exists in [notes/status-log/](notes/status-log/).
- **Walk-backs go first.** If a session walks back something currently in this file, that walk-back is the **first** edit of the session — and it goes in *Current state*, not stacked into *Recent updates*.
- Update at the **end of every working session**.
- Promote a "blocker" → "done" only when a report has **multi-seed CI evidence**.
- Monthly archive files are append-only (newest section at top within a file).
