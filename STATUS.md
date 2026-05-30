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
- **Headline metric:** Frame B within-seed log-exposure **slope** `Δβ(s) = β_real(s) − β_shuffle(s)` — **PROPOSED, pending user sign-off.** Spec: [2026-05-28-frame-b-exposure-slope-headline-design.md](notes/notes/2026-05-28-frame-b-exposure-slope-headline-design.md); pinned headline at [phase-3-deep-dive.md §Headline](notes/emergent-codebook/phase-3-deep-dive.md). The old C.3 `ΔE`-vs-shuffled-token control is **retired** (gauge-vacuous).
- **Active deliverable:** the **window-vs-slope branch decision** (pending `corr_AC_BD`; drop slope if `<~0.3`, default window on ties). [Report 115](reports/115_frameb_gate0_n10_drilldown.md) **written** (done-gate #4 discharged on the on-repo DiD; drill-down, not graduation). Gate 0 ran (n=10 wikitext): gauge **confirmed**, verdict **`G0->weak`** (DiD `+0.019` CI `[-0.121,+0.159]`, 6/10 positive, per-seed σ≈0.19 — underpowered, not a redesign trigger). Stage-1 variance decomp: within-seed **corpus-window draw dominates the codebook/atom draw ~24×** (≈96/4 on a single 10×4 run, df=9 — a ranking, not a %). **Dispositive 2026-05-29 finding:** a √K window-average **cannot clear the level-DiD off zero at n=10 by construction** (K=9 ⇒ CI≈[−0.013,+0.051]); **n≈27 seeds**, not more windows, is the lever. Decision-free prereqs (test 4a, axis rename, Report 115, walk-backs, stream-shuffle validity proof, Stage-1B `corr_AC_BD`) precede the **window-vs-slope branch** (drop slope if `corr_AC_BD<~0.3`, default window on ties). Slope σ(β) **untested** (Q1 open).
- **Blockers:** (1) Retrieval-time interventions over the existing FHRR substrate are **closed at smoke scale** (D1/D3/E1/M1 null; Reports 062–066). Training-time **M2 path remains open**; the range-shaped replay downstream lane is **closed** as novelty-without-retrieval ([Report 111](reports/111_phase5_prime_range_replay_downstream_viability.md)). (2) No path yet has an **n≥10 result with a control** — no graduation-style claim is licensed.
- **Paused:** **Phase 5′.** Pre-pause headline spec, binding once Phase 5 reopens: `ΔE = E_content-prior − E_role-prior`, magnitude floor `5.5e-3`, CI strictly > 0 ([phase-5-unified-design.md:282-297](notes/emergent-codebook/phase-5-unified-design.md)). The 2026-05-26 audit §9 flags a saturating `raw_scene_energy_v0` readout and an arbitration-shaped `min_branch` aggregator that must be fixed before any reopen ([audit-phase5-2026-05-26.md](audit-phase5-2026-05-26.md)).

## Recent updates

Newest first. Rolling window — ≤10 entries / ≤7 days, each ≤5 lines. Older
entries live **only** in [notes/status-log/2026-05.md](notes/status-log/2026-05.md).

- **2026-05-29** — **Decision-free prereq batch landed.** [Report 115](reports/115_frameb_gate0_n10_drilldown.md) (done-gate #4 ✅, drill-down); test 4a codebook-only eval-isolation guard (`tests/test_frameb_eval_isolation.py`, 4/4); c3 `n_consolidation_events`→`n_observations` **alias** (hard rename unsafe — CLI flag + gate0 + 9 notebooks + tests; JSON key preserved); [stream-shuffle control-validity proof](notes/notes/2026-05-29-stream-shuffle-control-validity.md); draft-flag pass on 01/04/05. Suite 561 tests, same 6 pre-existing Phase-5 errors, **0 new**. Verdict firewall **deferred** (no slope classifier exists to separate). [archive](notes/status-log/2026-05.md)
- **2026-05-29** — **Audit + plan + grill workflow (39 agents).** No binding contradiction (1 *moderate*: spec §Why-a-slope attributed σ to the codebook → **annotated**; minors). **Dispositive:** within-seed corpus-window draw dominates the codebook/atom draw **~24×** (a ranking, not a %); a √K window-average **cannot clear the level-DiD off zero at n=10 by construction** (K=9 ⇒ CI≈[−0.013,+0.051]) → **n≈27 seeds** is the lever. Slope σ(β) **untested** (Q1 open); reframe conditionally-approved, PROPOSED. [digest](brainstorm-workspace/2026-05-29-frameb-grill/07-workflow-digest.md) · [review](brainstorm-workspace/2026-05-29-frameb-grill/06-stage1-review-findings.md) · [archive](notes/status-log/2026-05.md)
- **2026-05-28** — Frame B slope-DiD headline **designed** (4-design panel → synthesis, winner 29/30); within-seed log-exposure slope, byte-identity guard, self-falsifying realized-σ gate. **PROPOSED pending sign-off.** [spec](notes/notes/2026-05-28-frame-b-exposure-slope-headline-design.md) · [archive](notes/status-log/2026-05.md)
- **2026-05-28** — Variance investigation: per-seed DiD σ=`0.196` is **structural** (~5× the binomial floor) → the single-point estimand is underpowered (~590 seeds needed); CUPED re-analysis + exposure-slope estimand prescribed. [archive](notes/status-log/2026-05.md)
- **2026-05-28** — Gate 0 **ran** (n=10 wikitext): gauge **confirmed** (4a byte-identical; 4b Δ CI contains 0), verdict **`G0->weak`**; caught/fixed a verdict-classifier bug. Report 115 pending. [archive](notes/status-log/2026-05.md)
- **2026-05-28** — Gate 0 **built + tested**, ready for Colab n=10: matched-world DiD, 4a byte-identity test, arm-E per-seed gauge amendment. 547 tests pass. [archive](notes/status-log/2026-05.md)
- **2026-05-28** — Gauge finding **verified sound** (adversarial + empirical probe); C.3 per-seed CI pseudo-replication bug **fixed** (per-seed inference, +6 tests). [archive](notes/status-log/2026-05.md)
- **2026-05-28** — Phase 3 **reframed** to continual learning (Frame B); C.3 shuffled-token control found gauge-vacuous (`E[Δ]=0` by exchangeability) → Path C/α/Γ1 null chain uninterpretable, Γ2 mandate not established. [anchor](notes/notes/2026-05-28-phase3-frame-b-continual-learning-and-gauge-control-finding.md) · [archive](notes/status-log/2026-05.md)
- **2026-05-27** — Γ1 family **CLOSED** via F1 lr_cr sweep: max mean Δ `+0.0117` < 0.02 threshold; lr_cr ∈ {0.20,0.50} give *negative* Δ (atom-vs-atom repulsion degrades the codebook). Γ2 (bundle-first) is next. [Report 114](reports/114_path_gamma_gamma1_family_closure.md) · [archive](notes/status-log/2026-05.md)
- **2026-05-27** — Γ1 headline gate **FAIL** at lr_cr=0.1: 5/10=50% per-seed (both clauses fail); matched-seed pull/push reproduces Report 112 v3 byte-identically. [Report 113](reports/113_path_gamma_gamma1_headline_gate.md) · [archive](notes/status-log/2026-05.md)
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
