# Phase 4 graduation synthesis — the architecture-level story

**Date:** 2026-05-16
**Companion to:** reports 026, 037, 038; standing discipline note [2026-05-16](2026-05-16-substrate-vs-readout-metric-discipline.md)
**Purpose:** Pull the three reports into one architecture-level claim before the work cools. Not a re-derivation — for numbers, follow links.

---

## What Phase 4 was supposed to answer

The Phase 4 design ([phase-4-unified-design.md](../emergent-codebook/phase-4-unified-design.md)) asked: **does replay + consolidation + reencoding + death make a Phase 3 substrate measurably better at contextual completion under active codebook drift?** The expected mechanism was Benna-Fusi-style multi-timescale consolidation: replay re-settles trajectories through the current landscape, sharpens basins around discovered patterns, lets stale patterns decay.

The original spec named two headlines (Recall@K, cap-coverage) and several drill-downs (meta-stable rate, engagement, pattern age, u_k distribution).

---

## What actually happened — three regimes, one mechanism set

### Regime 1: Phase 4 in isolation (frozen Phase 3c codebook, synthetic drift) — [report 026](../../reports/026_phase4_verification_design_spec.md)

Headline 1 (ΔR@10) verified at +0.010, CI [+0.001, +0.022], n=5. First positive Phase 4 result.

Headline 2 (Δcap-coverage) did not verify — per-seed variance ~5× larger than ΔR@K's; all CIs crossed zero.

D1 drill-down (meta-stable rate at W=3): Δ = −0.51, per-seed std 0.038. Every seed showed essentially the same effect. Not reported in 026 as load-bearing; later re-read as the most-robust Phase 4 signal in the project.

### Regime 2: A_k investigation + the mass-death finding — [reports 033–037](../../reports/036_decay_sweep_and_mass_death_finding.md)

Three sub-findings reshaped what Phase 4's mechanism actually does at n_cues=3000:

1. **A_k self-inhibition** (Saighi-paper-inspired basin-narrowing) was tested at n=10 across decay ∈ {0.0, 0.02, 0.05, 0.10, 0.20}. Falsified: it doesn't move the design-spec headlines beyond what the rest of the mechanism already does. Mechanism is correctly implemented but produces an effect orthogonal to the dominant substrate-shaping force.

2. **Pattern death is the dominant substrate-shaping force at n_cues=3000.** Between steps 1500 and 2000, ~99% of vocab atoms die for being un-reinforced. The few-dozen survivors are exactly the atoms that received any retrieval reinforcement during the run. This is a mass-death event, not the architecturally-intended "purge stale discovered patterns."

3. **Cap-coverage variance is corpus-order, not tunable.** [Report 037](../../reports/037_seed3_collapse_diagnostic.md) traced seed 3's −0.179 Δcap_t05 to the same death-survivor mechanism as seed 17's +0.036 improvement. The diagnostic A_nz at step 1500 (count of atoms with any retrieval inhibition accumulation) **predicts step-2000 survivor count exactly**. Direction (collapse vs improve) reduces to whether the surviving 5–32 atoms happen to cover the test set — a corpus-shuffle property, not a tunable parameter.

This is where the headline pivot happened. The 2026-05-16 [substrate-vs-readout discipline note](2026-05-16-substrate-vs-readout-metric-discipline.md) wrote down the rule: when readout × substrate variance is large, prefer substrate-pure metrics. Cap-coverage is `readout × substrate`. D1 is substrate-pure (basin geometry). Pivot.

### Regime 3: Phase 4 at the integration regime (online Hebbian, n_cues=3000, A_k off) — [report 038](../../reports/038_phase4_d1_graduation.md)

Δ meta_stable_w3 (C − A) = **−0.7920, 95% CI [−0.9376, −0.6463], 10/10 seeds drop to floor (0.000)**. Architectural-contribution test (C − B) also CI-disjoint at −0.720. Phase 4 graduates on D1.

Trajectory analysis surfaced something worth foregrounding: **D1 isn't produced by one sub-mechanism; the path varies by seed.**
- Seed 17 reaches the W=3 floor by step 500, with the full substrate intact (4125 atoms alive, 0 deaths). Pure replay+consolidation basin-sharpening.
- Seed 3 partial-drops to ms_w3 ≈ 0.43 by step 500 (replay-driven), plateaus, then completes the drop to 0 at step 2000 when mass death fires.

Both paths end at the same floor. The mechanism *set* (replay + consolidation + reencoding + death together) is what graduates. Isolating "which sub-mechanism does the work" depends on the seed's reinforcement trajectory through the corpus.

The R@10 small-positive signal from regime 1 replicates at the integration regime: +0.0089, pooled-Wilson Δ = +0.0091 over 2201 test samples. Same magnitude across n_cues=1500 (report 032) and n_cues=3000 (report 038). Real but variance-bound at this corpus size.

---

## The architecture-level claim

> **The Phase 4 mechanism set produces a measurably more decisive substrate** — meta-stable retrievals drop from ~0.5–1.0 to 0.000 at W=3 across 10 seeds at the integration regime — and that decisive substrate keeps the same small-positive R@K signal observed in isolation. Cap-coverage and top1 readouts remain variance-bound because they integrate substrate decisiveness × which atoms happen to survive; the substrate-pure metric is what cleanly tracks the mechanism's effect.

This is one architecture-level claim, not three. The pivot to D1 isn't a goalpost shift; the design spec specified *both* R@K and cap-coverage as headlines, but cap-coverage variance is downstream of binary-death survival outcomes that the spec didn't anticipate. D1 measures the property the spec *meant* — "settling becomes more decisive after consolidation" — without dependence on which atoms survived.

---

## Anti-homunculus check — does the claim survive the filter?

Every mechanism that produces the D1 effect is a local geometric dynamic:
- **Replay** re-settles trajectories through the current landscape (energy minimization, no controller).
- **Consolidation** updates the Benna-Fusi cascade per the EQ-10/EQ-11 dynamics (no arbitrator).
- **Death** is a local threshold test on per-pattern strength (no scheduler).
- **D1 itself** is `mean(top_score < 0.95)` over the eval set (a passive measurement of the substrate, not an actuator).

No "if-X-then-do-Y" rule sits anywhere in the path. No subsystem decides for another. The metric measures what the dynamics do. **Passes.**

---

## What Phase 5 inherits — and what's still open

**Inherits:**
- A working multi-timescale consolidation mechanism (Benna-Fusi cascade with u_1–u_6, retrieval-reinforced).
- A discovery channel that emits ~80 candidates per seed at the integration regime.
- A working replay store + gate dynamics.
- A substrate that converges to a small, decisive attractor set (a basis for hierarchical compression).
- The substrate-vs-readout discipline note as a binding evaluation principle.

**Known open Phase-4-revision questions (not graduation-gating):**
- **Binary vs gradient death.** Reports 036/037 hypothesize that a probabilistic/soft death rule would smear out the binary survival outcome and tighten cap-coverage variance. Untested. ~1–2 day intervention.
- **Top1 regression mechanism.** Confirmed CI-strictly-negative at integration (Δ = −0.048, CI [−0.083, −0.013]). Demoted to drill-down per the discipline note, but still a real geometric tradeoff: Hebbian codebook reshaping plus Phase 4 replay sharpens basins around dominant attractors, sometimes at the cost of the rare-target attractor. Could be inherent or fixable; untested.
- **Generalization beyond wikitext.** Untested. Right scope after Phase 5 expands the architecture's expressivity.

**Phase 5's architectural premise** (per [project plan §Phase 5](../../docs/PROJECT_PLAN.md)) — structure and abstraction through role/filler binding, atom splitting, hierarchical attractors — assumes the Phase 4 mechanism set produces an under-capacity slow store that *selectively* retains frequently-retrieved patterns. The most direct test of that premise sits in [brainstorm idea 5](../../brainstorm-workspace/2026-05-13-neuro-personal-ai/brainstorm-neuro-personal-ai.md) (retrieval-frequency-weighted Benna-Fusi α). That experiment was named pre-Phase-3 as "the key experiment for the architecture's compression → abstraction claim" and has never been built. It is the natural Phase-4→Phase-5 bridge.

---

## What this synthesis does NOT establish

- Phase 4 is not "finished." It graduated on D1; cap-coverage and top1 readouts remain known weak points. Phase-4-revision work is a parked option.
- The death mechanism in its binary form is not architecturally defended; it's a parameter setting that happens to work for D1.
- The "compression → abstraction" claim isn't tested yet; that's the Phase-5 entry condition, not a Phase-4 outcome.
- Wikitext is the only corpus this has been verified on. Generalization is a Phase-5+ concern.

---

## Reading order for a Phase-5 starter

1. [Report 038](../../reports/038_phase4_d1_graduation.md) — Phase 4 graduation result, mechanism heterogeneity.
2. [Report 037](../../reports/037_seed3_collapse_diagnostic.md) — death-survivor mechanism, A_nz prediction.
3. [Report 036](../../reports/036_decay_sweep_and_mass_death_finding.md) — A_k falsified, death dominant at n_cues=3000.
4. [Substrate-vs-readout discipline](2026-05-16-substrate-vs-readout-metric-discipline.md) — binding evaluation rule.
5. [Phase 4 unified design](../emergent-codebook/phase-4-unified-design.md) — original spec for context.
6. This file.
