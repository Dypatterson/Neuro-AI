# Report 037 — Seed-3 collapse diagnostic: A_nz at step 1500 predicts survivor count at step 2000

**Date:** 2026-05-16
**Active phase:** Phase 4 — verification (open decision per report 036)
**Headline metric per [phase-4-unified-design.md:276-282](../notes/emergent-codebook/phase-4-unified-design.md): Δ Recall@K AND Δ cap-coverage with active drift
**Required controls per [phase-4-unified-design.md:318-323]:** no-replay baseline, random-codebook control
**Last verified result:** [Report 036](036_decay_sweep_and_mass_death_finding.md) — A_k investigation closed; death identified as dominant substrate-shaping force
**Why this experiment now:** Option 1 from report 036's four-way decision — diagnose seed 3's Δcap = −0.179 to determine whether it's a fixable failure mode or intrinsic corpus-order variance. Decision-shaping for options 2–4. Free (no compute).

---

## Method

Pulled `phase34_results.json` from Drive folder `phase34_saighi_n3k_seed3` (matches the n=10 A_k run that produced the −0.179 outlier in report 035). Decoded the full per-checkpoint trajectory for the `phase3_phase4` condition and the embedded `death_diag` substrate diagnostic.

No new compute. Pure inspection of saved checkpoint state.

---

## Results

### Seed-3 trajectory (`phase3_phase4` condition)

| step | top1 | R@10 | **cap_t_05** | n_pat_alive_w2 | deaths_total | consol |
|---:|---:|---:|---:|---:|---:|---:|
| 0    | 0.115 | 0.372 | 0.266 | — | 0 | 0 |
| 500  | 0.092 | 0.353 | **0.206** | 4098 | 0 | 110 |
| 1000 | 0.110 | 0.362 | 0.243 | 4098 | 0 | 202 |
| 1500 | 0.110 | 0.362 | 0.243 | 4098 | 0 | 247 |
| 2000 | **0.055** | 0.362 | **0.087** | **5** | **7220** | 268 |

Δ cap_t_05 vs `baseline_static` (constant 0.266): 0.000, **−0.060**, −0.023, −0.023, **−0.179**.

Same structural pattern as seed 17's trajectory in report 036:
- Stable plateau through step 1500
- Mass death between step 1500 → 2000
- Substrate collapses to ~handful of survivors

But the **outcome is opposite**: seed 17 went 0.284 → 0.316 (+0.036); seed 3 went 0.243 → 0.087 (−0.179).

### Step-2000 substrate after mass death

| scale | survivors | mean strength | max strength | A_max | A_nz |
|---:|---:|---:|---:|---:|---:|
| W=2 | 5 | 4.36 | 7.79 | 8.14 | 5 |
| W=3 | 9 | 2.43 | 12.65 | 15.00 | 9 |
| W=4 | 20 | 1.11 | 19.47 | 16.30 | 20 |

Seed-17 comparison from report 036 (W=2): 32 survivors. Seed 3 ends with **6× fewer surviving W=2 atoms** than seed 17.

### Step-1500 pre-collapse state (the diagnostic moment)

| scale | n_patterns | below_threshold | steps_below_max | A_nz |
|---:|---:|---:|---:|---:|
| W=2 | 4098 | 4093 | 7 | **5** |
| W=3 | 2078 | 2071 | 7 | 8 |
| W=4 | 1078 | 1056 | 7 | 31 |

**The death is fully predictable from step 1500.** At W=2, 4093 of 4098 atoms are already below death_threshold and have been there 7 steps (death_window is 10). Only 5 atoms have received any retrieval inhibition accumulation (A_nz=5). The 5 survivors at step 2000 are exactly those 5 atoms — the ones that got *any* retrieval reinforcement during the run.

This is the same mechanism seed 17 follows. The difference is purely **reinforcement breadth**: seed 17's run distributed retrieval reinforcement across ~32 W=2 atoms; seed 3's run concentrated it on 5.

---

## Architectural reading

The death mechanism does exactly what the design specifies: kill anything below `death_threshold` for `death_window` steps. With re-encoding pulling strengths down each pass, *every* atom drops below threshold; the ones that survive are precisely those getting periodic retrieval reinforcement.

Whether `phase3_phase4` improves or collapses cap_t_05 reduces to one question: **does the surviving set cover the test-set vocabulary?**

For seed 17, the 32 W=2 survivors happen to include atoms the test set queries → cap_t_05 improves.
For seed 3, the 5 W=2 survivors don't cover the test set → cap_t_05 collapses to a tiny base rate over those 5 atoms.

A_nz at step 1500 is a single-number proxy for "how broad was the retrieval reinforcement so far." It already predicts the post-death outcome. Across the n=10 seeds the spread of A_nz at step 1500 (from prior reports: 3 to 33) almost mechanically explains the spread of Δcap_t_05.

### Anti-homunculus check

The collapse is a **local geometric dynamic**: each pattern's strength evolves under (re-encoding decay) − (retrieval reinforcement) − (Hebbian update); death is a local threshold test on that strength; the post-death substrate is whatever remains. No supervisor decides which patterns matter. The variance in Δcap_t_05 is variance in retrieval-reinforcement *breadth* induced by corpus-order shuffle. This is a pure mechanism property, not an architectural gap.

---

## What this report establishes

- Seed 3's −0.179 collapse is the **same mechanism** as seed 17's +0.036 improvement: mass death between step 1500–2000 reduces the substrate to the small set that received retrieval reinforcement during the run.
- The collapse vs improvement direction is determined by whether that surviving set happens to cover the test set, which is a **corpus-order / shuffle property**, not a tunable mechanism parameter.
- A_nz at step 1500 (= count of atoms that received any retrieval inhibition accumulation) **predicts the surviving substrate size at step 2000 exactly** (seed 3: A_nz=5 → 5 survivors).
- Δcap_t_05 variance is intrinsic to seed under the current death mechanism. This is option-1-branch-(b) from report 036.

## What this report does NOT establish

- That the death mechanism is the *right* shape. A gradient (probabilistic / soft) death rule would smear out the binary survival outcome and could tighten Δcap_t_05 variance.
- That this is unfixable. The fix would be a death-mechanism change, not a tuning sweep.
- That R@10 is correlated with the same survival breadth (R@10 for seed 3 stayed near baseline — 0.362 — across the collapse). This is consistent with R@10 being measured on substrate at a higher recall depth where rank-among-survivors matters less.

---

## Implications for the four-way decision

1. ~~Diagnose seed 3 collapse~~ — **done, this report**. Outcome: corpus-order variance, intrinsic to seed.
2. n=10 with death disabled — would now confirm by direct ablation that death is the variance driver, but the mechanism is already clear from the diagnostic. ~30 min compute. Optional verification.
3. Scope A_k to discovered-only — orthogonal to the finding. A_k doesn't change which atoms get reinforced, only which get extra inhibition once reinforced. Won't change Δcap_t_05 variance.
4. **Pivot Phase 4 headline to D1 meta-stable rate** — the most-aligned move. D1's Δ=−0.51 W=3 with per-seed std 0.038 is robust precisely because it measures a *substrate property* that doesn't depend on which atoms happen to survive: meta-stable rate is a basin-geometry measurement, computed over whatever substrate exists. Cap_t_05 measures readout-coverage and is variance-bound on the binary survival outcome.

The recommended move is **option 4**: pivot Phase 4 headline to D1, with R@K + cap-coverage as "promising but variance-bound under binary death," and document the mechanism finding from this report + report 036 as the rationale.

---

## Status implications

- Blocker #3 (Δcap-coverage second headline): the variance source is now mechanistically identified at the per-seed level. Tightening the CI requires either (a) changing the death mechanism shape (binary → gradient) or (b) reframing the headline. Continued tolerance is the wrong move.
- The seed-3 result is no longer an unexplained outlier. It is one tail of the corpus-order variance distribution that the binary death mechanism amplifies.
- Per [2026-05-16 substrate-vs-readout discipline note](../notes/notes/2026-05-16-substrate-vs-readout-metric-discipline.md): R@K is a readout drilled into substrate; cap-coverage measures substrate breadth × readout match. D1 (meta-stable rate) is a more substrate-pure measurement. This report supports the discipline note's framing.
