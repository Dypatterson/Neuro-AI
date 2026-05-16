# Report 035 — Saighi A_k n=10 verification: falsifies report 034

**Date:** 2026-05-16
**Active phase:** 4
**Headline metric per [phase-4-unified-design.md:276-282](../notes/emergent-codebook/phase-4-unified-design.md):** Δ Recall@K AND Δ cap-coverage with active codebook drift between cycles.
**Required controls per [phase-4-unified-design.md:318-323](../notes/emergent-codebook/phase-4-unified-design.md):** No-replay baseline (Condition A) — present in every seed. Random-codebook control — verified once at report 026, not re-run here.
**Last verified result:** [Report 034](034_saighi_ak_seed1_prototype.md) — A_k seed-1 prototype reported all three Phase 4 headlines improving (top1 +0.009, R@10 +0.022, cap_t05 +0.022).
**Why this experiment now:** STATUS.md blocker #2 was reshaped after report 034 to "verify A_k across n=10 seeds at the canonical seed set {17, 11, 23, 1, 2, 3, 5, 7, 13, 19}". Report 032 had already shown that n=5 produces *sample-lucky* results (report 029's +0.0145 became +0.009 with CI crossing zero at n=10). Same discipline applied here: a single-seed result for an architectural mechanism must be cross-validated at n=10 before it can graduate.

---

## Method

Identical to report 034 but at n=10 seeds and n_cues=3000.

```
--updater-kind hebbian --device cuda
--success-threshold 0.3 --death-threshold 0.05 --death-window 10
--inhibition-gain 0.01 --inhibition-decay 0.0
--n-cues 3000 --checkpoint-every 500 --no-reencode-discovered
```

n_cues was doubled from prior runs (1500 → 3000) per report 033's finding that patterns reach equilibrium at ~step 1200 and the prior 1500-cue setting was cutting off in the transient.

Ran on Colab H100 NVL (95 GB), 10 seeds launched in parallel. Wall-clock 18 min (vs. estimated 170 min sequential). Each worker ~1.6 GB GPU memory; aggregate peak 16.5 GB / 95 GB.

Note for posterity: the parallel-on-Colab path required two debug rounds documented in [notes 2026-05-15](../notes/notes/2026-05-15-saighi-hrr-replay-synthesis.md) and the memory file `colab_workflow.md` — (a) parent kernel must not touch CUDA before spawning workers, and (b) `experiments/19` must be invoked with `--device cuda` explicitly because the FHRR substrate's auto-detect picks MPS-or-CPU only.

Raw results: `/MyDrive/neuro-ai/results/phase34_saighi_n3k_seed{N}/phase34_results.json` × 10 seeds. (Drive folder IDs in conversation transcript 2026-05-16.)

---

## Result

### Per-seed (final checkpoint, step=2500 of Condition C)

| seed | A top1 | C top1 | Δtop1 | A R@10 | C R@10 | ΔR@10 | A cap | C cap | Δcap | A_max | A_nz | deaths | drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 17 | 0.129 | 0.062 | **−0.067** | 0.320 | 0.342 | +0.022 | 0.280 | 0.316 | +0.036 | 8.30 | 31 | 7200 | small |
| 11 | 0.110 | 0.014 | −0.095 | 0.362 | 0.371 | +0.010 | 0.238 | 0.233 | −0.005 | 5.52 | 8 | 7207 | small |
| 23 | 0.079 | 0.048 | −0.031 | 0.362 | 0.349 | −0.013 | 0.240 | 0.210 | −0.031 | 8.53 | 7 | 7206 | small |
| 1  | 0.079 | 0.132 | **+0.053** | 0.339 | 0.366 | +0.026 | 0.211 | 0.229 | +0.018 | 13.61 | 3 | 7217 | small |
| 2  | 0.114 | 0.032 | −0.082 | 0.291 | 0.295 | +0.005 | 0.227 | 0.164 | −0.064 | 8.16 | 7 | 7218 | small |
| 3  | 0.115 | 0.055 | −0.060 | 0.372 | 0.362 | −0.009 | 0.266 | 0.087 | **−0.179** | 8.14 | 5 | 7220 | small |
| 5  | 0.126 | 0.112 | −0.014 | 0.285 | 0.276 | −0.009 | 0.243 | 0.215 | −0.028 | 3.52 | 33 | 7205 | small |
| 7  | 0.142 | 0.017 | **−0.125** | 0.345 | 0.371 | +0.026 | 0.276 | 0.328 | +0.052 | 8.11 | 5 | 7214 | small |
| 13 | 0.102 | 0.047 | −0.056 | 0.312 | 0.330 | +0.019 | 0.251 | 0.205 | −0.047 | 13.15 | 4 | 7214 | small |
| 19 | 0.076 | 0.040 | −0.036 | 0.305 | 0.318 | +0.014 | 0.139 | 0.130 | −0.009 | 13.12 | 6 | 7217 | small |

### Aggregates (mean ± 95% t-CI, 9 df, t_crit = 2.262)

| metric | mean | 95% CI | pos/10 | report 034 (seed 1) had |
|---|---:|---|---:|---|
| **ΔR@10 (C−A)** | +0.0089 | [−0.0019, +0.0197] | 7/10 | +0.0220 |
| **Δtop1 (C−A)** | **−0.0512** | **[−0.0862, −0.0162]** | **1/10** | +0.0088 |
| Δcap_t05 (C−A) | −0.0256 | [−0.0720, +0.0207] | 3/10 | +0.0220 |
| ΔR@10 (C−B) | +0.0021 | [−0.0073, +0.0116] | 4/10 | (n/a) |

---

## Three falsifications of report 034 in this single run

**1. ΔR@10 is statistically indistinguishable from baseline.** Mean +0.0089 (CI [−0.002, +0.020]) is identical to [report 032](032_phase34_n10_verification.md)'s n=10 baseline (+0.009, CI [−0.003, +0.021]) within numerical noise. A_k does not improve R@10.

**2. Δtop1 is robustly NEGATIVE under A_k, not positive.** CI is strictly negative ([−0.086, −0.016]). Only 1/10 seeds (seed 1) shows positive Δtop1. Seed 1 was the only positive seed — exactly the seed report 034 was built around. Report 034's "A_k reverses blocker #6'" claim was a single-seed lottery.

**3. Δtop1 is WORSE with A_k than without:** −0.0512 here vs. −0.018 in report 032 baseline. A_k **amplifies** the top1 regression rather than fixing it.

These three findings are mutually reinforcing and not within rounding error.

---

## What the mechanism actually did

Telemetry shows A_k fired as designed:
- A_max per seed: 3.5 to 13.6
- A_nz per seed: 3 to 33 patterns inhibited
- Deaths fired: 7200+ per seed (mass-death of un-touched vocab atoms — as predicted by [report 033](033_phase4_death_mechanism_diagnostic.md) for n_cues=3000)

Honest mechanistic reading: **A_k pushes probability mass off the most-frequently-retrieved attractor toward less-used ones**. The intended consequence (broader exploration, more discovered patterns) is real but smaller than the side-effect (the correct top-1 token *was* one of those dominant attractors). Pushing mass off it broadens the neighborhood marginally (R@10 +0.009) while moving rank-1 onto neighbors that aren't the right token (top1 −0.051).

This is blocker #6's warning, amplified by the very mechanism we hoped would resolve it.

### Why seed 1 was an outlier in the prototype

Of the 10 seeds, seed 1 has the most concentrated A_k accumulation: A_max=13.6 but only A_nz=3 (compare to seed 5: A_max=3.5, A_nz=33; seed 17: A_max=8.3, A_nz=31). Seed 1's three over-used attractors get heavily inhibited and the system has to find replacement attractors — those happen to coincide with the correct tokens for that seed's test set, producing a +0.053 Δtop1.

For most seeds, the inhibition spreads across more patterns, and the replacement attractors don't coincide with correct tokens. Result: top1 drops across the board.

---

## A_k mechanism FEP audit (unchanged from report 034)

The mechanism itself remains FEP-clean. The problem is not the mechanism's *shape* but its *parameterization* and its interaction with the corpus. The CI-disjoint-negative Δtop1 means the parameterization produces consistently wrong dynamics for *this* task at *this* scale.

---

## What this report establishes — and does NOT establish

**Does establish:**
- Saighi A_k at gain=0.01, decay=0.0, applied uniformly to all patterns, at n_cues=3000, does NOT improve any of the three Phase 4 headlines vs. baseline.
- The mechanism robustly *worsens* Δtop1 (CI strictly negative, 1/10 positive).
- Report 034's optimistic seed-1 result was a sample-lucky outlier driven by the specific A_k accumulation pattern on that seed.
- Death now fires at n_cues=3000 (`deaths_total ~7200`), exactly as report 033 warned: mass-death of un-reached vocab atoms.

**Does NOT establish:**
- That A_k at *other* parameterizations is also bad. Saighi p. 4 explicitly notes gain-vs-decay as a trade-off knob; this run set decay=0 (monotonic growth). The next experiment in this session is the decay sweep ([scripts/colab_phase34_saighi_decay_sweep.ipynb](../scripts/colab_phase34_saighi_decay_sweep.ipynb), running concurrently with this writeup).
- That A_k scoped to discovered-patterns-only is also bad. The current implementation applies uniformly to initial vocab atoms AND discovered patterns. The architectural alternative (discovered-only) is implementable but not yet tested.
- That the Phase 4 architecture is fundamentally broken. The discovery channel still emits ~80 candidates per seed; the D1 meta-stable rate result from [report 026](026_phase4_verification_design_spec.md) (Δ=−0.51, std=0.038) remains robust and unrelated to A_k.

---

## Status implications

- **Blocker #2 stays open.** The Saighi A_k mechanism, as implemented and parameterized, does not close it. Three branches remain:
  - Decay sweep (in progress this session): does Saighi's intended forgiveness window restore positive Δtop1?
  - Discovered-only scoping: limit A_k to discovered patterns; protects vocab atoms from inhibition (~1h to wire + n=10 verify).
  - Accept and reframe: D1 meta-stable rate remains the most robust Phase 4 evidence; pivot Phase 4 headline there.
- **Blocker #6' (top1 regression) confirmed and *worsened* under A_k.** The regression is not just inherent to online Hebbian — it's amplified by the new mechanism. Investigation should now focus on whether the rank-1-vs-neighborhood tradeoff is fundamental to *any* basin-narrowing mechanism at this corpus size.
- **Blocker #5 (seed 23 outlier) persists.** Seed 23: Δtop1 = −0.031, ΔR@10 = −0.013. Less extreme than under prior runs (its outlier-ness softens at n=10 with A_k) but still on the negative side of mean.
- **No change to blockers #3, #4, #7.**

---

## What this run cost

The path to clean parallel execution on Colab took two debug iterations:
1. Workers initially hung because the parent kernel held CUDA exclusively. Fix: parent uses `device='cpu'` only.
2. Workers ran on CPU (silent fallback in substrate auto-detect) at 17 min/seed. Fix: pass `--device cuda` to subprocess.

Both fixes are documented in user memory (`colab_workflow.md`) and the notebook itself. The substrate's `torch_fhrr.py:31` auto-detect should arguably be patched to include CUDA, but that's a separate commit.

Net wall-clock for the n=10 sweep: ~18 min on H100 NVL with 10-way parallel. Cost-justified given the result is a clean falsification of report 034.
