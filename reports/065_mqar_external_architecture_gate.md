# Report 065 — MQAR External Architecture-Gate Discriminator

> **⚠ Partial supersession 2026-05-24.** The GHRR / bundle-first follow-up
> recommendation at the bottom of this report ("recommend GHRR /
> bundle-first discriminator" in the decision banner; "(c.1) GHRR
> substrate rebuild" in the §Implications section) is **superseded by
> the rewritten [Report 066](066_mqar_ghrr_bundle_first_discriminator.md)**.
> Report 066's diagnostic-instrumented re-run shows the GHRR algebra
> change does **not** rescue key-only MHN basin retrieval (
> `top_index_hits = 0/3072` at every N); only the bundle-first
> architecture change does. Read Report 065 for its substrate-shape
> diagnosis (still valid) and Report 066 for the corrected
> architectural-commit framing.

**Date:** 2026-05-24
**Active phase:** 5 (but **this is not a Phase 5 graduation experiment** — see framing below)
**Status:** Production run complete at D=4096, n_queries=1024, 3 seeds × 6 N values × 3 storage strategies (=55,296 trials). Discriminator confirms the substrate-shape diagnosis raised by Reports 062/063/064 and the 2026-05-24 brainstorm smoke.
**Decision:** Phase 5 stuck-state is consistent with a **substrate-shape** problem (key-only Hopfield retrieval scales toward chance), not a retrieval-tuning problem. M2 (training-time intervention) is now in question as a productive next step — it would be trying to train basin geometry that the substrate's binding algebra makes geometrically unreachable. ~~Recommend GHRR / bundle-first discriminator (Report 066, conditional task #4) before any M2 commitment.~~ **Superseded — see banner above. Report 066's actual finding: bundle-first rescues; GHRR does not.**

---

## Framing — what this experiment is and is not

The Phase 5 headline per [phase-5-unified-design.md:280-299](../notes/emergent-codebook/phase-5-unified-design.md) is:

> ΔE = E_content_prior − E_role_prior, paired per cue, magnitude floor 5.5e-3, CI > 0.

**This report does not measure ΔE.** Adopting MQAR as the Phase 5 graduation criterion would require an explicit spec amendment; this report does not propose one. MQAR is framed here as an **external architecture-gate discriminator**: a third-party benchmark that the substrate either passes or fails, used to disambiguate the diagnosis from the four-family null (D1/D3/E1/M1, Reports 062/063/064).

The question the discriminator answers: *does the FHRR + plain-Hebb Modern Hopfield substrate have key-only associative recall basins at all?* If the answer is no, the four retrieval-side mechanism nulls are predictable from substrate geometry alone, and the architectural-commit decision in [STATUS.md](../STATUS.md) blocker #3 has new evidence in favor of an algebra/storage pivot over an M2 training intervention.

**Why this experiment now:** The 2026-05-24 unconsidered-paths brainstorm smoke ([smoke_c_mqar.py](../brainstorm-workspace/2026-05-24-unconsidered-paths/smoke_c_mqar.py)) reported HRR-bundle at 100% top-1 and Hopfield key-only at 0% at D=4096, N=128, n_queries=32. The user (in collaboration with the codex review) ranked this as the cleanest available discriminator before committing to M2. This report runs the smoke at production scale (n_queries=1024, 3 seeds) to confirm the qualitative finding survives proper statistics.

---

## Setup

- Harness: [`experiments/42_mqar_external_gate.py`](../experiments/42_mqar_external_gate.py)
- Substrate: `TorchFHRR(dim=4096)` on MPS, fresh substrate per (seed, N) cell
- Storage strategies (registered in `_STRATEGIES`):
  - `hrr_bundle` — positive control. Bundle `Σ bind(k_i, v_i)`, recover via `unbind(M, k_j)`, top-1 over value codebook. Proves the task is **solvable on this substrate at this scale**.
  - `hopfield_perfect_cue` — memorization control. Store each `bind(k_i, v_i)` as an MHN pattern; cue is the bound pair itself. Proves **storage works**.
  - `hopfield_key_only` — discriminator. Same storage; cue is `k_j` alone. The substrate's MHN landscape must find the bound pattern by key alignment.
- Settings: `beta=30.0` (canonical retrieval per [STATUS.md "HAM regime split"](../STATUS.md)), max_iter=10, n_queries=1024, seeds=[17, 11, 23], N ∈ {16, 32, 64, 128, 256, 512}
- Raw payload: [`reports/phase5_mqar_external_gate/results.json`](phase5_mqar_external_gate/results.json)
- Total trials: 3 strategies × 6 N values × 3 seeds × 1024 queries = **55,296**
- Wall time: 6:47 on M-series MPS

---

## Headline (this experiment, not Phase 5)

Top-1 recall at D=4096 with Wilson 95% CI across (seeds × queries), 3072 trials per cell:

| N | HRR-bundle (pos. control) | Hopfield perfect-cue (memorization) | **Hopfield key-only (discriminator)** | Chance (1/N) |
|---:|:---:|:---:|:---:|:---:|
| 16  | 1.0000 [0.999, 1.000] | 1.0000 [0.999, 1.000] | **0.1113 [0.101, 0.123]** | 0.0625 |
| 32  | 1.0000 [0.999, 1.000] | 1.0000 [0.999, 1.000] | **0.0596 [0.052, 0.069]** | 0.0313 |
| 64  | 1.0000 [0.999, 1.000] | 1.0000 [0.999, 1.000] | **0.0264 [0.021, 0.033]** | 0.0156 |
| 128 | 1.0000 [0.999, 1.000] | 1.0000 [0.999, 1.000] | **0.0169 [0.013, 0.022]** | 0.0078 |
| 256 | 0.9876 [0.983, 0.991] | 1.0000 [0.999, 1.000] | **0.0094 [0.007, 0.014]** | 0.0039 |
| 512 | 0.6914 [0.675, 0.708] | 1.0000 [0.999, 1.000] | **0.0023 [0.001, 0.005]** | 0.0020 |

**Discriminator outcome:** Hopfield key-only is **above chance at low N but the gap collapses with N**. The ratio top-1 / chance is 1.78×, 1.90×, 1.69×, 2.16×, 2.41×, 1.15× for N = 16, 32, 64, 128, 256, 512 respectively. By N=512 the CI ([0.001, 0.005]) brackets the chance line (0.0020), meaning at production scale the discriminator returns **statistically indistinguishable-from-chance** retrieval.

---

## Refinement of the smoke's claim

The 2026-05-24 brainstorm smoke reported "Hopfield key-only collapses from 12.5% at N=16 to 0% by N=128." Production statistics sharpen this:

1. **The "0%" was an artifact of n_queries=32 noise** — at 3 seeds × 32 queries = 96 trials per N, the Wilson CI on 0% is [0%, 4%], which is consistent with the true value of 1.7% at N=128.
2. **There is a small-but-real key-only signal at low N.** N=16 returns 11.1% (CI [10.1%, 12.3%]) against chance 6.25% — this is **above chance with very high confidence** (chance is far below the CI's lower bound). So the substrate is not literally "no key-only basins"; it has weak, residual basins that the noise of FHRR phase alignment can produce at low capacity.
3. **The signal decays toward chance with N.** By N=512 the discriminator is at chance. The decay matches the geometric intuition: `bind(k, v)` in FHRR is roughly orthogonal to `k` alone (cosine ≈ 1/√D), so the energy landscape near `k` has no preferential basin, and adding more patterns dilutes whatever weak structure exists.
4. **HRR-bundle and Hopfield perfect-cue work fine.** This is the crucial frame: the task is **solvable** on this substrate (HRR-bundle: 100% up to N=128, 98.8% at N=256, 69.1% at N=512 — matching Plate's theoretical bundle capacity bound). And the substrate **stores** correctly (perfect-cue: 100% throughout). The failure is specifically in the **key-alignment retrieval geometry** of MHN over `bind(k, v)` patterns.

The qualitative diagnosis from the smoke and from Reports 062/063/064 stands: **Hopfield retrieval over bound vectors is not addressable by a single factor of the bind.** The four Phase 5 retrieval-mechanism nulls are downstream of this geometric fact.

---

## Drill-downs

### D-1: Bundle ceiling matches Plate's bound
HRR-bundle drops from 100% (N=128) → 98.8% (N=256) → 69.1% (N=512) at D=4096. The Plate capacity bound for a unit-magnitude FHRR bundle is roughly `N_max ≈ D / (4 log(D · n_codebook))`, which at D=4096 sits around N≈256–512. The break-point matches; the bundle is operating at the substrate's algebraic capacity ceiling, not at an implementation-bug threshold. This is **drill-down evidence that the substrate dimension itself is not the bottleneck** for the discriminator at N≤128.

### D-2: Perfect-cue holds at 100% — storage is fine
At every N up to 512, perfect-cue is 100%. The MHN landscape can store and retrieve 512 patterns at D=4096 with zero failure when the cue is the pattern itself. This is **drill-down evidence that the failure mode is specifically in the cue→pattern geometry, not in pattern→pattern recall.**

### D-3: Per-seed variance
At N=16 the per-seed key-only top-1 values are [0.197, 0.074, 0.062] — seed 17 is an outlier high. At N≥64 the variance is small. This drill-down explains why the smoke (seed 17 only at N=16) reported 25% — seed 17 is genuinely 19.7% at production scale, but the cross-seed mean is 11.1%. Seed-specific FHRR phase patterns can produce moderately aligned `bind(k, v) ↔ k` pairs by chance at small N.

### D-4: Above-chance ratio
For N ∈ {16, 32, 64, 128, 256, 512}: top-1 / chance = {1.78, 1.90, 1.69, 2.16, 2.41, 1.15}×. The ratio rises slightly up to N=256 then collapses — at small N, what little structure exists registers; at large N, dilution overtakes structure. This drill-down rules out the "key-only signal is meaningful but small" framing — it's not a consistent ~2× scaling that would suggest a real but weak mechanism; the ratio is non-monotone and converges to 1× at N=512.

---

## Implications for the architectural-commit decision

[STATUS.md blocker #3](../STATUS.md) lists three legitimate options:

- **(a) Commit M2 (training-time EqProp + role-shuffled-negatives + DSM warm-start) on a new branch.** This report makes (a) less attractive: M2 would be trying to train **role basins** into a substrate whose binding algebra makes single-factor retrieval geometrically marginal at all relevant N. Training cannot reshape `bind(k, v)`'s relationship to `k` — that's a property of the elementwise complex product.
- **(b) Accept Phase 5 closure / pivot per the 2026-05-23 SNR walk-back and the Path B' framing.** This report supports (b) in the specific sense that the substrate, as currently shaped, does not have the geometric primitive Phase 5 was trying to extract. A closure report should cite this discriminator alongside Reports 062/063/064.
- **(c) Try the dual-code GHRR P1 variant before M2 (algebra change rather than training change).** ~~This report **strengthens (c)**: the diagnosis is at the algebra layer, so the appropriate response is an algebra change. GHRR (non-commutative binding) is the smallest-blast-radius substrate-algebra variant that could rescue key-only addressability without re-doing the Phase 4 substrate from scratch.~~ **Superseded by [Report 066](066_mqar_ghrr_bundle_first_discriminator.md):** the GHRR-native variant returns `top_index_hits = 0/3072` at every N; the algebra change does **not** rescue key-only MHN basin retrieval at MQAR diagnostic scale. The rescue that *does* work is the architecture change (bundle-first storage + MHN cleanup over the value codebook), which reuses the FHRR substrate. See [Report 066](066_mqar_ghrr_bundle_first_discriminator.md) §Implications for the corrected option ranking.

~~**My read for the architectural-commit decision:** the conditional task #4 (GHRR / bundle-first discriminator on this same harness) is now the highest-value next experiment. It is a one-day plug-in to the strategy registry, and its result decides between (b) and (c). I recommend running it before any architectural commit. Reflected in the task list.~~ **Resolved by Report 066:** (c.1) GHRR is demoted; (c.2) bundle-first is the sole positive rescue at MQAR scale. The architectural-commit decision now lives in [STATUS.md](../STATUS.md) blocker #2 with updated weightings.

---

## Done-gate compliance

Per [CLAUDE.md "What 'done' looks like for an experiment"](../CLAUDE.md):

1. ✅ **Headline metric reported with confidence intervals.** Wilson 95% CI on every cell, 3072 trials per cell. Tabulated above.
2. ✅ **Control conditions on the same test set.** HRR-bundle (positive control, task is solvable here) and Hopfield perfect-cue (memorization control, storage works here) run on the same FHRR substrate, same keys, same values, same query indices per seed.
3. ✅ **Drill-down metrics explain anomalies.** D-1 explains the bundle ceiling; D-2 explains why "storage" is not the failure mode; D-3 explains the smoke's 25% at seed 17; D-4 rules out the "small but real" framing.
4. ✅ **Written up as a markdown report under `reports/`.** This file.
5. ✅ **Status note updated.** See STATUS.md `Recent updates` entry for 2026-05-24.

The Phase 5 control matrix from [phase-5-unified-design.md:309-314](../notes/emergent-codebook/phase-5-unified-design.md) (random-schema, K=1, γ=0, no-schema-store) does **not** apply here — this is not a Phase 5 graduation experiment and the controls listed there are for the ΔE headline, which this report does not measure.

---

## Anti-homunculus check

No new mechanism is proposed here. The harness measures a property of the existing substrate. No `if X then do Y` rule, no arbitration. The discriminator outcome **informs the user's architectural decision** — it does not gate any code path or trigger any mechanism. ✅

---

## Files

- Experiment script: [`experiments/42_mqar_external_gate.py`](../experiments/42_mqar_external_gate.py)
- Raw results: [`reports/phase5_mqar_external_gate/results.json`](phase5_mqar_external_gate/results.json)
- Brainstorm-scale smoke (predecessor): [`brainstorm-workspace/2026-05-24-unconsidered-paths/smoke_c_mqar.py`](../brainstorm-workspace/2026-05-24-unconsidered-paths/smoke_c_mqar.py) and [`brainstorm-workspace/2026-05-24-unconsidered-paths/test-results.md`](../brainstorm-workspace/2026-05-24-unconsidered-paths/test-results.md)
