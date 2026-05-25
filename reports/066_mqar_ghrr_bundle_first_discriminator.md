# Report 066 — MQAR Discriminator: Bundle-First Confirmed, GHRR Claim Walked Back

**Date:** 2026-05-24 (rewritten 2026-05-24 after codex review)
**Active phase:** 5 (this is **not a Phase 5 graduation experiment** — see [Report 065](065_mqar_external_architecture_gate.md) framing)
**Status:** **This report has been rewritten.** The original 2026-05-24 version claimed both GHRR matrix-binding and bundle-first rescued key-only retrieval. Codex review caught that the GHRR cell was structurally confounded: it stored flattened unitary matrices in `TorchHopfieldMemory`, whose per-element normalization and mean-real-product similarity are wrong-shape for the GHRR algebra. The MHN settled to near-uniform weights (entropy ≈ 1.0, `top_index_hits ≈ 0`) and the apparent "100% top-1" came from algebraic unbind over a bundle-like superposition, not from basin retrieval. A diagnostic-instrumented re-run and a corrected `ghrr_matrix_key_only_native` (polar-decomposition unitary retraction + Frobenius similarity) confirm: **the only positive rescue is `bundle_first_key_only`. GHRR algebra change does not rescue key-only Hopfield basin retrieval.**

---

## Framing — what this experiment is and is not

The Phase 5 headline per [phase-5-unified-design.md:280-299](../notes/emergent-codebook/phase-5-unified-design.md) is `ΔE = E_content_prior − E_role_prior` with magnitude floor 5.5e-3 and CI > 0 at n_seeds ≥ 10 (verified standard per [phase-5-checklist.md:10-16](../notes/emergent-codebook/phase-5-checklist.md)). **This report does not measure ΔE and is not n ≥ 10.** It is an n=3 external diagnostic at MQAR-style smoke scale. Treat its conclusions as architecture-gate signal, not as Phase 5 graduation evidence.

**Why the rewrite was needed.** The diagnostic-instrumented re-run was triggered by codex's external probe, which surfaced that `top_index_hits = 0/64` and mean normalized weight entropy = 0.9996 at N=128, seed=17 — meaning the Hopfield settling never found the right basin and the headline "100% top-1" was downstream algebraic decode over a near-uniform mixture. The harness's `top1` field cannot distinguish "MHN basin retrieval" from "bundle-style algebraic unbind from a uniform-mixture state." Without the three new diagnostics — `top_index_hits`, weight entropy, score margin — the original report could only have reported `top1` and would have concluded the wrong thing.

---

## Headline (this experiment, not Phase 5)

D=4096, n_queries=1024 per (strategy, N, seed), 3 seeds, n=3072 trials per cell, Wilson 95% CI on top-1. Diagnostics: `tix` = `top_index_hits` (queries where MHN's pre-decode `argmax` matched the cued pair index; the fraction `/ 3072` is the true basin-retrieval rate); `ent` = mean normalized weight entropy (0 = sharp basin, 1 = uniform mixture); `margin` = mean (top score − 2nd score). All raw payloads in [results_with_diagnostics.json](phase5_mqar_external_gate/results_with_diagnostics.json) and [results_ghrr_bundle_diagnostics.json](phase5_mqar_external_gate/results_ghrr_bundle_diagnostics.json).

### Headline table

| N | FHRR Hopfield key-only | GHRR matrix (confounded) | **GHRR matrix native** | **Bundle-first** |
|:--:|:--:|:--:|:--:|:--:|
| 16  | top1=0.1113 / tix=128 / ent=0.000  | top1=1.0000 / tix=186 / **ent=0.996** | top1=0.1110 / **tix=0** / ent=0.000 | top1=1.0000 / **tix=3072** / ent=0.000 |
| 32  | top1=0.0596 / tix=80  / ent=0.000  | top1=1.0000 / tix=92  / **ent=0.999** | top1=0.0319 / **tix=0** / ent=0.000 | top1=1.0000 / **tix=3072** / ent=0.000 |
| 64  | top1=0.0264 / tix=0   / ent=0.000  | top1=1.0000 / tix=44  / **ent=1.000** | top1=0.0117 / **tix=0** / ent=0.000 | top1=1.0000 / **tix=3072** / ent=0.000 |
| 128 | top1=0.0169 / tix=34  / ent=0.000  | top1=1.0000 / tix=28  / **ent=1.000** | top1=0.0104 / **tix=0** / ent=0.000 | top1=1.0000 / **tix=3072** / ent=0.000 |
| 256 | top1=0.0094 / tix=11  / ent=0.000  | top1=0.9408 / tix=13  / **ent=1.000** | top1=0.0023 / **tix=0** / ent=0.000 | top1=0.9827 / **tix=3019** / ent=0.000 |
| 512 | top1=0.0023 / tix=3   / ent=0.000  | top1=0.6527 / tix=6   / **ent=1.000** | top1=0.0010 / **tix=0** / ent=0.000 | top1=0.6146 / **tix=1888** / ent=0.000 |

### The discriminator outcome (with diagnostics)

- **`bundle_first_key_only` is the only positive rescue.** `top_index_hits = 3072/3072` at N ≤ 128, 3019/3072 at N=256, 1888/3072 at N=512 — meaning Hopfield is genuinely finding the right basin in the **value codebook**. Entropy is 0 throughout (sharp basin). This is the real signal.
- **`ghrr_matrix_key_only` (the original Report 066 GHRR cell) is bundle-style decode, not basin retrieval.** `top_index_hits` is 186, 92, 44, 28, 13, 6 out of 3072 — at most 6% basin rate even at N=16, dropping to 0.2% at N=512. Entropy is ≈ 1.0 (near-uniform) across all N. The high `top1` numbers are produced by `K^H @ state_mat` after Hopfield settles to a roughly-bundle-like state — *not* by Hopfield basin retrieval.
- **`ghrr_matrix_key_only_native` (corrected GHRR primitives — polar-decomposition unitary retraction + Frobenius similarity) does not rescue.** `top_index_hits = 0/3072` at every N. `top1` falls into the same range as FHRR Hopfield key-only: 11.1% at N=16 (≈ FHRR's 11.1%), down to 0.1% at N=512 (below FHRR's 0.2%). Entropy = 0 means the settling is locking onto a sharp basin — but it's the **wrong** basin in 100% of queries. The Frobenius-similarity MHN landscape over bound matrices does not have a basin near `K`.
- **FHRR Hopfield key-only diagnostics refine Report 065.** Where Report 065 saw top1=11.1% at N=16, the diagnostics show `tix=128/3072` (4.2% real basin retrieval) and ent=0 (sharp basin). The remaining 7% of "correct" top-1 comes from cases where Hopfield landed on a *wrong* pattern but the algebraic `unbind(state, key_qi)` recovered the correct value top-1 anyway. Entropy=0 across all N means FHRR Hopfield always settles sharply — just not on the right pattern.

---

## What the corrected story is

**The Phase 5 stuck-state diagnosis from Report 065 stands**: the FHRR substrate's elementwise `bind(k, v) = k * v` does not produce key-only addressable basins in the MHN landscape over bound pairs. Reports 062-066 form a falsification chain at *this diagnostic scale* (n=3 MQAR, not n=10 ΔE).

**What is new in this rewrite**: the **algebra change (GHRR) does not rescue** key-only MHN basin retrieval at MQAR diagnostic scale. The proper GHRR-native primitives still produce wrong-basin settling. The substrate-shape problem may be **deeper than which binding operator is used** — the MHN energy landscape over `bind(K, V)`-style patterns may simply not have key-only basins for any binding operator that produces high-dimensional approximately-random bound vectors. The only intervention tested here that works is the **architecture change**: change what is stored where. Bundle-first puts the bundled superposition Σ bind(k_i, v_i) in one slot and uses MHN over the *value codebook* (not over bound pairs) as a cleanup head.

This narrows the architectural-commit decision in [STATUS.md](../STATUS.md) blocker #2:

| Option | Status after this rewrite |
|---|---|
| **(a) M2 training-time on existing FHRR** | Not addressed by Report 066. Could still rescue **at the schema-store / prior-selection layer** (not at the algebraic bind layer). The retrieval-side falsification chain (Reports 062-064) was on retrieval-time interventions; M2 modifies training. Open. |
| **(c.1) GHRR substrate rebuild** | **Demoted.** GHRR-native at MQAR diagnostic scale shows no key-only basin retrieval (tix=0/3072 across N). If GHRR remains attractive on other grounds (multi-role compositional capacity, theoretical cleanliness), a properly-instrumented multi-role MQAR test is required first. The single-role MQAR result is null. |
| **(c.2) Bundle-first Phase-5 architecture** | **Sole demonstrated rescue.** `top_index_hits = 1.0` at N ≤ 128. Smallest-blast-radius change: reuses FHRR substrate, reuses MHN primitives, only reorganizes what is stored where. |

---

## Drill-downs

### D-1: Why GHRR-original looked like it worked
With patterns being flattened m×m unitary matrices stored in `TorchHopfieldMemory`, the substrate's per-element normalization (`fhrr.normalize` divides each element by its magnitude) destroys the unitary structure. The substrate's `similarity_matrix` returns `mean(real(p.conj() * q))`, which is the FHRR scalar similarity. For unitary-matrix patterns flattened to D-dim vectors, this similarity is dominated by random-phase coincidences, not by Frobenius structure. With β=30 it produces near-uniform softmax weights, so the "settled state" is essentially `mean(bound_flat)` ≈ bundle of all bound matrices. Then `unbind(K, mean_bundle) = K^H @ mean_bundle_mat = (1/N) Σ K^H @ K_i @ V_i`. The i = qi term gives `K_qi^H @ K_qi @ V_qi = V_qi` (because keys are unitary), and the other terms contribute random noise. So the top-1 over the value codebook gets V_qi with high probability — purely algebraically, with no contribution from MHN.

### D-2: Why GHRR-native fails (corrected after codex review)

**Earlier walk-back.** The previous version of this drill-down argued that the Frobenius score for cue `K_qi` against stored pattern `B_i = K_i @ V_i` reduces to `tr(V_i)` regardless of `i`, predicting a single fixed winner. That was algebraically wrong. The correct expression is `Re(tr(B_i^H @ K_qi))/m = Re(tr(V_i^H K_i^H K_qi))/m`. Only when `i = qi` does `K_qi^H K_qi = I` collapse the score to `Re(tr(V_qi^H))/m`. For `i ≠ qi` the score is `Re(tr(V_i^H @ U_random))/m` where `U_random = K_i^H K_qi` is a random unitary, and the score is a random variable with mean ≈ 0.

A direct probe (N=128, seed=17, n_queries=32) shows **21 unique winners across 32 queries** at the initial step and at the final settled step — not one fixed winner. Codex's external probe at the same parameters got the same count. So the failure mode is not "fixed deterministic loss" but "the correct-pair signal is buried in N-1 random-cross-product noise terms of comparable magnitude."

**The actual mechanism.** Per-pattern Frobenius scores have magnitude `O(1/m) = O(1/64) ≈ 0.016` for D=4096 — both the i=qi signal (`Re(tr(V_qi))/m`) and the i≠qi noise terms (`Re(tr(V_i^H @ U_random))/m`) live at this scale. With β=30 and these scores, softmax produces a peaked but not deterministic winner — one of the N patterns dominates per query, but **which one is determined by the noise lottery, not by the cue**. Across queries this means tix is near 1/N (chance level): expected 32/128 × hits = 0.25 at N=128, observed 0/3072 across the full production matrix (consistent with the binomial variance around 24 expected hits and the QR-induced bias in trace statistics).

The right summary: **the Frobenius MHN over `K_i @ V_i` patterns does not have a key-addressed basin** — its winner-take-all behavior is driven by a competition of O(N) noise terms with no preferential alignment toward the cued pair. This rules out the GHRR-native variant as a single-role MQAR rescue.

**What this does and doesn't rule out.** It rules out the *as-tested* GHRR-native cell at MQAR diagnostic scale: random Haar-style unitary keys + values, single-role pairs, Frobenius similarity + polar retraction. It does *not* rule out: (a) GHRR with learned (non-Haar) unitary codebooks where a correct-pair signal could be boosted by training; (b) other binding operators (block-circulant, residue-HDC, Hadamard-product variants); (c) multi-role binding where structural priors might rescue the noise problem. Those are not implemented; they're open questions.

### D-3: Why bundle-first works
Bundle-first stores `M = Σ bind(k_i, v_i)` as a single bundle vector (no MHN over bound pairs), and the MHN landscape is over the **value codebook** `{v_1, ..., v_N}`. The unbind step computes `unbind(M, k_qi)` which approximates `v_qi + noise` (Plate's HRR capacity argument). The MHN clean-up settles this noisy estimate to the nearest stored value — this is recall over a clean codebook, the operation MHN is actually good at (perfect-cue baseline is 100% throughout in Report 065). The tix=3072/3072 confirms the cleanup is doing real basin retrieval.

### D-4: FHRR Hopfield key-only — small basin signal exists, low cap
At N=16: tix=128/3072 = 4.17% basin rate; top1=11.1% includes ~7% of additional "lucky decodes" where Hopfield settled on a wrong pattern but `unbind(state, k_qi)` still produced a vector whose argmax over the value codebook was v_qi. At N=512: tix=3/3072 ≈ 0.1% basin rate, top1=0.2%. The basin rate falls faster than the decode rate. Entropy=0 throughout — FHRR Hopfield doesn't produce uniform mixtures like the confounded GHRR cell does. The substrate has weak, narrow basins that happen to land on the right pattern in ~4% of cues at N=16 (above the 6.25% chance for top1 but below it for tix). This is consistent with the Report 065 finding of "small above-chance signal at low N decaying to chance."

---

## Done-gate compliance

Per [CLAUDE.md "What 'done' looks like for an experiment"](../CLAUDE.md):

1. ✅ **Headline metric reported with CI.** Wilson 95% CI per cell, 3072 trials per cell. Diagnostics (`tix`, ent, margin) for every Hopfield-based strategy.
2. ✅ **Control conditions on same test set.** All 4 strategies + `hrr_bundle` + `hopfield_perfect_cue` run on same FHRR substrate (where applicable), same seed structure, same query indices per seed.
3. ✅ **Drill-down metrics explain anomalies.** D-1 explains the GHRR-original confound; D-2 the GHRR-native failure; D-3 the bundle-first success; D-4 refines the FHRR Hopfield key-only signal.
4. ✅ **Written up under `reports/`.** This file (rewritten).
5. ✅ **STATUS.md updated** — "Last verified result" → "Latest external diagnostic result"; banner explicitly walks back the GHRR claim.

The Phase 5 control matrix from [phase-5-unified-design.md:309-314](../notes/emergent-codebook/phase-5-unified-design.md) does not apply — this is not a Phase 5 graduation experiment.

---

## Anti-homunculus check

All four strategies measure a passive property of a substrate or storage configuration. No `if X then do Y` rule, no arbitration between subsystems. The downstream architectural-commit decision is the user's, informed by this evidence — not gated by any code path. ✅

---

## Files

- Experiment script: [`experiments/42_mqar_external_gate.py`](../experiments/42_mqar_external_gate.py) (strategy registry: 6 entries; `top_index_hits` / entropy / margin diagnostics on every Hopfield-based strategy; `ghrr_matrix_key_only` is now docstring-flagged `CONFOUNDED`; new `_ghrr_matrix_key_only_native` with proper algebra primitives)
- Raw results with diagnostics:
  - [`reports/phase5_mqar_external_gate/results_with_diagnostics.json`](phase5_mqar_external_gate/results_with_diagnostics.json) — Report 065 strategies re-run with new diagnostics
  - [`reports/phase5_mqar_external_gate/results_ghrr_bundle_diagnostics.json`](phase5_mqar_external_gate/results_ghrr_bundle_diagnostics.json) — GHRR-original, GHRR-native, bundle-first with diagnostics
- Pre-walk-back probe that triggered the rewrite: codex's external review identified `top_index_hits = 0/64`, ent=0.9996 at the original `ghrr_matrix_key_only` cell at N=128, seed=17, n_queries=64. Reproduced locally in `/tmp/ghrr_probe.py` (not committed).
- Superseded raw payloads (Report 066 original — kept for traceability of the walk-back): [`reports/phase5_mqar_external_gate/results_ghrr_bundle.json`](phase5_mqar_external_gate/results_ghrr_bundle.json). The top-1 numbers in that file are real; the interpretation they supported was wrong.

## Caveats and limits

- **n=3 MQAR drill-down, not n=10 Phase 5 verified evidence.** Per [phase-5-checklist.md:10-16](../notes/emergent-codebook/phase-5-checklist.md), verified status requires n_seeds ≥ 10 against the ΔE headline. This report is diagnostic, not graduation evidence.
- **GHRR-native uses random unitary matrices, not a derived-from-FHRR substrate.** A different GHRR formulation (e.g., real-valued block-circulant binding, or a learned GHRR codebook) might rescue. The single-role-pair MQAR test is null only for the formulation tested.
- **No multi-role binding tested.** The discriminator is single (k, v) pairs. If the project's Phase 5 goal needs structured multi-role binding (e.g., `bind(r1, f1) ⊕ bind(r2, f2)`), multi-role MQAR is the natural next test. Bundle-first might or might not scale; GHRR might or might not have a multi-role rescue that the single-role test missed.
- **Bundle-first as Phase-5' architecture is a design proposal, not an implementation.** Cashing it would require a new Phase 5' design spec — what role the schema store plays when storage is a single bundle, how the value codebook is structured under realistic word/context distributions, whether multi-role chains decompose cleanly. The Report 066 result says "this primitive works at MQAR scale," not "this is a complete Phase 5 architecture."
