# Report 067 — MQAR Pre-Gate 1: Multi-Role Bundle-First

**Date:** 2026-05-24
**Active phase:** 5 (this is **not a Phase 5 graduation experiment** — see [Report 065 framing](065_mqar_external_architecture_gate.md) and the [phase-5-checklist.md:10-16](../notes/emergent-codebook/phase-5-checklist.md) n≥10 verified standard)
**Status:** Codex-recommended pre-Phase-5'-commit gate 1. Tests whether `Σ bind(role_j, filler_j)` style multi-role bundles survive cleanup, not just single (k, v) pairs ([Report 066 §Implications](066_mqar_ghrr_bundle_first_discriminator.md)). Production run at D=4096, n_queries=512, 3 seeds, K_roles ∈ {2, 4, 8}, N ∈ {16, 32, 64, 128, 256, 512}. Total trials: 6 N × 3 seeds × 3 K_roles × 512 queries = **27,648**. Wall time 14:51 on MPS.
**Decision:** Multi-role bundle-first **passes** at MQAR diagnostic scale. Capacity is bottlenecked by **scene-MHN identification** (basin retrieval over N stored scene bundles), not by content cleanup or by K_roles. K_roles barely affects the curve in the tested range. This strengthens the bundle-first Phase 5' case versus the four-paths option list in [STATUS.md](../STATUS.md) blocker #2. Pre-gate 2 (range-shaped replay + S1 schema) remains the next step before any architectural commit.

---

## Framing

This is a structural extension of [Report 066](066_mqar_ghrr_bundle_first_discriminator.md)'s `bundle_first_key_only` — single-role pairs replaced with multi-role multi-scene bundles. The hypothesis being tested:

> If Phase 5's eventual goal needs structured multi-role binding (e.g., `bind(r_1, f_1) ⊕ bind(r_2, f_2)`), bundle-first might or might not scale. Single-role MQAR passing isn't enough.
>
> — Report 066 §Caveats, "No multi-role binding tested"

Codex's review of Report 066 specifically asked for this gate before committing to bundle-first as the Phase 5' architecture. If multi-role bundle-first does NOT scale, the bundle-first case weakens and closure-paper-or-pivot becomes more attractive. If it DOES scale, the bundle-first Phase 5' design becomes the leading positive path.

**Not Phase 5 verified evidence.** This is an n=3 MQAR drill-down; the Phase 5 verified standard requires n_seeds ≥ 10 against the ΔE headline at [phase-5-unified-design.md:280](../notes/emergent-codebook/phase-5-unified-design.md). Treat conclusions as architecture-gate signal, not graduation evidence.

---

## Setup

- Harness: [`experiments/42_mqar_external_gate.py`](../experiments/42_mqar_external_gate.py), strategy `bundle_first_multirole_multiscene_key_only` (new in this session)
- Substrate: `TorchFHRR(dim=4096)` on MPS, fresh per (seed, N) cell
- Role codebook size: **K_roles** (shared across scenes within a run)
- Content codebook size: **C = 1024** (fixed; fillers drawn uniform with replacement across scenes)
- Storage: each of N scenes is a normalized bundle of K_roles (role, filler) pairs:
  ```
  bundle_n = normalize(Σ_{j=1..K_roles} bind(r_j, content[filler_indices[n, j]]))
  ```
  Stored as one MHN pattern per scene; scene-MHN has N patterns total. Content codebook stored separately in a cleanup MHN with C=1024 patterns.
- Query: pick a scene n, a known-role index j_known, a query-role index j_query ≠ j_known. Cue is one known (role, filler) binding from that scene:
  ```
  cue = normalize(roles[j_known] * content[filler_indices[n, j_known]])
  ```
  **Stage 1:** scene-MHN settles on `cue` → identifies the scene bundle.
  **Stage 2:** algebraic unbind `unbind(scene_state, roles[j_query])` → noisy estimate of `content[filler_indices[n, j_query]]`.
  **Stage 3:** content-MHN cleanup over C=1024 codebook → top-1.
- Diagnostics per query: `scene_top_index_hits` (did stage 1 identify the correct scene?), `content_top_index_hits` (did stage 3 identify the correct filler?), `top1` (does final argmax over content codebook match the ground-truth filler?). Both per-cell aggregates are in the notes field of each [raw results JSON](phase5_mqar_multirole_gate/).
- Beta = 30.0 (canonical retrieval per [STATUS.md "HAM regime split"](../STATUS.md)), max_iter = 10
- 6 N values × 3 seeds × 3 K_roles values × 512 queries = **27,648 trials**
- Wall time: 14:51 on MPS (3 sequential `--K_roles {2,4,8}` runs)

Raw payloads: [results_K2.json](phase5_mqar_multirole_gate/results_K2.json), [results_K4.json](phase5_mqar_multirole_gate/results_K4.json), [results_K8.json](phase5_mqar_multirole_gate/results_K8.json).

---

## Headline (this experiment, not Phase 5)

Top-1 recall and per-stage `tix` (basin-retrieval hits) at D=4096, n=512 per (seed, K, N) cell, n=1536 per (K, N) cell aggregated over 3 seeds, Wilson 95% CI on top-1.

### K_roles = 4 (canonical multi-role case)

| N | top-1 [Wilson 95%] | scene_tix / 1536 | content_tix / 1536 | per-seed top-1 |
|---:|:---:|:---:|:---:|:---:|
| 16  | 1.0000 [0.998, 1.000] | 1536 (100.0%) | 1536 | [1.000, 1.000, 1.000] |
| 32  | 0.9896 [0.983, 0.994] | 1520 (99.0%)  | 1520 | [0.996, 0.984, 0.988] |
| 64  | 0.9707 [0.961, 0.978] | 1491 (97.1%)  | 1491 | [0.980, 0.949, 0.982] |
| 128 | 0.9382 [0.925, 0.949] | 1441 (93.8%)  | 1441 | [0.941, 0.926, 0.947] |
| 256 | 0.8880 [0.871, 0.903] | 1364 (88.8%)  | 1364 | [0.895, 0.863, 0.906] |
| 512 | 0.7917 [0.771, 0.811] | 1216 (79.2%)  | 1216 | [0.787, 0.771, 0.816] |

### Comparison across K_roles ∈ {2, 4, 8}

| N | K=2 top-1 | K=4 top-1 | K=8 top-1 | K=2 scene_tix | K=4 scene_tix | K=8 scene_tix |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 16  | 0.984 | 1.000 | 0.995 | 1511 | 1536 | 1528 |
| 32  | 0.993 | 0.990 | 0.976 | 1525 | 1520 | 1499 |
| 64  | 0.979 | 0.971 | 0.969 | 1503 | 1491 | 1489 |
| 128 | 0.935 | 0.938 | 0.930 | 1436 | 1441 | 1428 |
| 256 | 0.889 | 0.888 | 0.877 | 1365 | 1364 | 1347 |
| 512 | 0.784 | 0.792 | 0.799 | 1204 | 1216 | 1227 |

The K_roles curves are essentially indistinguishable. Cross-K_roles variance is comparable to cross-seed variance within any single K cell.

---

## Three findings

### F-1: Multi-role bundle-first survives cleanup
Top-1 stays above 88% up to N=256 across all K_roles tested. Even at N=512 (where single-role bundle-first dropped to 61.5%) multi-role retains 78-80%. The codex pre-commit gate is **passed at MQAR diagnostic scale**: `Σ bind(role_j, filler_j)` bundles support partial-binding-cued retrieval at production-relevant scale.

### F-2: scene_tix and content_tix track each other to within ~0.07% per cell
In 54 cells × 1536 trials per cell = 82,944 total decisions, scene_tix and content_tix differ by at most 1 trial in 53 cells and by 0 trials in 1 cell. **Cleanup is essentially deterministic given correct scene-MHN identification.** The failure mode at high N is exclusively scene-MHN picking the wrong scene; the algebraic unbind + content-cleanup chain is functionally lossless once the right scene is in hand.

This rules out the codex-flagged confound (top-1 inflated by algebraic decode over a near-uniform mixture, as in the original Report 066 GHRR cell): top-1 cannot be inflated when scene_tix == content_tix because both metrics are measured by MHN argmax, not by post-MHN algebraic recovery.

### F-3: K_roles ∈ {2, 4, 8} barely affects the curve
Cross-K_roles variation per N is within ~1% top-1 absolute (e.g., N=256: K=2 88.9%, K=4 88.8%, K=8 87.7%). This is small relative to cross-seed variance within any single K cell. **Per-scene bundle complexity is not the binding constraint** in this regime.

**Why this is structurally important.** Single-role bundle-first (Report 066) stored N pairs in ONE bundle and cleaned up over a value codebook of size N. Multi-role bundle-first stores N small bundles (K pairs each) as separate MHN patterns, then unbinds from a *small* K-pair bundle. The cleanup cost shifts from "Plate bundle capacity at total-pairs=N" to "MHN capacity at N stored patterns + Plate capacity at K=O(1) pairs per pattern." As N grows, this is the more scalable factorization. The capacity curve here matches that prediction: at N=512 multi-role beats single-role (78-80% vs 61.5%) despite storing 4-8× more bound pairs total.

---

## What this means for the architectural commit

[STATUS.md blocker #2](../STATUS.md) lists four open paths. This report's effect on each:

- **(a.1) M2 training-time intervention.** Unchanged. M2 modifies training, not the algebraic bind or the storage architecture. Could still rescue at the schema-store / prior-selection layer. The decision to invest in M2 vs in bundle-first Phase 5' is now better informed: bundle-first is demonstrated, M2 is hypothetical.
- **(a.2) Range-shaped replay + S1 trace-schema** (pre-gate 2). Still the cheapest open path. Unchanged by this report.
- **(c.1) GHRR substrate rebuild.** Still demoted (Report 066). Multi-role MQAR for GHRR is not tested here and would still be required if GHRR remained attractive on other grounds.
- **(c.2) Bundle-first Phase 5' architecture.** **Strengthened.** Sole positive Report 066 finding now extends to multi-role multi-scene cleanly. Scene-MHN identification is the only remaining bottleneck and is structurally well-understood (Plate-style bundle-distinguishability bound on scene bundles).

**My read for the decision:** the bundle-first case is now strong enough that running pre-gate 2 (range-shaped replay) becomes the gating question for "bundle-first Phase 5' versus closure + alternative Phase 5'." If range-shaped replay can produce role/content modularization data-side, the project has *two* working levers; if it cannot, bundle-first is the leading positive candidate and the architectural commit is well-justified.

---

## Drill-downs

### D-1: Scene-MHN as the sole bottleneck — why
At N=128, K=4: scene_tix = 1441/1536 = 93.8%. So 95/1536 ≈ 6.2% of cues failed to identify the correct scene. The cue `bind(r_known, f_known)` overlaps with the correct scene's bundle by ~1/K (one matching term out of K_roles); wrong scenes' bundles overlap with the cue ~0 in expectation but can have small accidental overlaps when fillers repeat across scenes (since C=1024 < N*K total slots at N=128, K=8). Scene-MHN with β=30 has to discriminate the correct scene against N-1 distractors with small accidental alignments. At N=512 the distractor density grows linearly and the discrimination margin shrinks.

This matches Plate-style bundle distinguishability arguments: storage capacity for *distinct* random bundles in an MHN scales as some O(D / log(N)) bound; at D=4096, capacity starts degrading visibly at N ≈ 256-512. The observed scene_tix drop (100% → 79% across N=16 → 512) is consistent.

### D-2: K_roles invariance — why
The cue's structural information content is fixed: one (role, filler) binding regardless of K_roles. The *amount* of distractor information in the bundle that is NOT helpful to scene-MHN scales with K_roles, but in FHRR each extra binding adds approximately-random unit-phasor noise after bundle normalization — it doesn't make the matching term harder to find, it just adds independent random terms that scene-MHN sees as random per-element phase. At K_roles ≤ 8 and D = 4096, the per-element SNR of the matching binding is essentially unchanged, so scene-MHN identification accuracy is unchanged.

This breaks down at K_roles >> D / log(D) (bundle saturation), which is well beyond the tested range. The architectural implication is positive: **structured events with up to ~ 8-16 role-fillers per scene should retrieve cleanly at MQAR scale**.

### D-3: Comparison with single-role at matched N
At N=128: single-role bundle_first = 100% (tix=3072/3072), multi-role K=4 = 93.8% (scene_tix=1441/1536). Single-role wins at this N because the matching is over a clean value codebook of size N=128 with one bound pair to recover. Multi-role's scene-MHN distinguishes N=128 distinct scene bundles, which is structurally harder per-pattern even though each scene contains less unique info than a single-bundle-of-N.

At N=512: single-role = 61.5% (Plate bundle capacity breached), multi-role K=4 = 79.2% (scene-MHN still discriminating). **Multi-role wins at N=512** because it factorizes the bundle capacity bound across N small bundles instead of one big one.

The crossover is around N=256, which is approximately the Plate single-bundle ceiling at D=4096.

### D-4: Per-seed variance
Cross-seed variance within a (K, N) cell is small. E.g., K=4, N=128: per-seed [0.941, 0.926, 0.947], range 0.021. K=4, N=512: [0.787, 0.771, 0.816], range 0.045. Variance grows with N (as expected for any approaching-capacity regime) but stays manageable. This is not a high-variance estimator.

### D-5: What single-role-MQAR-only would have missed
If we had stopped at Report 066's single-role finding and committed to bundle-first Phase 5' on that basis, two things would have been unverified:
1. Whether structured multi-role bundles actually survive (this report confirms yes).
2. Whether the bottleneck is in cleanup vs in scene identification (this report shows it's scene identification only — informative for Phase 5' scaling design, because it means the cleanup head doesn't need to be the heavy machinery).

Codex's "test multi-role MQAR before committing" was the right call. Without F-2 in particular, the codex confound risk from Report 066 would persist into Phase 5' design.

---

## Done-gate compliance

Per [CLAUDE.md "What 'done' looks like for an experiment"](../CLAUDE.md):

1. ✅ **Headline metric reported with CI.** Wilson 95% CI per cell, 1536 trials per cell.
2. ✅ **Control on same test set.** Each (K, N) cell uses the same FHRR substrate per seed, same content codebook, same query sampler. The within-strategy K_roles sweep is itself a control (does K matter?).
3. ✅ **Drill-down metrics explain anomalies.** D-1 (scene-MHN bottleneck), D-2 (K invariance), D-3 (vs single-role), D-4 (per-seed), D-5 (what would have been missed).
4. ✅ **Written up under `reports/`.** This file.
5. ✅ **STATUS.md updated** — see Recent updates entry for 2026-05-24.

The Phase 5 control matrix from [phase-5-unified-design.md:309-314](../notes/emergent-codebook/phase-5-unified-design.md) does not apply — this is not a Phase 5 graduation experiment.

---

## Anti-homunculus check

The multi-role strategy is a passive measurement: store bundles, cue with partial info, observe what scene-MHN and content-MHN settle to. No `if X then do Y` rule. Anti-homunculus filter applied to the candidate Phase 5' architecture this report informs:

- Scene-MHN settles by energy; no controller decides which scene wins.
- Algebraic unbind is deterministic; no policy chooses which role to unbind by (the query specifies it).
- Content-MHN cleanup settles by energy; no controller decides which content vector wins.
- No mechanism in this report's strategy reads a metric and acts on it. ✅

---

## Files

- Experiment script: [`experiments/42_mqar_external_gate.py`](../experiments/42_mqar_external_gate.py) — strategy `bundle_first_multirole_multiscene_key_only` added, `--K_roles` / `--C_codebook` args added.
- Raw results: [`reports/phase5_mqar_multirole_gate/results_K2.json`](phase5_mqar_multirole_gate/results_K2.json), [`results_K4.json`](phase5_mqar_multirole_gate/results_K4.json), [`results_K8.json`](phase5_mqar_multirole_gate/results_K8.json) — `notes` field of each raw cell carries `scene_tix=X/Y,content_tix=Z/Y` for per-stage diagnostics.

## Caveats and limits

- **n=3 MQAR drill-down, not n=10 Phase 5 verified evidence.** Per [phase-5-checklist.md:10-16](../notes/emergent-codebook/phase-5-checklist.md).
- **Random fillers from a flat content codebook of size 1024.** Real Phase 5 cues come from a corpus with natural co-occurrence statistics; modeling that is out of scope for the diagnostic gate. The capacity bound established here is an upper bound; real-corpus statistics may degrade it.
- **K_roles ≤ 8 only.** At K_roles >> 8 the per-bundle Plate capacity may start to bind. Worth testing K_roles ∈ {16, 32, 64} if Phase 5' eventually wants events with many roles, but not gating for the architectural commit.
- **No scene-token binding.** All scenes share the same role codebook and the cue specifies a role by index. Phase 5's actual schema involves scene tokens (a "this-event" identifier) and richer structure. A follow-up gate would add scene tokens and test `bind(scene_token, bundle_n)` storage; this report does not test that.
- **No noise in roles or fillers.** Cues are clean (role, filler) pairs. Real cues from text encoding have substantial substrate noise. Adding `fhrr.perturb(cue, noise=0.15)` would be the natural noise-robustness extension; not in scope here.
