---
date: 2026-05-06
project: personal-ai
tags:
  - notes
  - subject/personal-ai
  - project/personal-ai
---

# Phase 1 Validation Baselines

Phase 1 of the personal-ai implementation closed cleanly on 2026-05-06. Four Claude Code build sessions produced the substrate (FHRR via torchhd, D=4096), the encoding module (position_encoder, encode_sequence, decode_position), the test suite (six pytest tests covering round-trip fidelity, quasi-orthogonality, bundling capacity, structured retrieval, encoding API round trip, and computational characteristics), and an auto-generated validation report. All tests pass; the report's decision section read "Proceed to Phase 2 with FHRR at D=4096." This note records the empirical baselines that future phases will reference, plus a methodology finding from Test 5 that affects how all timing measurements should be interpreted going forward.

## Empirical baselines (D=4096 complex FHRR, M5 Pro, MPS)

These are the measured values from `experiments/01_phase1_validation.py` on 2026-05-06. They supersede the May 5 spike baselines for accuracy-critical comparisons (the spike's distribution measurements are still valid; its timing measurements are dispatch-time only — see the methodology section below).

**Test 1 — Round-trip fidelity (1000 random pairs):**
- Mean fidelity: 1.0000000000
- Worst-case fidelity: 1.0000000000
- Algebraically exact at float32. No drift across 1000 trials.

**Test 2 — Quasi-orthogonality (1000 vectors, 999,000 off-diagonal pairs):**
- |mean cosine sim|: 0.000019
- Std: 0.011071
- Theoretical std (1/√(2D) at D=4096 complex): 0.011049
- Relative std error: 0.21%
- mean(|sim|): 0.008836 (matches spike's 0.008822)

The off-diagonal distribution is N(0, 1/√(2D)) as predicted by the central limit theorem. The substrate behaves as ideal random vectors with no detectable anisotropy or hidden structure.

**Test 3 — Bundling capacity (200 trials × 100 noise vectors per K):**

| K | Recovery rate | Signal | Theoretical 1/√K |
|---|---|---|---|
| 2 | 100% | 0.6367 | 0.7071 |
| 5 | 100% | 0.4017 | 0.4472 |
| 10 | 100% | 0.2820 | 0.3162 |
| 20 | 100% | 0.1988 | 0.2236 |
| 30 | 100% | 0.1622 | 0.1826 |
| 50 | 100% | 0.1255 | 0.1414 |

Fitted log-log scaling exponent: −0.505 (theoretical −0.5, within 1%). 100% recovery at all K values tested, including K=50 — better than the K=10 minimum the spec required. The signal-vs-K relationship cleanly tracks the central limit theorem prediction; the substrate's bundle behaves as an ideal sum of independent random unit-modulus vectors.

The 11–12% offset between measured signal and naive 1/√K (e.g., K=10 gives 0.282 vs 0.316) is consistent across K and is an artifact of per-dimension modulus normalization in the bundle wrapper. This is not a bug; it's the expected consequence of normalizing each complex dimension to unit modulus rather than normalizing the whole vector to unit L2.

**Test 4 — Structured retrieval (5-node directed graph, 5 edges, triple-binding):**
- All 5 queries returned correct destination
- SNR range: 14.3× to 39.5× (criterion >10×)

The wide SNR spread is a small-N artifact: with only 4 incorrect candidates per query, the maximum noise sibling has high variance. With codebook-scale candidate pools (Phase 3+), this spread will narrow substantially.

**Test 5 — Computational characteristics (100 timed runs, 10 warmup):**

| Operation | MPS | CPU | MPS speedup | vs Spike (dispatch-time) |
|---|---|---|---|---|
| bind [1000×4096] | 567.7 µs | 834.2 µs | 1.47× | 1.44× |
| bundle [K=50, D=4096] | 230.3 µs | 59.7 µs | 0.26× | sync-dominated |
| cosine_matrix [500×500] | 1.74 ms | 3.60 ms | 2.07× | 0.65× |
| hopfield_retrieval [1×1000] | 1.25 ms | 0.99 ms | 0.79× | N/A |

MPS device confirmed. No operation exceeded the 5× spike-baseline failure threshold (where comparable). See methodology section below for why bundle is flagged "sync-dominated" and why bind/cosine_matrix appear to deviate from the spike.

## Substrate determinism

The K=10 signal of 0.282 has now been measured to three decimal places in three independent runs: the May 5 spike (standalone script), the Session 2 Test 3 (pytest test in `tests/test_encoding.py`), and the Session 4 validation script (`experiments/01_phase1_validation.py`). Three different code paths, three identical numerical results.

This matters for Phase 3 and beyond. When codebook growth dynamics fail — and they will, because Phase 3 is the gate where most architectural risk lives — the failure can be read as architectural rather than substrate-level. The substrate is pinned down to numerical reproducibility; if Phase 3's masked-token retrieval performance is poor, it is not because the bundle is wrong or the bind is drifting. It is because some architectural choice in the codebook growth machinery is wrong. Eliminating the substrate as a candidate failure source is exactly what Phase 1's purpose was, and it has been accomplished.

## MPS timing methodology

Test 5 surfaced a methodology finding worth stating clearly because it affects every timing comparison going forward.

The May 5 spike's timing baselines (bind: 394.8 µs, bundle: 26.5 µs, cosine_matrix: 2.7 ms) were measured without `torch.mps.synchronize()` between the operation and the timer. On MPS, GPU operations dispatch asynchronously: when the Python call returns, the GPU may not have completed the work. Measuring time from before-dispatch to after-dispatch captures dispatch latency, not completion time.

Phase 1's `tests/timing_utils.py` `measure()` helper synchronizes before and after every timed call. This captures completion time — when the GPU has actually finished the work and the result is available. The synchronize call adds a fixed ~150–200 µs of overhead per measured operation. This overhead is roughly constant across operations because it is dominated by the round-trip cost of the synchronization barrier itself, not by the size of the operation being measured.

Subtracting the ~170 µs offset from each Phase 1 measurement gives times consistent with the spike:

- bind: 567.7 − 170 ≈ 398 µs (spike: 394.8 µs)
- bundle: 230.3 − 170 ≈ 60 µs (spike: 26.5 µs — still some additional overhead, but in the right neighborhood)
- cosine_matrix: 1740 − 170 ≈ 1570 µs (faster than spike's 2700 µs, plausibly due to PyTorch/MPS version improvements)

Bundle is the operation where the methodology mismatch is most visible because its true compute time (~28 µs) is much smaller than the sync overhead (~170 µs). The 8× spike-ratio for bundle does not indicate a regression; it indicates that the spike measured ~28 µs of dispatch and Phase 1 measured ~28 µs of compute plus ~170 µs of sync, comparing two genuinely different things.

Test 5 handles this by skipping the bundle spike-ratio assertion, sanity-checking only that bundle runs in under 5 ms. The bind and cosine_matrix assertions pass cleanly because their compute times are large enough that 170 µs of overhead is a small fraction.

**Recommendation for Phase 3+ if bundle/small-op timing accuracy starts to matter:**
- Time the operation in a loop without per-iteration sync, then sync once at the end and divide. This amortizes the sync overhead across many iterations and recovers per-call dispatch+compute time.
- Or use a kernel-level micro-benchmark that bypasses the Python dispatch overhead entirely.
- Or accept that completion-time *is* the right metric for real workflows (because downstream operations can't use the result until it's actually computed) and stop comparing to the spike's dispatch-time baselines.

The current `measure()` helper is correct for what it measures (completion time). The issue is only that the spike measured something different.

## Hopfield retrieval test sizing

Test 5's `hopfield_retrieval` operation tests "1 query bundle vs 1000 stored bundles, top-1 by cosine." At this size, MPS is *slower* than CPU (1.25 ms vs 0.99 ms), because the sync overhead and dispatch latency dominate the small computation. This does not mean MPS is bad for retrieval; it means the test is sized too small to demonstrate MPS's advantage.

Real retrieval workflows at Phase 3 and beyond will use codebooks of thousands to tens of thousands of stored patterns. At those sizes, the cosine_matrix operation scales much faster than the fixed sync overhead, and MPS will dominate. This test's sizing should scale up when timing accuracy starts mattering — likely Phase 3 or Phase 6.

## Implications for downstream phases

**Phase 2 (dual-objective baseline on static codebook):** the substrate is verified clean. If masked-token retrieval substantially outperforms next-token retrieval as expected from the 2026-05-04 contextual-completion commitment, that result reflects the architecture's fit to associative-memory dynamics, not noise from the substrate. If both objectives perform poorly, the substrate is not the cause; the Hopfield setup or codebook initialization is.

**Phase 3 (codebook growth dynamics):** the error signal in failed retrievals is genuinely informative because the substrate produces algebraically exact bind/unbind and CLT-clean bundling. Error-driven consolidation pathways operate against a clean baseline; spurious signal from substrate noise is not a candidate failure mode. The 1/√K signal scaling means context bags should stay small (K≤10) by design even though the substrate supports K=50 at 100% recovery — signal margin is what matters, not just discrimination above noise.

**Phase 6 (replay store, integration):** the worktree-based session isolation pattern from Claude Code (each session in `.claude/worktrees/<branch>`) is now standard; fast-forward merge to main is the review-cycle norm.

## References

- Generated validation report: `experiments/01_phase1_validation.md` (in the project repo, regenerable by running `uv run python experiments/01_phase1_validation.py`)
- Session digest: `journal/2026-05/2026-05-06-personal-ai-phase-1-closure.md`
- Substrate validation spike (precursor): `projects/personal-ai/notes/2026-05-05-substrate-validation-spike.md`
- Phase 1 spec (amended 2026-05-06): `PHASE_1_SPEC.md` in the project repo
- Decisions: `brain/decisions/2026-05.md` (entries 2026-05-05 and 2026-05-06)
- Project repo: `github.com/Dypatterson/personal-ai` (private), tip at `fe97c19`
