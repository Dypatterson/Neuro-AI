# FROZEN PRE-COMMIT — Phase-3 capacity sweep (adaptive-assignment-under-a-fixed-budget)

*Frozen 2026-06-02 BEFORE running. Branch `experiment/tem-local-reachability-oracle`.
Harness: extend `experiments/72_online_local_on_build_s.py` (reuses exp70's `online_local_kwta`,
exp65/68 helpers) — the ONLY new code is a sweep loop over capacity k + the non-monotonicity test.
Grounds: the 2026-06-02 capacity-compression grounding (verdict: capacity is
**necessary-not-sufficient — the load-bearing step is ADAPTIVE ASSIGNMENT under the budget**),
Report 127 §0.5 (online-local-vs-offline-global, pre-registered NEXT, never run), the SQHN card
(the count-averaged local write = online VQ), and the user's "fixed capacity is the forcing function
for abstraction" hypothesis (refined here).*

## 0. Preamble (CLAUDE.md experiment-preamble)

> **Active capability:** Codebook-growth (P3 structure), substrate-free oracle (IN-SCOPE — a
> single-layer online-local writer on build_S; NOT a multi-layer/FHRR build, which stays the user's fence).
> **Headline metric:** within-paradigmatic-SET **label-shuffle B-KILL** of the adaptive online-local
> writer, **CONJOINED** with `spec_ci_lo > 0` AND `recover_frac_of_E ≥ 0.10` (the degenerate-code
> guard — see §1), as a function of the **capacity k** (the sweep is the money-plot x-axis).
> **Required controls:** the 5 lockstep arms (§3); the +0.1092/0.222 calibration anchor every run
> (INVALID otherwise); the converged-writer calibration (decay=1.0 must reproduce the offline ceiling
> — confirms the writer isn't broken); the flat-SPPMI grow_G floor (~+0.0002 apparatus check).
> **Last verified:** Report 129 (frozen-random slots NULL at fixed k; global ASSIGNMENT load-bearing);
> Report 127 (fixed-k partition = global k-means; online-locality the open differentiator); exp72
> (k=32: adaptive 0.179 vs frozen 0.059).
> **Why now:** the capacity grounding settled that the user's capacity lever is real but
> necessary-not-sufficient; the cheap decisive test is whether the paradigmatic signal PEAKS in the
> under-capacity regime under an *adaptive* local writer. Converges with 127's NEXT + SQHN.

## 1. The hypothesis + the reconciliation it rests on

**Reconciliation of 129 (raw-array-confirmed):** at a fixed budget k, the 129 writers differ ONLY in
assignment — global-assignment `E_nmf` passes B-KILL 10/10 (recover 80-100%); frozen-random
`tem_frozen` recovers 0.00; local per-token `tem_frozen_comp` recovers 15-28% as hubness (B-KILL
0-4/10). A *tighter* budget made the local writer WORSE (recover 0.152→0.209→0.282 as k 8→16→32) →
**compression-pressure ALONE does not force useful structure.** Capacity is the arena; the
load-bearing step is a local rule that ADAPTS the assignment under the budget (online competitive
learning / VQ = the SQHN count-averaged write = 127's never-run NEXT).

**Hypothesis under test:** an ONLINE-LOCAL adaptive-assignment writer, swept over capacity k, shows
paradigmatic B-KILL that is **NON-MONOTONE** — rising into an under-capacity regime, **peaking at
k\* ≪ V**, falling toward the floor as capacity grows ample — AND beating frozen-random (A−B) and
offline-global-k-means (A−C) at k\*. That would confirm: *adaptive assignment under a binding
capacity bottleneck is the local route to the paradigmatic structure.*

**DEGENERATE-CODE GUARD (new, from the 129 raw array):** `tem_frozen` scored B-KILL 8-10/10 on a
`spec=0.000 / recover=0.000` zero-code (noise tips the shuffle-diff positive). So **B-KILL seed-count
ALONE is vacuous on near-zero codes.** The headline B-KILL is therefore CONJOINED with
`spec_ci_lo > 0` AND `recover_frac_of_E ≥ 0.10`. Tiny-k cells (k=2,4) are most prone — a peak there
is suspect until the conjunction holds.

## 2. The local rule (anti-homunculus-safe, primary-grounded)

`exp70.online_local_kwta()` VERBATIM. Per streamed chunk: recency-bounded co-occurrence
`Π = decay·Π + C_chunk` (decay=0.7 ⇒ Π NEVER equals the global operator = genuinely capacity/recency-
bounded LOCAL); form current build_S row-profiles `L`; ONE competitive step `a=relu(L Wᵀ)`, keep each
token's top-`cap` units by a FIXED uniform rank-threshold, count-averaged Hebbian move of WON units
`W = l2rows(W + 0.5·(won/cnt − W))`. This IS the online-VQ / SQHN one-hot CAPTURE step (SQHN card
Eq.4 count-averaged single-column write) — the **kwta capture ONLY, NOT Eq.4 threaded through a
downward weight** (the SQHN card's anti-backprop guard). No backprop, no global error, no metric-gated
branch. Read = static slot-cosine (`exp68.read_specificity`), NOT grow_G.

**Fixed capacity, imposed identically across the whole grid AND across the para/collo/rand probes:**
(i) codebook width k; (ii) k-WTA `cap = max(2, k//4)`; (iii) `decay=0.7`. A fixed problem-generic
SCAFFOLD = CONTEXT.md §3 ("the precommitted k-WTA cap is LEGAL"). NO neurogenesis (it grows capacity
toward the memorization floor — the wrong pressure, per the SQHN card Dury caveat). `k*` is read
**OFFLINE/post-hoc** from the curve (a batch statistic, exempt); the writer NEVER sees B-KILL or a
para-label at runtime.

## 3. The five lockstep arms (same seed at each k) — the controls that isolate the lever

- **A — ADAPTIVE online-local** (`online_local_kwta`, learn=True). The candidate.
- **B — FROZEN-RANDOM online-local** (learn=False). **A−B isolates adaptive-assignment** (the 129 separation).
- **C — OFFLINE-GLOBAL k-means one-hot** at matched k. **A−C isolates online-locality** (the Report-127 differentiator over Lloyd k-means).
- **D — OFFLINE-GLOBAL k-WTA** at matched k. Partition ceiling.
- **E — GLOBAL-NMF** at matched k (`exp65.nmf_slots`). The Oracle-E flashlight ceiling + the `recover_frac_of_E` denominator + the **CEILING-GUARD** (arm A must NOT exceed it — exceeding = inflation artifact).
Plus: **converged-calibration** (decay=1.0 must reproduce the offline ceiling, exp72 ratio≈1.02) + the grow_G linear floor.

Report **DIFFERENTIALLY** only (A−B, A−C, label-shuffle); NEVER absolute cosine (inflation caught 3×: R125/127/129).

## 4. The sweep + PASS/NULL sub-cases (frozen)

Operator = **build_S** primary, V≈2002. Sweep **k ∈ {2,4,8,16,32,64,128,256,512,1024}** (compression
ratio ~1000:1 → ~2:1). n=10 seeds, across-seed bootstrap CI per cell. Anchor every run.

- **PASS-COMPRESSION:** arm A's headline curve is NON-MONOTONE — peaks at **k\* ≪ V**, strictly
  beating both the smallest-k and the largest-k cells by ≥+0.02 (across-seed CI-lo>0); AND `(A−B)`
  CI-lo>0 at k\* (adaptive-assignment load-bearing = 129 reconciliation confirmed); AND `(A−C)`
  CI-lo>0 at k\* (online-locality beats global k-means = a genuine local-route win); AND arm A never
  exceeds arm E (CEILING-GUARD). The headline B-KILL conjunction (§1) holds at k\*.
- **NULL-MONOTONE/FLAT:** B-KILL flat or monotone-increasing in k (no under-capacity peak) ⇒ capacity
  is NOT the lever.
- **NULL-CAPACITY-INERT:** A ≈ B across all k ⇒ the cap doesn't bind / adaptive assignment buys nothing
  ⇒ the 129 frozen-slot null generalizes across the k-range.
- **NULL-EQUALS-KMEANS:** A ≈ C at every k ⇒ online-locality buys nothing over a global partition (the
  127 trap re-fires) ⇒ no local-route win even if the curve is non-monotone.
- **INVALID:** anchor misses +0.109/0.222.

**CHEAPEST DECISIVE FIRST CUT (~half a day):** exp72 already gives k=32. Run **k ∈ {4, 64, 512}** at
n=10, arms A/B/C/E + converged-calibration + anchor → a 4-point curve decides PASS-COMPRESSION vs
NULL-MONOTONE BEFORE committing to the full 10-point grid. Fill {2,8,16,128,256,1024} only if non-flat.

## 5. Honest ceiling + scope (frozen)

A PASS = "a LOCAL, online, capacity-bottlenecked writer reaches the global compressed code — locally,
without backprop — and the under-capacity regime is where the paradigmatic signal is strongest." It is
NEVER "capacity beats SVD/NMF" (the global NMF/k-means ALSO operate under a k-bottleneck and already
reach king/queen; arm A exceeding arm E = artifact). The verdict rests ENTIRELY on the differential
controls. **Bank any NULL ONLY as scoped to "single-layer online-local-VQ under the tested k-range,"
NEVER as "compression is not the lever"** — the depth axis (exp77/SQHN multilayer-deflation) is the
other live escape 129 signposted, and a single-layer null cannot separate "assignment not the lever"
from "single-layer flatness."

## 6. Open risks (carry into the run)

- **Operator confound (false-negative):** on build_S the paradigmatic axis is subdominant, so a perfect
  local writer compresses to the DOMINANT collocational mode. **If build_S nulls flat, re-run the curve
  on M_SR/M_trans (para = slow/dominant) BEFORE banking a null.**
- **Over-capacity fake (false-positive, R126 §5):** a high B-KILL at ample k can be the saturated-memory
  hubness fake; the hypothesis PREDICTS B-KILL falls at ample k, so a high-k B-KILL itself falsifies the
  compression story. Pin a memory-validity gate at every k.
- **Degenerate-zero-code (false-positive):** §1 conjunction guard (B-KILL ∧ spec ∧ recover).
- **Reduces to global k-means (R127 trap):** arm C every k; PASS needs A−C CI-lo>0.
- **Common-mode collapse (R127 §5):** watch rank-1 collapse on M_trans peaky rows; the k-WTA
  rank-threshold avoids soft-NSM collapse but confirm.
- **Literature not load-bearing:** Pehlevan-Chklovskii / IB / Koopman / grokking are link_only —
  MOTIVATING only; the carded primary is the SQHN card; Dury is motivating-only (banned mechanism).

## 7. Build checklist

- [ ] Reuse via importlib: `exp70.{online_local_kwta, kmeans_onehot, kwta_offline}`, `exp65.{nmf_slots,
      faithful_read}`, `exp68.{read_specificity, derange_partners}`, `exp63.{select_pairs,
      raw_sppmi_svd_anchor}`, `exp61.build_S`.
- [ ] New code only: the k-sweep loop, the compression-ratio x-axis, the non-monotonicity / peak-k* test,
      the B-KILL∧spec∧recover conjunction gate, the CEILING-GUARD.
- [ ] Anchor every run; converged-calibration arm; grow_G floor.
- [ ] Planted-corpus smoke before WikiText.
- [ ] Freeze: decay=0.7, cap=k//4, k-grid, n=10, B-KILL≥8/10 ∧ spec_ci_lo>0 ∧ recover≥0.10, the 4
      PASS/NULL sub-cases, build_S primary / M_SR fallback, SimLex≥5 sha.
- [ ] Adversarial 3-lens verification BEFORE banking any positive (the 129 pattern).
