# Report 148 — Bet B / SCAN-MCD lever 1: composition-as-INFERENCE resolves the CCC confound — the thesis is REDUNDANT-BUT-VALID a 5th time (composition ≈ the holistic baseline)

**Status:** the lever-1 test (RETROSPECTIVE-addendum §4) **RAN → the composition thesis does NOT clear the bar, with the CCC confound removed.** Composition-as-inference manufactures real MCD generalization (beats vanilla, GECA, pooling, random — CI-disjoint) BUT is **matched-or-beaten by a simpler non-compositional conditioning** (the encoder's holistic state): `cca − cca_holistic = −0.026, CI[−0.084, +0.024]`. **REDUNDANT-BUT-VALID, the 5th time** (133/139/144/147-as-null + this), now on the confound-free arena. Per the re-assessment's pre-registered criterion, this **earns the "wall" reading.** (2026-06-17.) **Experiment:** `experiments/95_betb_scan_mcd_cca.py`.

## Preamble (CLAUDE.md)

- **Active capability:** Bet-B Stage-1 mechanism on the GECA-resistant MCD arena (146), resolving the [147](../147_betb_scan_mcd_ccc/report.md) frozen-decoder confound.
- **Headline per [RETROSPECTIVE-addendum-2026-06-17 §4 lever 1](../../notes/RETROSPECTIVE-addendum-2026-06-17-first-discriminating-test.md):** `cca − cca_holistic` CI-disjoint > 0 AND `cca ≥ faithful GECA`, ≥8 seeds — composition (as the inference path) beats the **matched-architecture** non-compositional baseline.
- **Controls:** vanilla_plain floor · cca_holistic (matched arch, holistic conditioning) · cca_random (must NULL) · cca_nosplit (composition-vs-pooling separator) · vanilla_geca (redundancy decider).
- **Why now:** the re-assessment lean — resolve CCC's confound before banking a "wall" reading; if composition-as-inference *also* fails the matched comparison, the wall is earned.

## The mechanism (CCA) — what changed from CCC

CCC ([147](../147_betb_scan_mcd_ccc/report.md)) consolidated a trained model with the **decoder frozen**, forcing `h → e_comp` — so composition was a *regularizer target*, never the inference path, and the frozen decoder preferred the standard `h` (it hurt). **CCA removes both confounds:** the decoder is conditioned (init + per-step input concat) on a representation that is EITHER the composed clause rep `e_comp` (the mechanism) OR the holistic `h` (matched baseline), and is **trained end-to-end** (decoder not frozen). The only difference between `cca` and `cca_holistic` is composed-vs-holistic conditioning → isolates composition. Parser + compose operators = the anti-homunculus-cleared exp94 ones, now in the forward pass.

## Result (mcd1, n=8, 30 epochs)

| arm | mean | CI95 | per-seed spread |
|---|---|---|---|
| vanilla_plain | 0.153 | [0.108, 0.219] | 0.09–0.37 |
| **cca_holistic** (matched, holistic) | **0.300** | [0.233, 0.362] | 0.15–0.45 |
| **cca** (composed-clause) | **0.274** | [0.211, 0.334] | 0.10–0.42 |
| cca_random (must-null) | 0.045 | [0.033, 0.059] | 0.03–0.08 |
| cca_nosplit (separator) | 0.182 | [0.151, 0.211] | 0.11–0.25 |
| vanilla_geca (redundancy) | 0.101 | [0.078, 0.125] | 0.07–0.15 |

**Paired deltas (bootstrap CI):**
- `cca − cca_holistic` = **−0.026, CI[−0.084, +0.024]** — NOT disjoint, slightly **negative**. **Headline FAILS.**
- `cca − cca_nosplit` = +0.092, CI[+0.039, +0.135] — disjoint. Clause structure beats naive whole-pooling.
- `cca − cca_random` = +0.229, CI[+0.170, +0.287] — disjoint. (random split → garbage `e_comp`.)
- `cca − vanilla_geca` = +0.173, CI[+0.106, +0.245] — disjoint (but this is the *architecture*, not composition).
- `cca − vanilla_plain` = +0.121, CI[−0.007, +0.215] — nearly disjoint (again, the conditioning architecture).

## Verdict — REDUNDANT-BUT-VALID (5th); the "wall" reading is earned

The confound is removed and the result **changed** — from CCC's *hurts* (147) to CCA's *helps-but-redundant* — but the thesis bar is **not cleared either way:**

1. **Composition is REAL, not inert** (unlike CCC's `ccc ≈ ccc_nosplit`): the clause-structured conditioning beats naive pooling (+0.092) and random splits (+0.229), CI-disjoint. The clause split carries genuine signal as an inference-path conditioning vector.
2. **But it is REDUNDANT to a simpler non-compositional baseline:** `cca (0.274) ≈ cca_holistic (0.300)`, the paired delta slightly *negative*. **The encoder's own holistic GRU state is at least as good a conditioning signal as the hand-imposed clause composition.** The thesis — "brain-shaped composition manufactures structure the *components* can't" — fails because the component (the holistic encoder summary) already does it.
3. **The big lever is the conditioning ARCHITECTURE, not composition** (both cca and cca_holistic ≈ 0.27–0.30 vs vanilla 0.15, GECA 0.10) — a generic per-step global-context conditioning, **not brain-distinctive** (lands in the general-purpose neural range; cf. T5-base 0.262, still far below the structure-injecting ceiling AuxSeq 0.998 / LeAR 1.0).

So this is **REDUNDANT-BUT-VALID, the program's recurring pattern a 5th time** (133 SGNS≈SVD, 139 BF≈EWC, 144 GECA>consolidation, 147 composition-as-regularizer inert, now 148 composition-as-inference ≈ holistic) — but **now on the confound-free arena with the confound removed.** The re-assessment's pre-registered criterion (RETROSPECTIVE-addendum §4/§5: "if composition-as-inference also nulls with the architecture confound removed, the wall reading is earned") is met: across both the regularizer (147) and inference (148) forms, **brain-shaped clause composition does not beat the matched non-compositional baseline on the discriminating arena.**

## Honest scope

- This is mcd1; the thesis fails the matched comparison here, so mcd2/3 (harder) would only confirm. mcd1 is decisive for the verdict.
- Composition is a *clean* (problem-generic conjunction-split) operationalization; a *richer* grammar-aware composition might do more, but that is **structure-injection** (the bind the re-assessment named — the field's only winners inject structure, which the charter forbids as a mechanism).
- The `cca_holistic`-beats-vanilla effect (a real ~2× lift from per-step global-context conditioning) is a **generic architecture observation, not a thesis claim** — not verified as a contribution (it is a known technique), recorded honestly.

## Disposition

- **Lever 1: DONE → the composition thesis is REDUNDANT-BUT-VALID, confound-free → the "wall" reading is earned.**
- **The honest close (RETROSPECTIVE-addendum §5 option c):** bank the durable contributions (the Bet-A local-vs-global bound; the GECA-resistant arena + the first non-redundant null + this confound-free redundancy; the discipline engine) and write the program up honestly. Levers 2 (structure-injection ceiling-control) and 3 (carry-142-substrate-to-MCD) would *characterize* the gap, not *rescue* the thesis — lower priority, the user's call.

## Done-gate checklist

1. **Headline + CI:** ✅ all 6 arms, 8-seed bootstrap CI + paired deltas.
2. **Control on same test set:** ✅ all arms on identical mcd1 test; the matched `cca_holistic` baseline + random/nosplit cleanliness controls + GECA redundancy decider.
3. **Drill-downs explain the headline:** ✅ composition real (>pooling/random) but redundant (≈holistic); the architecture is the generic lever.
4. **Markdown report:** ✅ this file; raw `experiments/exp95_mcd1_n8.json` alongside.
5. **Status/memory updated:** ✅ STATUS + memory (this session).
