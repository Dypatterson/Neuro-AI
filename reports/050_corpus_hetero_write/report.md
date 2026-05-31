# Real-corpus validation of the heteroassociative write — qualified, then resolved

> **Status:** decision-relevant. **Part 1** (the heteroassociative write alone) does NOT
> transfer to the rich-context masked-token path — store-as-is already wins; this
> qualified the verdict. **Part 2** (added 2026-05-30): with the **cue-space
> decorrelator** (whitening, validated in Report 049 §4), the surgical mechanism **DOES
> transfer** — it recovers the real **sparse-cue** regime where store-as-is fails. The
> decorrelation lever's real-data payoff is **confirmed**. Harness:
> `experiments/50_corpus_hetero_write.py`. Design: `phase-4-heteroassociative-write-design.md`.

## Part 2 — decorrelation transfers to the real sparse-cue regime (the resolution)

Part 1's write failed in the *rich-context* regime because store-as-is already wins
there. The faithful analog of the role-binding null is the **sparse-cue** regime
(few context positions), where store-as-is fails. Re-running with `--observed 1` (one
context token) + the whitening decorrelator (`--whiten-keys` analog, the validated
upper-bound lever), D=256→512, 2 seeds:

| cue | store-as-is | raw write | **write + decorrelation** | floor |
|---|---|---|---|---|
| full context (observed=5) | 1.000 | 0.081 | 1.000 | 0.045 |
| sparse (observed=1), N/D=0.9 | **0.024** | 0.058 | **0.590** | 0.047 |

Sparse-cue N-sweep (D=512, observed=1, key cosine ~0.48 — real keys are highly correlated):

| N/D | store-as-is | raw write | **write + decorrelation** | floor |
|---|---|---|---|---|
| 0.45 | 0.211 | 0.087 | **0.678** | 0.046 |
| 0.89 | 0.024 | 0.058 | **0.590** | 0.047 |
| 1.78 | 0.010 | 0.049 | **0.519** | 0.044 |
| 3.59 | 0.005 | 0.048 | **0.242** | 0.048 |

**In the real sparse-cue regime where store-as-is AND the raw write both fail (≈floor),
decorrelation + the write recovers 0.24–0.68** — up to 48× store-as-is and 12× the
floor — degrading gracefully with N/D (the rank reach, exactly as the synthetic Report
049 §4 predicted). **The crux open question is answered: the real role-binding-analog
bottleneck is ill-conditioned and decorrelation-helpable, NOT rank-bound/ill-posed.**
The surgical direction (heteroassociative write + cue-space decorrelation) **transfers
to real data** where the project's current store-as-is fails.

---

## Part 1 — the heteroassociative write alone does not transfer (the original qualification)

## Experiment preamble

- **Active phase:** Phase 3 consolidation-write sub-program (Stage-1, pre-Phase-5′).
- **Headline per [phase-3-consolidation-write-design.md §Headline]:** masked-token Recall@1 via `top_index_hits`, heteroassociative write **vs store-as-is on the same windows**.
- **Required controls:** random-codebook (→0), shuffled-key (→floor), N/D capacity sweep.
- **Last verified result:** Report 049 — synthetic key-only write rescues store-as-is at high N/D (uncorrelated keys).
- **Why now:** confirm the rescue survives integration on real data (AH-required headline experiment; the AH pass did not license a graduation claim).

## Setup

Masked-token contextual completion on `repo_sample` (offline): key = the masked
context-window encoding, value = the masked target token. The write runs over a
**closed, seed-fixed buffer** (AH condition 3). Store-as-is = the project's current path
(full-window MHN → retrieve from masked cue → unbind at the masked position → cleanup).
D=256, 512-token decode set (chance 0.0005), 2 seeds. The dense `H` was validated
(Report 049) and the module is unit-tested (`tests/test_hetero_write.py`, 7/7).

## Result — N-sweep (D=256)

| N/D | store-as-is | hetero-delta | hetero-contrastive | shuffled-key floor |
|---|---|---|---|---|
| 0.90 | 0.996 | 0.719 | 0.691 | 0.019 |
| 1.80 | 0.998 | 0.203 | 0.209 | 0.032 |
| 3.59 | 0.994 | 0.051 | 0.050 | 0.049 |
| 7.21 | 0.993 | 0.057 | 0.057 | 0.057 |
| 14.4 | 0.993 | 0.058 | 0.058 | 0.058 |

Controls clean: random-codebook = 0.000 (readout sound); shuffled-key tracks the
frequency floor (the masked target is often a frequent token, so the effective floor is
~0.02–0.06, above 1/512).

## Reading — the rescue does NOT transfer, and why

1. **Store-as-is does not decay on real data** — it holds ~0.99 even at N/D=14, the
   opposite of the synthetic key-only toy. Reason: the masked-token cue is a **rich
   context** (5 of 6 positions), and real-text windows are **distinctive**, so the
   full-window MHN retrieves the right window reliably and the unbind recovers the
   target. This is the easy/rich-cue regime where store-as-is already works (consistent
   with exp 48). The substrate clearly **can** hold these associations → **not a rebuild
   situation.**
2. **The heteroassociative write collapses to the floor** as N grows. Reason: real
   context keys are **highly correlated** (windows share frequent tokens), so the key
   matrix is rank-deficient and the delta-rule cannot fit it — exactly the correlated-key
   capacity collapse of Report 049 §Result 3. The synthetic rescue relied on
   **near-orthogonal random keys**, which real text does not provide.

## What this changes (the honest walk-back)

- **The fork verdict stands at: NOT rebuild.** Store-as-is recovers real associations at
  ~0.99 — the substrate holds them. G-A (no readout defect) and G-0/G-C (algebra is not
  the wall) are unaffected.
- **But the surgical *mechanism* (dense heteroassociative `H`) is NOT a real-data win.**
  On the rich-context masked-token task store-as-is already wins; under real key
  correlation the write collapses. The Report 049 "write rescues" result is
  **regime-specific** (uncorrelated keys, hard N/D) and must not be cited as a real-data
  graduation result.
- **The real lever is DECORRELATION, not heteroassociation per se.** The write failed
  precisely where the keys correlate. This **redirects the surgical direction toward an
  orthogonalizing/decorrelating consolidation objective** — which is what the
  diagnosis's *other* ranked candidates provide: **FEP self-orthogonalizing**
  (`arxiv:2505.22749`, the anti-Hebbian term is exactly a decorrelator) and **Dorrell
  rectangular-support / nonneg modularity** (`arxiv:2410.06232`). The MESH fixed-scaffold
  form also decorrelates via its scaffold. The contrastive/heteroassociative candidate
  drops from front-runner.

## Open question for the next gate

The faithful analog of the Phase-5 role-binding null (`hit_role=0.000`) is the
**single-role-cue** (key-only) case on **real correlated** atoms — untested here (this
used rich context). The correlated-key result (049 §3) predicts **both** store-as-is and
a plain write struggle there, so that is where a **decorrelating** write (FEP / Dorrell)
should be tested next: does orthogonalizing the codebook restore single-cue role
recovery where plain heteroassociation cannot?

## Next

- Build a **decorrelating consolidation write** (FEP single-phase self-orthogonalizing,
  `arxiv:2505.22749`; card it from primary first) and re-run both the synthetic
  correlated-key gate (049 §3 regime) and the real single-role-cue case.
- Card `arxiv:2505.22749` + `arxiv:2410.06232` from primary before they become
  load-bearing for the redirected direction.
