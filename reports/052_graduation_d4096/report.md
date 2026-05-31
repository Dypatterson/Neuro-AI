# Graduation run @ D=4096 on WikiText-2 — DOES NOT GRADUATE (the local rescue did not transfer)

> **Status:** graduation gate — **FAILED**. The surgical mechanism (heteroassociative
> write + cue-space ZCA decorrelator) does **not** clear its selectivity floor at the
> sparse cue it was built to rescue. The strong local result (Reports 050/051: 0.59 at
> the sparse cue on D=512 source code) **did not transfer** to D=4096 WikiText prose.
> The notebook's on-screen "HEADLINE PASS" was a **false positive** (gated against
> store-as-is, not the shuffled-key floor). Verdict adversarially verified (3
> independent perspectives → converged). Artifacts: `results_pooled.json`,
> `graduation_plot.png` (Drive raw: `MyDrive/neuro-ai/results/graduation_d4096_*`).

## Experiment preamble

- **Active phase:** Phase 3 consolidation-write sub-program (graduation gate for the surgical mechanism).
- **Headline per [phase-3-consolidation-write-design.md §Headline]:** Recall@1 via `top_index_hits`, write+decorrelation vs store-as-is, **two-floor rule** (must clear the selectivity floor with disjoint Wilson CIs — beating store-as-is is *not* sufficient).
- **Required controls per [§Required controls]:** shuffled-key floor, random-codebook, store-as-is baseline, cue-richness sweep.
- **Last verified result:** Reports 050/051 — local sparse-cue rescue 0.59 vs floor 0.047 (D=512, repo_sample source code).
- **Why now:** the scaled graduation test (real D=4096, real WikiText prose) of the validated mechanism.

## Setup

D=4096, WikiText-2, window=6, max_vocab=2000, N=1000 (≈725 decodable/seed), 3 seeds (N=2174 pooled), chance=1/2000=0.0005. Cue richness swept by `observed` ∈ {1,2,3,5} (sparse single token → rich). Ran on Colab/T4 via `notebooks/graduation_d4096_colab.ipynb`.

## Result (pooled, Wilson 95% CI)

| observed | store-as-is | write (no decorr) | **write+decorr** | floor (shuffled-key) | clears floor? |
|---|---|---|---|---|---|
| 1 (sparse) | 0.036 [.029,.045] | 0.096 [.084,.109] | **0.098 [.086,.111]** | 0.096 [.084,.109] | **NO** (tie, +0.002) |
| 2 | 0.035 | 0.096 | **0.114 [.101,.128]** | 0.096 | **NO** (CIs overlap ~.008) |
| 3 | 0.062 | 0.096 | **0.150 [.136,.166]** | 0.096 | **yes** (disjoint, +0.027) |
| 5 (rich) | 0.997 [.994,.999] | 0.101 | **0.263 [.245,.282]** | 0.096 | yes — **but store-as-is=0.997 wins** |

## Verdict: DOES NOT GRADUATE

1. **Fails the floor gate at the sparse cue it was built to rescue.** At observed=1, write+decorr (0.098) ties the floor (0.096) — CIs overlap almost entirely. At observed=2, CIs still overlap. The mechanism only clears the floor at observed≥3 (richer cues), and at observed=5 it loses to the no-op store-as-is baseline by ~0.73. **No operating point exists where write+decorr both clears its floor AND beats store-as-is.**
2. **The local result did not transfer.** The validated headline (Reports 050/051: observed=1, D=512, repo_sample, 0.59 vs floor 0.047 ≈ 12.6× disjoint — the faithful Phase-5 role-binding-null analog) collapsed to **1.02× (tie)** at the matching cell here. The 0.59 was **corpus-specific** (near-orthogonal source-code keys, D=512), not a property of the mechanism on natural correlated prose (key cosine ~0.48) at production D.
3. **The notebook "HEADLINE PASS" @ observed=1 was a false positive** — RETRACTED. It compared write+decorr (0.098) to **store-as-is** (0.036), which is itself **sub-floor** at the sparse cue (0.036 < 0.096, disjoint *below*). Beating a baseline that is itself broken clears nothing. The correct comparison (vs the shuffled-key floor) is a tie. This is exactly the failure the two-floor done-gate was written to catch.

## The honest positive (directional-only — a drill-down, not graduation)

The cue-space ZCA decorrelator **is a real, control-passing active ingredient** on natural prose at production scale: at richer cues it beats its own shuffled-key floor with **disjoint** CIs (observed=3: 0.150; observed=5: 0.263) and scales **monotonically** with cue richness (0.098→0.114→0.150→0.263). Crucially, the **raw heteroassociative write sits *at* the floor** (0.0961, byte-identical to the shuffled-key control at observed=1/2/3; 0.101 vs 0.096 at observed=5) — so **decorrelation is the entire signal, not heteroassociation**, exactly as Report 050 predicted ("the real lever is DECORRELATION"). But this signal lives only in the rich-cue regime where store-as-is already wins, so the lever has **no operating point of practical value**. The direction is **demoted, not falsified**: from "real-data transfer validated" to "cue-rich-only, below-incumbent; sparse-cue rescue did not transfer."

## Load-bearing finding

The make-or-break variable is **corpus/representation, not the mechanism's anti-homunculus shape**: source-code (repetitive → near-orthogonal cues, D=512) gave a 12.6× sparse-cue rescue; natural prose (correlated cues, D=4096) gives a tie. The above-floor signal **migrated out** of the sparse-cue regime (the role-binding analog) and **into** the rich-cue regime where it is redundant with store-as-is.

## Open questions / next (one probe, with a hard abandon commitment)

1. **Is store-as-is being sub-floor at sparse cues an eval artifact or a real sharp-wrong-basin?** Ship the `entropy`/`margin` drill-downs (the spec already requires them) to confirm the full-window MHN is confidently retrieving a *wrong* window from a sparse cue, not an eval bug.
2. **Pre-committed probe:** does a *stronger* orthogonalizing write (FEP self-orthogonalizing `arxiv:2505.22749`, or Dorrell) move **only the observed=1 cell** above its floor? **Hard pre-commit: abandon the surgical direction if observed=1 stays at floor** — that is the only cell that matters for the sparse-cue / role-binding claim.
3. **Isolate the killer variable:** a controlled corpus-swap / D-sweep at fixed N/D (key correlation vs dimensionality vs rank reach) to find what collapsed the rescue.

**Do NOT cite Reports 050/051 as a D=4096 real-data graduation.** This run demotes them to corpus-specific.
