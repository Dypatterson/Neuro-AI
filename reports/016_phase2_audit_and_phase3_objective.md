# Report 016 — Phase 2 matrix audit + Phase 3 codebook-growth objective

**Date:** 2026-05-13
**Source data:** [reports/phase2_full_matrix/02_phase2_retrieval_baseline.csv](phase2_full_matrix/02_phase2_retrieval_baseline.csv)
(504 rows, wikitext-2-raw-v1, vocab=2050, D=4096, MPS)
**Audits PROJECT_PLAN Tier C items:**
- Item 3: *"Finish the Phase 2 static contextual-completion matrix on MPS."*
- Item 4: *"Use Phase 2 results to decide the first Phase 3 codebook-growth objective."*

---

## ⚠️ 2026-05-13 update (post-report 018) — objective retired and replaced

The original section "Recommendation: pin the headline metric, leave
the algorithm open" below proposed `masked_token:generalization
Recall@1` as the Phase 3 primary headline, with a bar of 0.205
(single-seed) to beat.

Two subsequent reports invalidated that proposal:

* [Report 017](017_phase3_codebook_comparison.md) showed the 0.205
  bar was a single-seed point estimate at the upper edge of the
  actual baseline distribution. Multi-seed pooled baseline: **0.089
  [0.060, 0.125]** at the same operating point. Under Recall@1, no
  learned codebook reliably beat random.
* [Report 018](018_phase3_synergy_comparison.md) showed that Recall@1
  is the wrong **shape** of metric for this substrate, not just an
  under-powered one. At n_pooled=2191, all three learned codebooks
  (Hebbian, reconstruction, error_driven) beat the random baseline
  by **~125× on settled synergy at the masked position** with
  crisply disjoint CIs, while Recall@1 either tied random (Hebbian)
  or regressed below it (reconstruction, error_driven). The codebooks
  are doing real structural work; the argmax-Recall@1 projection
  discards it.

**Revised Phase 3 codebook-growth objective:**

> **Primary headline (new):** Maximize **settled synergy at the
> masked position** over the operating envelope's generalization
> windows. Beat the random baseline of **0.002 [0.001, 0.003]** with
> non-overlapping pooled CIs at ≥ 6 seeds, test_samples ≥ 500 per
> seed.
>
> **Drill-down / sanity check:** `masked_token:generalization
> Recall@1` against the multi-seed pooled random baseline of **0.089
> [0.060, 0.125]**. Large Recall@1 regressions vs. random (CI
> disjoint from baseline on the upper side) flag that the learned
> codebook is over-merging atoms in its semantic neighborhood — a
> trade-off characterization, not a failure verdict.
>
> Operating envelope (unchanged from below): β=3, L=64, W=8, mask=1
> at end position, wikitext-2-raw-v1, D=4096, MPS.

**First Phase 3 codebook-growth result (per report 018):** Hebbian
co-occurrence learning. Settled synergy 0.252 [0.249, 0.256], the
clear winner with disjoint CIs vs. all other codebooks; no Recall@1
regression vs. random.

The original Item 3 audit (matrix completeness) below is unchanged
and still valid. The Item 4 objective proposal below is retained for
the reasoning trail but **superseded** by the box above.

---

## Item 3: Matrix completeness audit

### What was run

The Phase 2 full matrix is **already a complete sweep** across:

| dimension          | values                          |
| ------------------ | ------------------------------- |
| dataset            | wikitext-2-raw-v1               |
| substrate D        | 4096                            |
| vocab              | 2050                            |
| window sizes       | 4, 8, 16                        |
| landscape sizes    | 64, 256, 1024                   |
| betas              | 3, 10, 30, 100                  |
| mask positions     | center, edge, end (masked_token)|
| mask counts        | 1, 2                            |
| objectives         | masked_token, next_token        |
| retrieval          | memorization, generalization    |
| test samples       | 64 per cell                     |
| device             | MPS                             |

Total: 504 rows. Memorization = retrieve from windows stored in the
landscape; generalization = retrieve from held-out validation windows.

### What's missing

In its current form: **nothing required for Item 4 is missing.** The
matrix has enough coverage to identify the operating regime for
generalization (the regime Phase 3 needs to improve) and the regime
for memorization (the regime that already works).

Optional gaps worth flagging — not blockers:

1. **Single-seed.** The CSV doesn't include multi-seed Wilson CIs
   across cells; the existing 95% CIs are computed within each cell
   over the 64 test samples. If a Phase 3 result claims to beat
   baseline by a few points, a 3–5 seed re-run of the most relevant
   cells (the best Phase 2 generalization rows) would tighten the
   comparison.
2. **No D-sweep.** Run at D=4096 only. The synergy probe
   ([report 011](011_synergy_probe_phase4.md)) found that
   compositional structure is preserved at the project's intended
   operating point; we don't have evidence Phase 2 should rerun at
   smaller D, but a quick D=2048 spot-check would be the cheap
   safety net.
3. **No multi-scale comparison in this matrix.** The
   `09_multiscale_retrieval.py` and related experiments run
   multi-scale, but they're not in this CSV. That's by design —
   the Phase 2 baseline is the single-scale baseline.

**Recommendation: declare the matrix complete for Item 4 purposes.**
Optional gaps above are follow-ups, not prerequisites.

## Item 3 headline: what the matrix says

### Memorization works

| objective    | W=4    | W=8    | W=16   |
| ------------ | ------ | ------ | ------ |
| masked_token | 1.000  | 1.000  | 1.000  |
| next_token   | 0.981  | 1.000  | 1.000  |

(Best cells; β=10, L=64, mask position center.) The Hopfield substrate
correctly retrieves any window it has stored. The retrieval substrate
is sound; this is the same conclusion every Phase 0/1 result reached.

### Generalization fails on random codebooks

| objective    | W=4    | W=8    | W=16   |
| ------------ | ------ | ------ | ------ |
| masked_token | 0.100  | 0.205  | 0.091  |
| next_token   | 0.100  | 0.205  | 0.091  |

(Best cells per row; β=3, L=64, mask position end / N/A.) On held-out
windows that weren't in the stored landscape, the random codebook
recalls correctly only 9–21% of the time. **This is the Phase 3
problem statement.**

### Three operating-point observations

1. **β=3 wins for generalization.** Lower temperature gives a softer
   landscape that returns *something* relevant rather than committing
   sharply to the wrong attractor. Phase 2's headline number is
   β=3 / L=64. The blend regime is the only regime where novel
   windows get any signal.
2. **L=64 wins for generalization.** Smaller landscapes have fewer
   attractors to interfere with novel queries. Larger L hurts.
3. **W=8 is the sweet spot.** Too-short windows (W=4) don't carry
   enough context; too-long windows (W=16) dilute via more mask
   targets and longer bundles.

These three observations frame the Phase 3 operating envelope:
**β=3, L=64, W=8**, and the bar to beat is **0.205 masked_token
generalization accuracy**.

## Item 4: Phase 3 codebook-growth objective

### Framing

The Phase 2 matrix gives Phase 3 a single concrete number to beat
(0.205) at a single concrete operating point (β=3, L=64, W=8,
masked_token, end-position mask, mask_count=1). Everything else —
which learning algorithm, what loss, what update rule — is
implementation. The decision the project plan asks for is the
*objective*: what is the codebook being grown *toward*?

### Three candidate objectives

**A. Generalization-direct.** Train the codebook to maximize
masked_token:generalization accuracy at the operating envelope. This
is the headline metric the Phase 2 matrix defines. It is the
clearest pass/fail signal: does the codebook get the held-out target
in the top-1 prediction, more often than random? Bar: 0.205. The
risk: this is task-shaped supervision, biases the codebook toward
this specific evaluation rather than substrate-general structure.

**B. Reconstruction-loss.** Train the codebook to minimize the
substrate's own reconstruction error on stored windows: given a
window minus one token, how close to the original is the recovered
state? This is the Phase 3c objective already in
`experiments/04_phase3c_reconstruction.py`. The advantage: it's
self-supervised — no held-out validation set required, just the
substrate's own algebra. The risk: a codebook that perfectly
reconstructs its own training windows can still fail to generalize.

**C. Hebbian co-occurrence.** Adjust codebook atoms based on which
patterns co-fire together during retrieval, no explicit loss. This
is the Phase 3a objective in
`experiments/03_phase3a_hebbian_codebook.py` and was chosen over the
removed `03_phase3b_error_driven.py` (commit f4475f9: "Replace
online error-driven codebook updater with Hebbian pathway"). The
advantage: it has the strongest anti-homunculus story — a co-
occurrence Hebbian rule is the canonical FEP gradient on the
likelihood that frequently-co-occurring atoms should be more
similar. The risk: the relationship between Hebbian similarity
and the headline accuracy metric is indirect.

### Recommendation: pin the headline metric, leave the algorithm open

The Phase 3 codebook-growth objective should be formally:

> **Maximize masked_token:generalization Recall@1 at the operating
> envelope (β=3, L=64, W=8, mask=1 end-position) on wikitext-2-raw-v1
> validation. Beat the Phase 2 random-codebook baseline of 0.205.
> Report Wilson CIs across at least 3 seeds.**

Drill-down metrics (per the 2026-05-09 headline-vs-drill-down rule):

- **bigram accuracy**: is the codebook at least picking up adjacency
  statistics?
- **cap_error_0_5**: does the right answer appear with score ≥ 0.5
  in the top-K?
- **mean entropy**: how committed is the chosen prediction?
- **synergy** (per [report 011](011_synergy_probe_phase4.md)): does
  the codebook produce compositional bindings, or does it collapse
  toward atom-like representations?

This pins *what* Phase 3 is optimizing against, without pinning *how*.
Phase 3a (Hebbian), Phase 3c (reconstruction), and any future variant
can be evaluated against the same number on the same data.

### What this implies for the existing Phase 3 experiments

The repo already has Phase 3a, 3c, and Phase 3/4 integrated
experiments. The natural next step is **to run each against the
operating envelope of the Phase 2 matrix** and report headline
Recall@1 + drill-downs in a uniform format. Whichever existing
approach beats 0.205 first (with CIs that don't overlap 0.205) is
the de facto first Phase 3 codebook-growth winner. None of those
have published numbers at this exact operating envelope yet — the
existing `phase2_full_matrix/` is the random-codebook baseline; the
matching learned-codebook numbers don't exist as a single
comparable table.

The minimum viable Phase 3 closeout artifact is a single CSV with
one row per (learning method, seed) at the Phase 2 envelope.

## Anti-homunculus check / FEP audit

This is a measurement-and-targets report. No mechanisms added or
modified. The headline metric is a deterministic geometric quantity
(unbinding-and-similarity); the drill-downs are also pure
measurements. Passes trivially.

## Recommended next steps

The project plan's standing Tier C is now fully audited.
Three actionable items follow naturally:

1. **(Item 4 follow-through.)** Run each existing Phase 3 codebook-
   growth approach (3a Hebbian, 3c reconstruction, 19/21 integrated
   variants) at the Phase 2 operating envelope and produce the
   comparison CSV above. ~1–2 sessions.
2. **(Item 3 optional gap closure.)** Multi-seed Wilson CIs on the
   Phase 2 baseline cells the comparison will be drawn against (the
   ~12 cells around β=3, L=64, W=8 covering best/best±1 settings).
   ~0.5 session.
3. **(Item 2 follow-through, only when needed.)** Port the remaining
   six reference-path experiments to torch on demand — the pattern
   from [report 015](015_phase0_torch_port.md) is established.

After (1) the project has its first Phase 3 codebook-growth result
under a uniform measurement frame, which closes the standing
PROJECT_PLAN Tier C list and unblocks the Phase 4/5 work the
brainstorm-driven sidebar enabled.
