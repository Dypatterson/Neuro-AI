# Report 058 — End-to-end wiring check: the integrated consolidation path reproduces the graduated mechanism BIT-IDENTICALLY

> **Status: WIRING CONFIRMED.** Driving identical masked-encoding data through the
> integrated `OnlineCodebookUpdater` public API (`observe(cue=)` → `consolidate_hetero()`
> → `recall_hetero()`, Report 057) reproduces the standalone exp-56 harness (the code
> behind Reports 055/056) **bit-identically** — `torch.equal` on both the dense `H` and
> the basin indices, in **all 24 local cells** (D=1024/2048, CPU) **and at the D=4096
> WikiText-2 graduation scale on GPU** (Colab/CUDA), where it lands **exactly** on the
> published 055/056 anchors (`A−report = +0.000` on all three obs cells). Adversarial
> verification (4-agent workflow, 3 independent falsification probes) returns **PASS, high
> confidence, no genuine defect.**
>
> **CLASSIFICATION: DRILL-DOWN / integration-wiring validation — NOT a graduation
> experiment.** The mechanism already graduated (055) and passed G-D (056). The claim under
> test is *"the production fold-in reproduces the validated mechanism"*, not a fresh
> graduation. Measurement-only; no new mechanism; anti-homunculus risk nil.

## Experiment preamble

- **Active phase:** Phase 3 — consolidation-write sub-program.
- **Headline metric per [phase-3-consolidation-write-design.md:89–101](../../notes/emergent-codebook/phase-3-consolidation-write-design.md):**
  in-sample role-Selectivity-Δ of write+L2 = `top_index_hits(true-position cue) −
  top_index_hits(position-deranged cue)` over the value codebook, two-floor Wilson rule
  (true-role arm CI lower > `1/|value codebook|` **and** Δ CI > 0; anti-overlap guard),
  plus the write-marginal anchor.
- **Required controls per [phase-3-consolidation-write-design.md:118–137](../../notes/emergent-codebook/phase-3-consolidation-write-design.md):**
  random-codebook ablation → chance; role-shuffle = per-scene fixed-point-free position
  derangement (not codebook relabel).
- **Last verified result:** Reports [055](../055_graduation_certificate/report.md) /
  [056](../056_gd_selectivity_panel/report.md) — measured through the **standalone exp-56 harness**.
- **Why now:** closes [Report 057](../057_phase4_hetero_integration/report.md) "Remaining" #1
  / STATUS "Next: real-corpus run *through* the integrated updater".

## Design — head-to-head equivalence on byte-identical data

For each `(corpus, D, seed, observed)` cell, the harness
([`experiments/57_e2e_integration_wiring_check.py`](../../experiments/57_e2e_integration_wiring_check.py))
builds the masked-encoding substrate, the value codebook, the per-window **raw masked cue**
`K_true` (the key), the per-scene fixed-point-free **position-deranged** cue `K_der` (the
role-shuffle arm), and the value-codebook-local targets — by **reusing exp-56's own**
`TopicCorpus` / `CorpusWindows` / `encode_cue` / `build_position_vectors` and a verbatim copy
of its `deranged_cue_set` (seed formula `seed*100003 + si*131 + 17`). The same data then runs
through two paths:

| | Path A (reference) | Path B (integrated) |
|---|---|---|
| write | `exp56.write_H` | `OnlineCodebookUpdater.observe(cue=)×N` → `consolidate_hetero()` |
| read | `exp56.write_read` | `recall_hetero(cue)` |
| code | the standalone code behind 055/056 | the production fold-in (Report 057) |

**Why this is the decisive test.** Both paths terminate in the *same* leaf functions
(`phase4.hetero_write.heteroassociative_write` + `recall_top_index` + `CueDecorrelator(l2)`)
with **identical hyper-parameters** (`lr=0.5, epochs=20, ridge=1e-5, β=10, max_iter=12`), so a
**bit-identical** result proves the fold-in introduced **zero computational drift**. The
wrapper still owns the composition glue the leaf functions do not — `observe()`→buffer
accumulation+order, fitting the decorrelator inside `consolidate_hetero()` and re-applying the
*frozen* one at `recall_hetero()`, and the codebook the targets index into. The verifier
injected three wrapper bugs (recall skips the decorrelator; write over raw keys; `vidx`
off-by-one) and **all three flip `torch.equal` to False** — so the equivalence is genuine, not
a tautology.

**Load-bearing mapping.** The integrated path writes targets via `self.codebook[vidx]` and
reads `top_index` over `self.codebook`; exp-56 uses `value_cb`. Path B is therefore constructed
with **`codebook=value_cb`** and `target_id` = the value-codebook-local index, so
`self.codebook[vidx] == value_cb[tgt]` and read-chance `= 1/|value_cb|` match.

In-sample only (`tr==te==range(N)`): the target capability is memorization-recall (CLAUDE.md
contextual-completion, **not** prediction). Held-out is out of scope for a wiring check
(real-text held-out recall ≈ chance *by design* — memory not learner, [[neuro_ai_memory_not_learner]]).

## Results

```
ALL basin indices bit-identical (A==B) across every cell: True     (24/24 cells; H also torch.equal)
```

Pooled in-sample role-Selectivity-Δ through the **integrated** path (Path B):

| cell | role-Δ (B) | B == A | two-floor | report-056 anchor | rand-cb / chance | n_pooled |
|---|---|---|---|---|---|---|
| repo_sample obs1 | 0.366 | ✅ | PASS | 0.352 | 0.0000 / 0.0020 | 1040 |
| repo_sample obs2 | 0.755 | ✅ | PASS | 0.775 | 0.0000 / 0.0020 | 1040 |
| repo_sample obs3 | 0.909 | ✅ | PASS | 0.933 | 0.0000 / 0.0020 | 1040 |
| synthetic obs1 | 0.308 | ✅ | PASS | — | 0.114 / 0.125 | 2500 |
| **synthetic obs2** | **0.592** | ✅ | PASS | **0.592** | 0.112 / 0.125 | 2500 |
| synthetic obs3 | 0.776 | ✅ | (anti-overlap†) | — | 0.114 / 0.125 | 2500 |

- **The wiring witness is `B == A` bit-identical** (`torch.equal` on `H` *and* basin indices),
  exact in every cell — independently of any report anchor.
- **Deterministic anchor:** synthetic obs2 (the closed-form topic-toy, Report-056 5-seed config,
  `n_pooled=2500`) lands on **0.592 = published 0.592**.
- **Controls:** the random-codebook readout-leak control — read through the **integrated**
  `H` + **integrated** decorrelator against a fresh random codebook — collapses to chance
  (repo 0.0000; synthetic scatters around chance at the small `|value_cb|=8`). No readout leak
  survived the fold-in.

## D=4096 WikiText-2 confirmation (Colab / CUDA) — lands EXACTLY on the 055/056 anchors

The graduation-scale convincer, run through the integrated path on GPU
([`notebooks/058_integration_wiring_d4096_colab.ipynb`](../../notebooks/058_integration_wiring_d4096_colab.ipynb),
WikiText-2-raw-v1, D=4096, N=1000, 3 seeds, in-sample):

| obs | role-Δ (B) | == A | true_rate (B) | two-floor | Report-056 anchor | A−report | rand-cb / chance |
|---|---|---|---|---|---|---|---|
| 1 | 0.323 | ✅ | 0.423 | PASS | 0.323 | **+0.000** | 0.0000 / 0.0005 |
| 2 | 0.730 | ✅ | 0.755 | PASS | 0.730 | **+0.000** | 0.0000 / 0.0005 |
| 3 | 0.907 | ✅ | 0.925 | PASS | 0.907 | **+0.000** | 0.0000 / 0.0005 |

`ALL basin indices bit-identical (A == B): True`.

Unlike `repo_sample` (a *live* corpus → small anchor drift), **WikiText-2 is a fixed external
corpus**, so the standalone Path A reproduces the published Report-056 role-Δ **to 3 decimals**
(`A−report = +0.000`), the integrated Path B matches it **bit-identically**, and `true_rate`
lands precisely on the Report-055 information-ceiling recall (0.423/0.755/0.925). This is the
cleanest anchor: the integrated path is behaviorally equivalent to the graduated mechanism **at
the real substrate dimension on real text**, with zero drift.

## Adversarial verification

A 4-agent workflow (3 refutation lenses → synthesis) returned **`wiring_validated: true`,
`confidence: high`, `any_genuine_defect: false`**. Each lens ran live falsification probes:

- **Routing/non-triviality:** instrumented the buffer filling to `N` before consolidate; `H_a`
  and `upd.hetero_H` are **distinct objects with distinct `data_ptr`**; dropping one cue or
  disabling B's decorrelator **breaks** the match → B's `H` is a genuine function of B's own
  accumulated buffer, not a copy of A's.
- **Mapping correctness:** `codebook=value_cb` verified; three injected wrapper bugs all caught.
- **Numbers/controls/AH:** anchor reproduced; repo drift traced to corpus growth (below);
  recall terminates in a `top_index` equality count (never energy / min-over-branches / ΔE — the
  Phase-5′ fence); the write is batch-offline over a **frozen** buffer; entropy/margin are
  dead-ended (never fed back into write-gating).

## Disclosed caveats (from the adversarial synthesis)

1. **This is a wiring validation, not a re-graduation.** It re-proves the integrated path
   reproduces 055/056; it does not independently re-graduate the mechanism.
2. **`repo_sample` anchor drift is corpus growth, not a wiring bug.** `repo_sample` is the
   *live* corpus (`README.md + notes/**/*.md`), which grew since Report 056: the harness pools
   `n=1040` windows vs the Report-056 artifact `corpus_repo_D1024.json`'s `n=993` (same config;
   the unk/mask center-token filter yields a per-snapshot-variable count). obs3 drifts −0.024,
   obs1 +0.014 — the dilution signature of a larger pool. Since **A==B is bit-exact intra-run**
   (both paths consume the same live windows), any drift can only live in the shared upstream
   data, not in Path B. The report anchor is a **cross-snapshot** comparison; the **intra-run
   A==B** is the actual wiring witness.
3. **†synthetic obs3 anti-overlap is the documented strong-selectivity signature, not a
   failure.** The role-shuffled arm dips *significantly sub-chance* (selectivity is strongest at
   obs=3), so the two-floor rule correctly refuses to score it as a pass
   ([Report 056](../056_gd_selectivity_panel/report.md):51-52). **A and B agree on it
   bit-identically** (identical CIs); the true arm (0.873) is the strongest of the three.
4. **Bit-identity is contingent on a coincidence-of-defaults** (`CueDecorrelator.fit` default
   `ridge=1e-5` == the integrated `decorrelator_ridge=1e-5`). This is **self-guarding**: the
   harness's `torch.equal` assertion flips `all_basin_indices_bit_identical` to False if either
   ever drifts, so a future divergence fails the check loudly rather than silently.
5. **Dense `H` only.** The fold-in ships the dense form (~128 MB @ D=4096); the **MESH-scaffold**
   scaling form is **deferred and unvalidated** ([[neuro_ai_mesh_scaling_decision_open]]). This
   check validates the dense path exclusively.
6. **Scale: local + graduation-scale both confirmed.** The local head-to-head ran on CPU at
   D=1024 (repo) / D=2048 (synthetic); the **D=4096 WikiText-2 graduation scale was confirmed
   on GPU** (Colab/CUDA — see the §"D=4096 WikiText-2 confirmation" section above; bit-identical,
   `A−report=+0.000`) via
   [`notebooks/058_integration_wiring_d4096_colab.ipynb`](../../notebooks/058_integration_wiring_d4096_colab.ipynb).
   The equivalence is D-independent (same leaf functions), and this is now demonstrated end-to-end
   from D=1024 through the real D=4096 substrate dimension.

## Verdict

The surgical consolidation write is confirmed wired into the production path: the
`OnlineCodebookUpdater` public API reproduces the graduated 055/056 mechanism **bit-identically**
in every cell, controls intact, anti-homunculus invariants preserved. The integration is not
just present (Report 057) but **behaviorally equivalent to the validated mechanism**.

## Artifacts / reproduce

- Harness: [`experiments/57_e2e_integration_wiring_check.py`](../../experiments/57_e2e_integration_wiring_check.py) (`--corpora repo_sample synthetic` local; `--corpora wikitext --device cuda` for D=4096)
- Results: [`wiring_results.json`](wiring_results.json) · per-cell log: [`wiring_stderr.log`](wiring_stderr.log)
- D=4096 GPU convincer: [`notebooks/058_integration_wiring_d4096_colab.ipynb`](../../notebooks/058_integration_wiring_d4096_colab.ipynb) (generator: [`notebooks/_build_058_integration_wiring_notebook.py`](../../notebooks/_build_058_integration_wiring_notebook.py))

```bash
PYTHONPATH=src .venv/bin/python experiments/57_e2e_integration_wiring_check.py \
    --device cpu --out reports/058_e2e_integration_wiring/wiring_results.json
```
