# Report 057 — Surgical mechanism folded into the real consolidation path (dense H, batch-offline, default-off)

> **Status: INTEGRATED.** The validated heteroassociative write + L2 cue-space decorrelator
> (Reports 055 graduation / 056 G-D) is folded into the production consolidation-write
> orchestrator `OnlineCodebookUpdater` (`src/energy_memory/phase34/online_codebook.py`) as a
> **separate batch-offline pass behind a default-off flag**. The streaming pull/push path is
> **byte-identical** when off (117 insertions / 0 deletions; byte-identity test green).
> Anti-homunculus reviewer **PASS** (all 7 points, all 3 conditions). This is where the
> mechanism stops being a harness result and becomes part of the substrate. **Dense H; MESH
> deferred** (open user scaling decision, noted in 3 durable places).

## What this closes

The Phase-4 heteroassociative-write design's [§Integration plan](../../notes/emergent-codebook/phase-4-heteroassociative-write-design.md)
step 2 ("wire into the masked-token contextual-completion path as a separate batch-offline
consolidation pass"). Steps 1 (the module) and 3 (corpus validation) were already done
(Reports 049/050/055/056); step 2 was the open item.

## Where it landed (and why there)

A 5-agent mapping workflow located the real consolidation-write orchestrator:
**`OnlineCodebookUpdater`** (the "Path C" updater with `use_pull_push` / `use_context_residual`
flags + a byte-identity reproducibility test) — **not** the Benna-Fusi atom-consolidation
(`consolidation.py`, a different abstraction: per-atom u-chains, not cue→target associations)
and **not** raw `ErrorDrivenLearner` (lower-level). `OnlineCodebookUpdater` already has the
0.0-default-flag pattern and the `test_consolidation_path_c_byte_identity` guardrail, so it is
the natural, lowest-risk home.

## The integration (additive, default-off)

| piece | what it does |
|---|---|
| `observe(..., cue=None)` | new optional **raw masked-cue** param; when `hetero_write_enabled` and a cue is passed, appends `(cue, target_id)` to a CLOSED hetero buffer (ALL observations, never quality-gated). `cue=None` / flag-off → inert. |
| `consolidate_hetero()` | a **separate batch-offline pass**: fit L2 `CueDecorrelator` on the closed cue buffer → freeze a `HeteroConsolidationBuffer` → write **dense `H`** (pull-only delta-rule by default). Never fires inside the per-K streaming `_consolidate()`. |
| `recall_hetero(cue)` | `cleanup(H·decorr(cue))` over the codebook → `(top_index, entropy, margin)`. The graduated read path; terminates in a `top_index` count. |
| 8 new `__init__` flags | `hetero_write_enabled=False`, `decorrelator_enabled=True`, `hetero_lr=0.5`, `hetero_epochs=20`, `hetero_contrastive=False`, `hetero_lr_push=0.1`, `hetero_neg_seed=0`, `decorrelator_ridge=1e-5`. All reproducibility-preserving. |

**The key is the RAW masked cue, not the post-unbind `slot_query`** — the design-decisive
choice. The graduated advantage (beating store-as-is 8–40× at sparse cues) came precisely from
`H` **bypassing** the scene-MHN-retrieve+unbind that corrupts `slot_query` at sparse cues
(scene-MHN blend-corruption). Writing `H` over a corrupted `slot_query` would inherit that
corruption and forfeit the advantage; over the raw cue, it does not.

## Anti-homunculus (reviewer PASS, all 3 conditions discharged)

- **Batch-offline (condition 3):** `consolidate_hetero()` freezes the buffer before the write;
  `heteroassociative_write` hard-raises if not frozen. Never fired per-K in the streaming loop.
- **Thermostat removed (condition 1):** the new code never invokes the
  `error_driven_learner:103-115` `sims.argmax`-mined buffering; the optional negative is a
  precommitted seed-fixed swap draw, never `sims.argmax`.
- **MESH deferred (condition 2):** dense `H` only; the scaffold form is an open user decision.
- The decorrelator is a batch ZCA statistic (exempt); the read terminates in a `top_index`
  count (never energy / min-over-branches / ΔE — Phase-5' fence); `entropy`/`margin` are
  dead-ended diagnostics, **never** fed back into write-gating (the design guard against
  re-importing the thermostat in a new costume).

## Reproducibility

`hetero_write_enabled=False` (default) → the new `observe()` block is skipped, `consolidate_hetero()`
raises if called, and `_consolidate()` (pull/push + context-residual + C.2.1–C.2.5) is
**textually untouched**. The diff is **117 insertions / 0 deletions**. The byte-identity test
(`test_consolidation_path_c_byte_identity`) and all 102 consolidation-related tests stay green.

## Tests

`tests/test_hetero_consolidation_integration.py` (7/7): flag-off-is-inert (byte-identity guard),
recall-before-consolidate-raises + no-cues-returns-None (AH/contract guards),
write-then-recall-memorizes (recall > 0.8), decorrelator-off-still-memorizes,
random-codebook-control-collapses (readout-leak control), recall-accepts-single-and-batch. Full
consolidation suite (102 tests) green; no regressions.

## DENSE H vs MESH — the open scaling decision (do not lose)

Dense `H` is `D×D` complex (~128 MB at D=4096). The **MESH-scaffold** form (scales with the
number of associations, not `D²`) is the literature-motivated production target for scale but is
**NOT yet validated** and the card warns it "can conflict with emergent-codebook goals if treated
as the final architecture." **Trigger to revisit:** dense `H`'s memory/compute cost becoming a
bottleneck. Noted in **three durable places**: (1) the active design spec §Integration plan
status banner; (2) the `OnlineCodebookUpdater.__init__` docstring; (3) memory
`neuro_ai_mesh_scaling_decision_open`.

## Verdict

The surgical consolidation write is now a **substrate component**, not just a harness result:
opt-in via `hetero_write_enabled`, dense `H`, batch-offline, AH-clean, reproducibility-preserving.
The contextual-completion memory (role-selective recall from sparse cues, Reports 055/056) is
wired into the real consolidation path.

## Remaining

- A real-corpus run *through the integrated `OnlineCodebookUpdater`* (vs. the exp-56 harness) to
  confirm the integration reproduces the 055/056 recall — a thin end-to-end check.
- The MESH scaling decision (deferred, user).
- Phase-4 checklist line item for the integrated heteroassociative-write path.
