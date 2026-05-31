# Report 119 — WS-InfoNCE value-codebook shaping: no held-out composition margin (memory-not-learner confirmed)

**Status:** RESOLVED — terminal finding (not an open blocker).
**Classification:** **DRILL-DOWN / exploratory** (secondary generalization track). **NOT a graduation experiment.**
**Date:** 2026-05-31
**Spec:** [notes/emergent-codebook/phase-3-within-scene-predictive-jepa-design.md](../../notes/emergent-codebook/phase-3-within-scene-predictive-jepa-design.md) (the WS-InfoNCE design; §§2,3,4,5).
**Code:** `src/energy_memory/phase4/ws_infonce.py` (+ `tests/test_ws_infonce.py`, 10/10 pass) · `experiments/58_ws_infonce_codebook_shaping.py`
**Builds on:** Report 056 (G-D memory-not-learner) · Report 067 (F-REPRO discharged) · Report 058 (wiring bit-identical).

---

## Experiment preamble (per CLAUDE.md)

- **Active phase:** Phase 3 — secondary/exploratory **generalization** track. The memorization track graduated/integrated/wiring-confirmed (Reports 055–058) and is the FLOOR, not relitigated.
- **Headline metric** (jepa-design.md:25,33): the **held-out-roles composition-margin** Δ(C′) − Δ(raw C) on the TRUE arm, Newcombe CI, **in the sparse-cue regime where store-as-is fails**. In-sample Selectivity-Δ is a **FLOOR GUARD ONLY** (H already tracks the Bayes ceiling — no headroom). Headline definition cross-ref: consolidation-write-design.md:89-116.
- **Required controls** (jepa-design.md:26-27): random-codebook leak detector; no-negatives ablation; role-pairing-derangement (gauge-safe, per-scene); content-matched non-positional (bag); perfect-cue upper bound; coverage-matched store-as-is; + **mandatory held-out-roles** Δ-vs-frac_seen.
- **HARD FLOOR GUARD** (jepa-design.md:22): write+L2 on C′ must not regress below raw C in-sample at **any** cue richness; in-sample regression = ABORT (read no held-out headline).
- **Last verified result:** Report 056 (real-text held-out ≈ chance = memory-not-learner) · Report 067 (cleanup deterministic given scene-ID; bottleneck is scene-ID, not cleanup).
- **Honest prior** (jepa-design.md:6): on real text, the analytic prediction is a held-out NULL — a **third memory**. This harness is built to *detect* that, not assume it away.

---

## TL;DR — verdict

**WS-InfoNCE Stage-1 FAILS F-COMPOSE and confirms memory-not-learner (F-KILL).** Shaping the value codebook with a within-scene InfoNCE pass produces **no held-out composition-margin lift over plain H at any cue richness** — *significantly negative* at the rich-cue end and null elsewhere — across the full (τ, epochs) range. The in-sample floor guard passes, which licenses the held-out read. The honest prior (a third memory) is confirmed and sharpened: on real text the value-codebook-shaping route **can only subtract** on held-out, because the InfoNCE collapses the codebook (d_eff 341→~165) and degrades the cleanup separability the held-out read depends on, while real text has no generalizable low-rank role→filler structure for the InfoNCE to write (consistent with Report 056).

**DON'T-SCALE-A-NULL honored: no WikiText run** (jepa-design.md:35) — held-out real-text margin is not > 0.

---

## Mechanism (what was built)

`src/energy_memory/phase4/ws_infonce.py` — a batch-offline InfoNCE pass that shapes the FIXED Phase-3 value codebook into `C′`, which the **unchanged** graduated `H` + `CueDecorrelator` (Reports 055/056) then read (`value_codebook=C′`). Stage-1 form (design §2.5):

- **Within-scene self-target** (§2): `S_rest = Σ_{p≠m} bind(positions[p], codebook[tok_p])`; `slot_query = normalize(unbind(S_rest, positions[m]))` — the FHRR conjugate unbind is the only inversion (the Dorrell-killer is absent); `y_target = codebook[t*]` (the true masked filler of THIS scene, a fixed function of data).
- **g_φ = identity**, no predictor MLP, no EMA/stop-grad.
- **Learnable = value-codebook atom phases**, parametrized as `exp(iθ)` so `C′` stays on the FHRR unit-phasor manifold (a valid codebook H reads unchanged).
- **Full-dictionary InfoNCE** (the precommitted candidate/negative set is the value codebook): `loss = CE((slot @ value_cb.conj().T).real / D / τ, t*)`. The no-negatives ablation is pull-only (no denominator).

**Anti-homunculus / fence (verified, tests pass):** single batch-offline pass over a FROZEN scene buffer; self-target is the fixed index `t*`, never `sims.argmax`-mined (the deleted thermostat, `phase2/error_driven_learner.py:106`); negatives precommitted; composition is a sequential data-shaping handoff (no supervisor); the read terminates in a `top_index` basin count (`recall_top_index`), never an energy → min/argmin → ΔE; the InfoNCE owns its own log-softmax (τ ≠ read-time β). `tests/test_ws_infonce.py` asserts the frozen-buffer guard, unit-phasor output, loss-decreases, determinism, and that the module references no fenced energy surface (AST check).

---

## Results

All numbers pooled across 3 seeds (each = a distinct corpus + codebook). Read = `top_index_hits` Selectivity-Δ, two-floor Wilson; margin = Newcombe diff-of-proportions on the TRUE arm.

### Real text — repo_sample, D=1024, N=500, W=6, τ=0.05, epochs=400 (the go/no-go)

| cell | chance | raw C true | C′ true | **margin Δ(C′)−Δ(rawC) true** | Newcombe CI | store-as-is | frac_seen | F-COMPOSE |
|---|---|---|---|---|---|---|---|---|
| **heldout obs1** | 0.0020 | 0.041 | 0.018 | **−0.024** | **[−0.046, −0.003]** | 0.004 | 0.57 | **FAIL** |
| heldout obs2 | 0.0020 | 0.020 | 0.020 | +0.000 | [−0.018, +0.018] | 0.006 | 0.23 | FAIL |
| heldout obs3 | 0.0020 | 0.031 | 0.026 | −0.006 | [−0.028, +0.015] | 0.012 | 0.07 | FAIL |
| insample obs1 | 0.0020 | 0.462 | 0.437 | −0.025 | [−0.068, +0.019] | 0.010 | 1.00 | (floor) |
| insample obs2 | 0.0020 | 0.772 | 0.768 | −0.004 | [−0.040, +0.033] | 0.024 | 1.00 | (floor) |
| insample obs3 | 0.0020 | 0.934 | 0.933 | −0.001 | [−0.023, +0.021] | 0.201 | 1.00 | (floor) |

- **Floor guard PASSES** (all in-sample margins straddle 0; no CI upper bound < 0). H's delta-rule write is refit to the collapsed `C′` targets, so in-sample does not regress → the held-out read is licensed.
- **F-COMPOSE FAIL, 0/6 cells, 0/3 seeds** (`seeds_cprime_beats_raw = 0/3` everywhere). Held-out margin ≤ 0 at every cue richness: **significantly negative at obs1** (CI entirely below 0), null at obs2/obs3.
- **F-KILL: memory-not-learner CONFIRMED.** Both raw C and C′ held-out sit near chance and **decay with frac_seen** (0.57→0.07). C′ never flattens the curve (the learner signature); it tracks or undercuts raw C's decay.

### Drill-down — the negative margin is codebook collapse

The InfoNCE trains to near-zero loss (6.4→0.00) by writing co-occurrence structure, which **collapses the value codebook**: d_eff 341 → 157–195 (held-out 157–168 across seeds), atom drift ~0.33–0.43. Collapsing the cleanup dictionary makes basins overlap; on held-out cues (which H was *not* fit on) this degrades recall — hence the negative obs1 margin. This is the grill-flagged "negative-Δ floor regression" risk (Report 114 precedent; the value codebook is 3-way coupled: regression target + cleanup dictionary + decorrelator covariance source).

### Robustness — (τ, epochs) sensitivity (1 seed, held-out obs1)

| τ | epochs | d_eff (341→) | held-out obs1 margin | CI |
|---|---|---|---|---|
| 0.05 | 100 | 173 | −0.006 | [−0.063, +0.051] *(null — best case)* |
| 0.05 | 400 | 168 | −0.063 | [−0.113, −0.021] |
| 0.20 | 100 | 11 | −0.075 | [−0.124, −0.037] |
| 0.20 | 400 | 9 | −0.069 | [−0.118, −0.029] |
| 0.50 | 100/400 | 6–7 | −0.075 | [−0.124, −0.037] |

**No configuration yields a positive held-out margin.** The best case is a null (gentlest shaping that barely trains); every other setting is significantly negative, scaling with collapse. The verdict is not a tuning artifact: real text offers no generalizable low-rank structure to write, so shaping the value codebook can only subtract held-out separability.

### Controls

- **No-negatives ablation** (pull-only): loss diverges, d_eff collapses further (7.0→6.5 on the toy); in-sample two-floor still passes (H carries it), held-out ≈ chance. Confirms the contrastive push is not the source of any (absent) held-out lift.
- **Random-codebook leak detector:** held-out true ≈ 0.000 (chance) — no readout leak.
- **Content-matched bag / perfect-cue:** behave as in Report 056 (perfect-cue over C′ = 1.0; bag collapses positional selectivity).

### Topic-toy (synthetic) — sanity arm only (uninformative by design)

3 seeds, D=2048, L=8: floor guard passes; held-out margins all straddle 0 (−0.003, −0.013, −0.007). Both raw C and C′ generalize on the low-rank topic structure (Δ +0.26→+0.55) and so does store-as-is (0.34→0.60) — i.e. **store-as-is does not fail here, so a null margin is uninformative** (jepa-design.md:25,33). The informative contrast is precisely that this regime *works* for store-as-is while real text does not, and C′ helps in neither.

---

## Falsifier ledger

| Falsifier | Outcome |
|---|---|
| **F-REPRO** (port dead?) | Discharged by Report 067 (no new run); read-path machinery de-risked. |
| **F-COMPOSE** (InfoNCE adds over plain H on held-out?) | **FAIL** — margin ≤ 0 everywhere; significantly negative at the rich-cue end; robust across (τ, epochs). |
| **F-KILL** (generalizes vs memorizes?) | **Memory-not-learner CONFIRMED** — held-out ≈ chance, decays with frac_seen, not flattened by C′. |
| F-NEG / F-RANDOM / F-CONTENT | Controls behave; no leak; contrastive push is not a hidden lift. |

---

## What this means

The generalization track's only admissible source of a generalization claim was a held-out-roles PASS (after citation repair, zero paper-backed generalization evidence — the JEPA cluster rests on PAM alone, itself memory-not-learner). **That PASS did not occur.** WS-InfoNCE on the value codebook is not merely inert on held-out real text — it is net-negative at the rich-cue end. The honest prior is confirmed: this is a third memory, and the value-codebook-shaping route is a dead end for generalization on real text.

The memorization mechanism (Reports 055–058) is unaffected — it is the FLOOR and remains the graduated, integrated, wiring-confirmed result. WS-InfoNCE composes *on top of* it and is default-off; nothing in the graduated path changed.

### Forward options (NOT pursued here)

1. **Stop the value-codebook-shaping route** (recommended by the evidence): it can only subtract on held-out real text.
2. **Stage-2** (learned g_φ MLP + EMA) is gated behind a Stage-1 basin + the random-codebook falsifier (jepa-design.md:18-19,38). Stage-1 produced no basin worth pursuing, so Stage-2 is **not licensed**.
3. If the generalization question is revisited, Report 067's caveat points elsewhere: the bottleneck is **scene-identification**, not cleanup — a generalization lever would have to act on scene-ID, not on the value/cleanup codebook.

---

## Reproduce

```bash
# go/no-go (real text, 3 seeds)
PYTHONPATH=src .venv/bin/python experiments/58_ws_infonce_codebook_shaping.py \
  --preset repo_sample --epochs-infonce 400 --out reports/119_ws_infonce_codebook_shaping/repo_sample_3seed.json
# topic-toy sanity arm
PYTHONPATH=src .venv/bin/python experiments/58_ws_infonce_codebook_shaping.py \
  --preset synthetic --epochs-infonce 400 --out reports/119_ws_infonce_codebook_shaping/synthetic_3seed.json
# unit tests
PYTHONPATH=src .venv/bin/python -m unittest tests.test_ws_infonce -v
```

Result JSON/logs are gitignored (`reports/**/*.json`, `*.log`); this `report.md` is the durable record.
