# Frame B grill — code-claim verification (items 9, 11, 5)

**Date:** 2026-05-29
**Status:** verification artifact — NOT a binding spec edit. Grill paused on Q1
(LEVEL→RATE reframe) pending user sign-off. Nothing sign-off-dependent touched.
**Method:** read-only adversarial workflow (verify → independent refute) against
the real repo, cross-checked against the Frame B design spec
(`notes/notes/2026-05-28-frame-b-exposure-slope-headline-design.md`).
**The "c3" file is** `experiments/c3_phase3_exit_criterion.py` (1883 lines) — not
any `scripts/phase34_*` path.

---

## Item 9 — checkpoint axis — ✅ CONFIRMED as a design-spec bug

**Verdict: the grill is CORRECT.** (The workflow's auto-label said "grill-wrong,"
but that was a **category error** — the verifier searched the *code* for the
checkpoint schedule, didn't find it because Frame B is UNBUILT, and concluded the
object was "fabricated." The grill's object is in the **design spec**, not the
code. Item 9 is a spec bug, not a code bug.)

The schedule is real and lives in the spec:
- `frame-b-...-design.md:88-89` — "`M=7` log-spaced checkpoints on the cumulative
  consolidation-**event** axis (`n_consolidation_events=1000`): e ∈ {16,32,64,
  125,250,500,1000}."
- `:258` — repeats the schedule in the sign-off list (item 1).
- `:95` — the load-bearing false claim: "Lower anchor `e=16` sits just after the
  first `consolidation_k`-buffer flush."

The code semantics that break it (all verified exactly against c3, line refs hold):
- `c3:1025` `n_consolidation_events: int = 1000` — **a misnomer.** Plumbed as
  `n_events` (`c3:932`) into `for event_idx in range(n_events)` (`c3:617`), where
  each iteration is **one `observe()` call** (`observations_made += 1`, `c3:645`).
  So it counts **observations/exposure**, not consolidations.
- `consolidation_events` increments only when `consolidate_if_ready()` returns
  non-None (`c3:651-653`); `consolidation_k=100` (`c3:541`); plus one final
  `force_consolidate()` (`c3:657`). → **~10 + 1 = ~11 consolidations** over 1000
  observations.

So the `:95` claim is **false on either reading of the axis**:
- **As consolidation counts** (what the spec NAMES — "consolidation-event axis"):
  only ~11 consolidations ever fire, so e=125/250/500/1000 are **unreachable**.
- **As observe()/exposure counts** (what `n_consolidation_events` actually
  controls, and what ":58 cumulative-exposure checkpoints" implies): the first
  buffer flush is at observe **≥100**, so **e=16/32/64 land BEFORE the first
  consolidation** → pinned to the frozen `e=0` intercept → they flatten/bias the
  OLS slope (`β_w`, spec `:66-68`).

**Fix (spec edit, pre-build — needs user sign-off as it touches the schedule in
the sign-off list):** (a) pin the axis explicitly to `observe()` calls / exposure,
(b) rename the misnamed knob `n_consolidation_events` → `n_observations` (or
`n_exposures`) in the existing code, (c) move the lower anchor to ≥`consolidation_k`
(≥100 exposures) so no fitted point sits on the frozen intercept, (d) correct/delete
the false `:95` sentence. This **strengthens** the grill's "lock the axis before the
floor" dependency: the 0.01/e-fold floor (`:127-129`) is computed over an e-fold
span that the broken axis miscounts.

**Residual (does not change verdict):** the exact flush count (~10 vs ~11) wasn't
line-verified inside `online_codebook.py`'s `consolidate_if_ready` (buffer==k vs
>k); irrelevant to the axis bug.

---

## Item 11 — byte-identity / eval RNG — ⚠️ grill's STRONG claim is FALSE

**Verdict: the grill is WRONG on the blocking claim; a minor hygiene fix remains.**
(Confirmed by both verify and independent refute. This matches my opening-message
finding and **overrides** the grill+audit "guard likely FAILS" assertion.)

- `_evaluate_recall_at_k` **does** draw `substrate.random_vector()` once for a mask
  (`c3:282`, hoisted before its window loop) — grill citation exact. ✓
- **But nothing that runs after an eval and affects the final codebook draws from
  `substrate.generator`:**
  - `online_codebook.py` updater — observe / consolidate / _consolidate(pull-push) /
    context-residual / anti-collapse / cap-coverage / splitting-tension: **zero RNG**
    (grep rc=1).
  - `torch_hopfield.py` `retrieve()` settling loop: **zero RNG**.
  - `phase4/consolidation.py` C.2 dynamics: `torch.zeros` + arithmetic, **zero RNG**.
  - the only `phase3/` RNG is `bimodality_diagnostic.py` (separate generator, and
    **not imported by c3** — c3 imports only regime + theta_prime, `c3:75-79`).
  - the consolidation orchestrator draws its **own** mask once before the loop
    (`c3:587`); no per-iteration generator draw.

→ Inserting read-only checkpoint evals **preserves final-codebook byte-identity.**
The grill's mechanism ("subsequent consolidations draw different numbers") **does
not exist**. ∴ the **anti-homunculus PASS stands firmly** — the audit's conditional
("PASS assumes the byte-identity guard holds") is satisfied, not threatened.

**The existing byte-identity test is ORTHOGONAL.**
`tests/test_consolidation_path_c_byte_identity.py` is a Γ1.c pull/push refactor-
parity test (`torch.allclose(codebook, _expected_pull_push(...), atol=1e-7)`); it
never calls the eval path, never inserts a checkpoint, does not snapshot RNG. An
eval hook neither passes-differently nor fails it.

**The spec's MANDATED guard does not exist yet.** Spec `:168-171` + `:204-205`
require a **new** test 4a: "checkpointed final codebook bit-identical to
non-checkpointed." That regression pin is the right deliverable — not because
byte-identity is broken (it isn't), but to lock it before any future C.2 dynamic
adds a generator draw.

**Real residual (minor):** the eval draws a **fresh** mask per call (`c3:282`), so
M=7 checkpoint evals would each use a different mask → avoidable noise in the
7-point slope fit. **Fix:** pass one fixed mask vector to the hook (+ defensive
RNG snapshot/restore). Pure measurement hygiene.

**Residual on method:** both agents confirmed statically (code-path enumeration),
not via a runtime `get_state()/set_state()` byte-equality test. The spec's test 4a
IS that runtime check — writing it closes the loop.

---

## Item 5 / 14 — Report 115 / done-gate #4 — ✅ grill CORRECT

`reports/` stops at **114** (`114_path_gamma_gamma1_family_closure.md`). No Report
115, no Gate-0/Frame-A/Frame-B numbered report, **zero committed JSON/CSV artifacts**
in the repo. STATUS.md itself (`:8`, `:22-26`) states the Gate-0 (n=10, verdict
weak) result "lives in this STATUS block; no committed JSON artifact." → **done-gate
#4 (written-up report under reports/) is UNMET**, and there is nothing to cite in an
experiment preamble. Corroborates the grill; corrects my earlier optimism.

---

## Net effect on the paused grill

| item | grill said | verified | direction |
|---|---|---|---|
| 9 checkpoint axis | code/design bug | **CONFIRMED (spec bug)** | hardens; workflow's "grill-wrong" label is a category error |
| 11 byte-identity | guard "likely FAILS" | **strong claim FALSE**; hygiene fix only | downgrades; anti-homunculus PASS stands firmly |
| 5/14 no Report 115 | done-gate #4 unmet | **CONFIRMED** | stands |

**Decision-free prerequisites (independent of Q1 and the thresholds):**
1. Write spec test 4a (runtime byte-identity guard) — the binding falsifier.
2. Hygiene: fixed mask vector into the checkpoint eval hook.
3. (pre-build, needs sign-off — touches the schedule) lock the exposure axis +
   rename `n_consolidation_events` + move lower anchor ≥100 + fix the false `:95`
   sentence.
4. A cheap Report 115 + per-seed scatter to satisfy done-gate #4 before the build.

**Q1 (LEVEL→RATE reframe) remains the user's call.** No spec/STATUS edits made.
