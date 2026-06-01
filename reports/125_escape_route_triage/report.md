# Report 125 — Escape-route oracle triage (4 routes) → the bound is route-invariant for LOCAL dynamics; the one live route is a LOCAL writer for the TEM factorization

**Status:** Phase-3. Four substrate-free **DRILL-DOWN feasibility oracles** (NOT graduation),
triaging the four structural escape-routes from the subdominant-modes bound after the flat-code
growth family was exhausted (Reports 121/123/124). Frozen pre-commit:
[phase-3-escape-route-oracle-triage-precommit.md](../../notes/emergent-codebook/phase-3-escape-route-oracle-triage-precommit.md).
Harness: `experiments/65_escape_route_triage.py`. Floor (055–058) untouched. Adversarially
verified (4-agent workflow; disposition + the two guardrails below are its adjudication).

**Verdict (corrected from the harness auto-string — see §4):** **NOT "all four NULL."** The three
**LOCAL `grow_G` reads (B SFA/SR, C eligibility, D order) are clean nulls** — confirming the
local-vs-global bound is **route-invariant for local single-projection dynamics** (now seven
operators across 121/123/124/125). The one **GLOBAL flashlight read (E, TEM nonnegative
slot-factorization) is a real positive** — the structure×content factorization **target** is
confirmed, but its **local writer is untested**. **→ Greenlight GROUNDING the TEM route only**
(NOT a build, NOT a Phase-5 fence lift); the mandatory first deliverable is a **TEM-specific
local-reachability oracle**.

---

## 1. What ran (WikiText-2, V=2002, calibration anchor PERFECT: +0.1092, kq 0.222)

The verdict-bearing read is the faithful `exp61.grow_G` LOCAL dynamic (Report 124 §4 proved the
SVD low-rank spectral read is a contraction artifact), EXCEPT Oracle E which is a STATIC read on a
global NMF (the flashlight; the local Hebbian-path-integration is the *build*, by design).
Gauge-free means-based para-vs-random specificity, n=40 SimLex≥5 non-co-occurring pairs. Gate
(PASS=all): g1 spec CI-lo>+0.04; g2 beats the flat-SPPMI `grow_G` baseline (−0.0003) by +0.02;
g3 corr(log cooc, drift) CI-hi<0.15; g4 d_eff_ratio≥0.5.

| Oracle | read | best spec | corr_hi | gate | result |
|---|---|---|---|---|---|
| **B** SFA/SR slowness (the LEAD) | grow_G local | M_trans +0.0024, M_SR +0.0029 | 0.44/0.43 | g1✗g2✗g3✗ | **clean NULL** |
| **C** eligibility×surprise | grow_G local | β0.5 +0.0007, β1.0 −0.0011 | 0.16/0.28 | g1✗g2✗ | **clean NULL** |
| **D** order channel | grow_G local | +0.0047 | 0.225 | g1✗g2✗ | **clean NULL** (syntagmatic) |
| **E** TEM slots | static global NMF | +0.124/+0.176/+0.188 (k=8/16/32) | 0.25/0.35/0.44 | g1✓g2✓**g3✗** | **flashlight POSITIVE** |

## 2. B/C/D — the bound is route-invariant for LOCAL dynamics

All three local `grow_G` reads reach ≈0 collapse-free specificity (≈ the flat-SPPMI baseline
−0.0003), collocational, failing g1 AND g2 first. Concretely:
- **B (the lead) nulls** — the symmetrized transition (M_trans) and SR-reweighted (M_SR=Σγ^t Tᵗ)
  operators, whose *dominant* modes are the slow/SR low-frequency (substitutability) axis, are NOT
  reached by `grow_G`+H_anti. **This confirms the completeness-critic's "locality trap": power
  iteration + local whitening converges to the dominant collocational mode, not the slow modes.**
  The subdominant→dominant transform is not locally reachable by this dynamic.
- **C** — the multiplicative frequency-novelty gate does NOT beat plain SPPMI (g2 fail); the cheap
  eligibility surrogate dies. (The full two-timescale eligibility×surprise — needing the substrate's
  settling residual as the local surprise signal — is untested; the cheap surrogate is killed.)
- **D** — the position-spectrum order channel is syntagmatic (collocational), exactly as the
  directional-precommit theory predicted; not a Report-009 re-run (009 tested order for recall).

**The local-vs-global bound now reproduces across seven operators** (flat 1st-order 121; flat
2nd-order SPPMI 123; directional 124; + transition / SR / eligibility / order here). For a LOCAL
single-projection growth dynamic over a flat code, the bound holds regardless of the operator.

## 3. E — a real GLOBAL-flashlight positive (the TEM factorization target is real; the local writer is untested)

The NMF factorization of the symmetrized transition operator M_trans into k≈8–32 nonnegative
relational slots clusters king/queen with specificity **+0.124/+0.176/+0.188** — and the
adversarial controls confirm the signal is **real, not a low-dim artifact**: a random-nonnegative
factorization of the same operator yields spec ≈0 in the same cosine regime, and a label-shuffle
collapses spec to ≈0. **E recovers essentially all of M_trans's full-rank paradigmatic
specificity (≈+0.18, dimension-matched SVD) in just 8–32 nonnegative dims.**

**Two honest caveats (verification guardrails):**
1. **The headline absolute cosines are low-dimensionality-INFLATED** — random-pair cosine is +0.47
   to +0.67 (vs the SVD anchor's +0.037) because 8–32 nonnegative slots make everything somewhat
   similar; king/queen cos 0.85–0.90 is partly this baseline. The *specificity* (para−random) nets
   this out (the controls confirm), but these cosines are NOT what the D=4096 substrate would show.
2. **Do NOT claim "E beats the global SVD (+0.19 > +0.109)."** That compares *different operators*
   (the +0.109 anchor is raw SPPMI; E reads M_trans=(T+Tᵀ)/2; a fair SVD-300 on M_trans itself
   gives +0.183). The defensible — and stronger — statement is the "recovers ~all of M_trans's
   full-rank spec in 8–32 dims" above.

**g3 is mis-applied to E's static read (scoped finding).** E's *sole* failed gate is g3. The
project's *accepted* +0.109 SVD anchor — an immovable reference that predates E — **also fails g3
identically**: recomputed corr(log cooc, para-cos) point **+0.077**, CI **[−0.323, +0.440]**,
**CI-hi 0.440** (= E's k=32). At n=40 over cooc∈{0:24, 1:14, 2:2}, `corr_hi<0.15` is unachievable
for *any* static global read regardless of signal — the corr CI is simply too wide. **This does
NOT reopen the 123/124 local nulls:** g3 validly killed those local `grow_G` positives (it operated
on per-pair *drift*, the read-class it was designed and validated for), and B/C/D fail g1/g2 *first*
anyway. The correction is scoped to **static reads only**, and it survives the rationalization-timing
test precisely because it is anchored to a reference the implementer did not choose and cannot move.

**E's epistemic status = exactly what the SVD oracle was for the flat code:** a global flashlight
showing the structure exists, with the *local writer untested*. Greenlighting a TEM substrate/FHRR
build on E alone would repeat the 121→124 error (mistaking a flashlight for a mechanism).

## 4. Why the harness auto-verdict ("all four NULL") is superseded

`experiments/65` mechanically applied g3 to E's static read and printed "NULL — all four routes
fail." The SVD-anchor g3 control (§3) + the adversarial adjudication correct this: E is a real
flashlight positive whose only failed gate is the n=40-underpowered g3. The banked verdict is §0.

## 5. Disposition (adjudicated) — ground the TEM route; build a TEM local-reachability oracle FIRST

**The asymmetry that licenses forward motion (why E is not bound by the B/C/D route-invariance):**
TEM changes the **dynamic** (online Hebbian path-integration with a structural slot layer), not
just the operator — so the *single-projection* route-invariance result does NOT bind it; and E's
NMF target is **nonnegative / part-based / additive** — the representational class a local Hebbian
rule operates in — the first positive evidence in the whole arc that the target is the right
**shape** for a local writer.

**Mandatory next deliverable (the make-or-break, ~exp63 scale, no FHRR port, no Phase-5 commitment):**
a **TEM local-reachability oracle** asking one question — *does a LOCAL online Hebbian
path-integration rule (fixed random structural slots; content vectors Hebbian-bound to the slots
they co-occur in across windows; accumulated ONE WINDOW AT A TIME, with NO global factorization)
recover most of E's +0.19, or null to ≈0 like B/C/D?* The load-bearing distinction from E: E
**factorized** M_trans globally (NMF); the oracle must **accumulate** slot-occupancy locally/online.
- If local path-integration recovers most of +0.19 → the slot architecture breaks the locality trap
  → a TEM build is licensed (then: open Whittington 2020 + the Hebbian-not-backprop locality
  question; FHRR-port Stage-1; the Phase-5 fence — the user's to lift).
- If it nulls like B/C/D → the global NMF optimization was the part that mattered → TEM inherits the
  same trap → kill the TEM build for ~zero cost.
- **Gates: g1 (CI-lo>+0.04), g2 (beat flat-SPPMI by +0.02), g4 (no collapse), PLUS a NEW
  frequency-matched / label-shuffle collocational control — but NOT g3 (dead at n=40).** Expand the
  non-co-occurring SimLex set or replace g3 with a powered discrimination control in the TEM precommit.

**All four NULL contingency did NOT fire** — so the "escalate to a multi-layer hierarchy / substrate
change" disposition is NOT triggered; the live, cheaper question (a local writer for the
factorization) comes first.

## 6. Artifacts
- `experiments/65_escape_route_triage.py`; `reports/125_escape_route_triage/_triage_wikitext.json`
  (+ `.stderr`). New operators: `build_transition_operator`, `build_SR_operator`,
  `build_eligibility_operator`, `build_order_operator`, `nmf_slots`.
- Frozen pre-commit (4 gates); 2-workflow grounding + brainstorm + completeness critic
  ([brainstorm-workspace/2026-06-01-nonflat-phase3/](../../brainstorm-workspace/2026-06-01-nonflat-phase3/)).
- The g3-on-SVD-anchor control (corr pt +0.077, CI-hi 0.440) — the spine of the E disposition.
- Banked guardrails: drop "E>SVD"; E cosines are low-dim-inflated; g3 mis-application scoped to
  static reads (does NOT reopen 123/124).
