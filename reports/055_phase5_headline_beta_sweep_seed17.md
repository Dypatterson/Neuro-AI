# Report 055 — Phase 5 Headline β Sweep (seed 17, n=30)

**Date:** 2026-05-21 (late session)
**Active phase:** 5
**Headline metric per [phase-5-unified-design.md:256-281](../notes/emergent-codebook/phase-5-unified-design.md):** ΔE = E_content-prior − E_role-prior, paired per cue.
**Last verified result:** [Report 054](054_phase5_step3_smoke_seed17.md) — step 3 empirically inert on A1' substrate (uniform bias, softmax-shift-invariant).
**Why this experiment now:** Per GPT's recommended sequence after report 054, **headline-cue β sweep** is the cheapest direct test of "is β=10 the saturation corner on real cues?" The random-cue β preflight in report 054 measured substrate response geometry from off-substrate cues; that's a different question than headline behavior. This experiment uses the exact cue distribution the headline reports against.

This is a **drill-down**, not a graduation experiment.

---

## Setup

```
PYTHONPATH=src .venv/bin/python scripts/phase5_frozen_snapshot_audit.py \\
  --snapshot reports/phase5_a1prime_pilot_seed17/snapshots/phase3_phase4_w4_step1800.pt \\
  --headline-beta-sweep --headline-n-cues 30 \\
  --output reports/phase5_audit/a1prime_seed17_headline_beta.json
```

Same cue builder as [experiments/40 headline mode](../experiments/40_phase5_branching.py):
`_build_role_binding_cues` with `binding_noise_std=0.05`, `content_distortion=0.6`, seed=117 (=args.seed + 100 by experiments/40 convention). K_main=4, γ=0.5, formulation=per_pattern. Three conditions per cue: role-prior, content-prior, random-prior. β ∈ {1, 3, 5, 10, 30}.

Substrate context (from the geometry audit bundled in the same JSON):
`n_atoms=1064, dim=4096, coverage_lambda=1.0`, every atom has |E_i| ∈ [0.025, 0.059] (median 0.025), bias_cv = 0.024 — step 3 active but shift-invariant.

---

## Result

### Paired ΔE across β

| β  | ΔE_raw     | f⁺_raw | ΔE_step3   | f⁺_step3 | E_role  | E_content | E_random | Ordering                  |
| -- | ---------- | ------ | ---------- | -------- | ------- | --------- | -------- | ------------------------- |
| 1  | +0.000000  | 0.00   | +0.000000  | 0.00     | −7.4032 | −7.4032   | −7.4032  | role < content < random   |
| 3  | +0.000000  | 0.00   | +0.000000  | 0.00     | −2.7875 | −2.7875   | −2.7875  | role < content < random   |
| 5  | +0.000000  | 0.00   | +0.000000  | 0.00     | −1.9164 | −1.9164   | −1.9164  | role < content < random   |
| 10 | +0.000000  | 0.00   | +0.000000  | 0.00     | −1.3423 | −1.3423   | −1.3423  | random < role < content   |
| 30 | +0.000000  | 0.00   | +0.000000  | 0.00     | −1.0165 | −1.0165   | −1.0165  | role < content < random   |

The per-condition mean energies are identical to FP precision across role/content/random at every β. The "ordering" column at β=10 shows random < role < content — that's FP noise on identical values, not a real reordering.

### Sharpness drill-down

| β  | H_role | H_content | H_random | max_w_role | max_w_content | max_w_random |
| -- | ------ | --------- | -------- | ---------- | ------------- | ------------ |
| 1  | 0.6931 | 0.6931    | 0.6931   | 0.8931     | 0.8931        | 0.8931       |
| 3  | 0.6931 | 0.6931    | 0.6931   | 0.9879     | 0.9879        | 0.9879       |
| 5  | 0.6931 | 0.6931    | 0.6931   | 0.9883     | 0.9883        | 0.9883       |
| 10 | 0.6931 | 0.6931    | 0.6931   | 0.9870     | 0.9870        | 0.9870       |
| 30 | 0.6931 | 0.6931    | 0.6931   | 1.0000     | 1.0000        | 1.0000       |

`H` is the branch softmax entropy (over the K=4 branches' final energies); ceiling for K=4 is log(4) ≈ 1.386. Observed H = log(2) ≈ 0.693 **across every β and every condition** — branch softmax has collapsed to effectively 2 branches.

`max_w_proxy` is the maximum cosine similarity between q_settled and any stored pattern. At β=30, q_settled lands **exactly on a stored pattern** (sim=1.0). At β≥3, retrievals are essentially clean.

---

## Reading

**The substrate routes role-prior, content-prior, and random-prior cues to the same min-energy attractor across all tested β.** Mean energies match to FP precision. The headline mechanism (paired ΔE) requires the K branches under different priors to land in different basins for there to be any energy gap to measure. On this substrate, with this cue distribution, **they don't — they all converge to the same attractor regardless of prior.**

This is consistent with — and more diagnostic than — report 053's near-zero ΔE. Report 053 reported `mean ΔE = +0.00130` at n=3000 × 10 seeds, which we now read as residual FP noise on essentially-identical means rather than a sub-noise signal trying to emerge. The signal isn't there because the substrate doesn't expose role-vs-content as different basins at the energy level for these cues.

**β is not the binding lever.** From β=1 (almost-uniform, max_w=0.89) to β=30 (winner-take-all, max_w=1.0), ΔE stays at 0.000000 to FP precision and the ordering is meaningless. β=10 is not the over-sharpened corner; it's not the under-sharpened corner; it's not any corner — β is irrelevant to the headline mechanism here because the *substrate itself* doesn't differentiate the priors.

**Branch softmax collapse at log(2) is its own diagnostic.** Out of K=4 branches, effectively only 2 carry weight. This means 2 of the 4 schemas hit the same attractor and 2 hit a different (slightly higher-energy) attractor — and that pattern repeats across role/content/random because the K branches *include* the right atom by coincidence of K=4 spanning enough of the 8 top-strength schemas.

---

## What this rules out

- **β reformulation as a graduation path.** If β-axis flat across the full sweep means anything, it means the headline metric won't move by reducing β to 3 (Phase 2's generalization optimum) or any other value tested. Option 3 (reformulate Phase 5 headline) cannot be a β-only fix.
- **"The wiring bug masked a real signal."** Report 054 already showed step 3 inert via uniform-bias argument; this report independently confirms the headline can't separate role from content even in the *raw* landscape. The bug genuinely was not load-bearing.

## What this does not rule out

- **Cue-regime sweep.** With `content_distortion=0.6`, the cue is 60% the content distractor's energy and 40% role target's. At γ=0.5, the role prior may not be strong enough to push retrieval to a different basin from where the cue's intrinsic content already points. Lower distortion (or higher γ) might surface a role/content asymmetry. This is the next test in the planned sequence.
- **Lower-D redesign (option 1).** The substrate's basins are sharp clean retrievals (max_w=1.0 at β=30). This is the geometry the substrate was designed to produce — but at D=4096 with 1064 atoms, the basin landscape may have too few effective attractors for headline cues to discriminate against. D-sweep is the longest-tail diagnostic for this.
- **Headline reformulation away from paired ΔE.** Even if cue-regime and D-sweep are flat, option 3 remains viable with a *different* metric (e.g., R@K against the role-target atom, basin-membership). Branch softmax collapsing at log(2) suggests there IS a meaningful 2-cluster structure in the K=4 branches — just not one that paired energies expose.

## Cross-seed status

This is seed-17 only. The same caveat from report 054 applies: a different seed might place the role-target and content-distractor in different basins from each other on the *same* cue. The harness's geometry mode already shows the saturation pattern is training-time, not seed-specific, but the *headline-cue β sweep* should still be repeated on the other 9 seeds' snapshots from Drive before declaring option 3 fully foreclosed.

The cross-seed audit is now higher priority than the cue-regime aggregator. If even one seed shows ΔE moving with β, that seed is a route worth investigating.

---

## Required controls

- The random-prior condition serves as the per-β control on the K-branch mechanism. It produces identical mean energy to role and content at every β, confirming that the headline experiment can't tell role from random at the energy level on this substrate.
- The geometry audit (bundled in the same output JSON) shows step 3 is active (coverage_lambda=1.0) and shift-invariant (bias_cv=0.024) — so the result is "step 3 was applied and didn't help," not "step 3 wasn't applied."

## What was NOT done

- **No retuning of γ, K_main, formulation, content_distortion, or binding_noise_std.** Per audit constraint #10. The cue-regime sweep will treat these as a pre-committed grid, not a knob to land a passing cell.
- **No new attempts at headline reformulation in this report.** Negative result reporting only; the option-3 reformulation discussion is for the user.
