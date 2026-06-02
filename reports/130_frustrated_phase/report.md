# Report 130 — Frustrated-phase oracle (LEAD 1): NULL — the FHRR phase channel does not carry paradigmatic structure on real build_S; the single-projection bound is CHANNEL-INVARIANT

**Status:** NULL (clean, flat across D∈{64,128,256}, n=10; B-KILL 0/0/1 of 10). Spec-tightening result. Banked.
**Date:** 2026-06-02. **Branch:** `experiment/tem-local-reachability-oracle`.
**Harness:** `experiments/78_frustrated_phase_oracle.py`. **Precommit:** [phase-3-frustrated-phase-precommit.md](../../notes/emergent-codebook/phase-3-frustrated-phase-precommit.md).
**Origin:** the 2026-06-02 invent-a-faithful-mechanism workflow — the ONE genuinely-novel survivor (the phase-oscillator cluster; uses FHRR's phase channel the 121-129 arc left idle).

## Preamble

- **Active capability:** Codebook-growth (P3 structure), substrate-free oracle (in-scope).
- **Headline:** within-paradigmatic-set label-shuffle **B-KILL** on the settled-phase complex cosine, CONJOINED with `spec_ci_lo>0`, ≥8/10 seeds.
- **Required controls:** attractive (no +π) must null; η=0 null; collocational<paradigmatic; the inverse-frequency phase arm (128 §8) must be beaten; CEILING reach-not-exceed; the spectral-reduction kill-gate; the +0.1092/0.222 anchor.
- **Last verified:** invent-workflow adversary toy (+0.498±0.025; attractive null). **Apparatus re-validated here by the planted smoke** (frustrated +0.091, attractive −0.231).
- **Why:** the only novel-in-channel lead; it reaches the subdominant mode by INVERTING the target (frustration → bottom of the signed spectrum), a route the bound does not formally cover.

## Result (WikiText-2, anchor VALID +0.1092/0.222)

| D | frustrated spec (CI-lo) | B-KILL | attractive | η=0 | freq-arm |
|---|---|---|---|---|---|
| 64  | −0.0023 (−0.040) | 0/10 | +0.001 | ~0 | +0.002 |
| 128 | −0.0022 (−0.028) | 0/10 | +0.011 | ~0 | +0.000 |
| 256 | +0.0055 (−0.013) | 1/10 | −0.000 | ~0 | +0.001 |

Ceiling: NMF(k32) +0.169, SVD anchor +0.109. (Spectral-ref top-32 +0.386 — see harness caveat.)

## Verdict — NULL, and precisely *which* null

**The toy positive (+0.498) collapsed to ≈0 on real WikiText `build_S`** — exactly the path the adversary flagged and that killed 127's k-WTA and 128's replay toy-positives. The frustrated-phase dynamic produces **no paradigmatic signal at all** on real co-occurrence: spec ≈ 0 across all three D, CI-lo negative everywhere, B-KILL 0–1/10.

It is **not** any of the "interesting" null sub-cases:
- **Not hubness** — there is no positive spec to decompose (spec ≈ 0, not "positive-but-shuffle-fragile").
- **Not deflation-to-PMI** — the inverse-frequency arm is *also* ≈0 here; neither works.
- **Not a spectral flashlight** — frustrated (≈0) is nowhere near the spectral reference; the kill-gate was never reached. The mechanism simply doesn't reach the structure.

**Apparatus is valid** (not a broken-dynamic null): the planted smoke produced frustrated +0.091 with the attractive control at −0.231, so the dynamic *does* settle and *does* produce signal when clean cluster structure is present. On real `build_S` it produces nothing — the toy positive was an artifact of the planted clean clusters.

## What this banks — the spec-tightening clause

The single-projection / single-layer bound is **CHANNEL-INVARIANT**: it holds for the **magnitude** channel (121–129) *and* the **phase** channel (here), under frustration that explicitly inverts the convergence target. This rules out "we just weren't using the right substrate affordance" — the most creative substrate-native swing (FHRR phase, theta-gamma) does not break the wall. **New clause in the spec the walls are building:** *the wall is not about which channel or which direction a single-projection local dynamic converges to; it is about single-projection itself.* That points harder at the genuine LOCAL MULTI-LAYER frontier (Report 129 §56) — and it means the phase channel is closed as a *single-layer* escape (a multi-layer phase mechanism is a different, untested object).

## Honest caveats / harness notes

- **Spectral-ref inflation (carry forward):** the top-32 eigenvector read scored +0.386 — low-dim cosine inflation (Report 125 §3: random-pair cosine +0.47–0.67 at k=8–32), far above the calibrated rank-300 SVD anchor (+0.109). So the kill-gate's `spectral_ref` was contaminated; any future phase/spectral oracle must use a **rank/dim-matched** spectral reference, not raw top-k. Moot here (frustrated ≈ 0, nowhere near it).
- **Operator scope:** run on `build_S` as the precommit froze. A frustrated dynamic on `M_SR`/`M_trans` (where paradigmatic is the slow/dominant axis) is an untested variant — but given the *complete* flatness (≈0, not "weak-but-present"), an operator artifact is unlikely; banked as NULL on `build_S`.
- **Not adversarially 3-lens-verified** (the precommit reserves that for a *positive*); a flat null across 30 cells with a smoke-validated apparatus is low false-null risk. Available on request.

## Disposition

Phase channel **closed as a single-layer escape.** Per the standing principle (an all-died/null round is iterate-fuel + the wall builds the framework): this is round-one of the invented mechanisms drying up *and* a new spec clause. The accumulating evidence (121–130) now points decisively at the **genuine local multi-layer** frontier and/or the **capacity/adaptive-assignment** lever (the capacity sweep, [Report 131], running). Next ideas-round, if pursued, should target multi-layer or a non-single-projection primitive — not another single-projection channel.
