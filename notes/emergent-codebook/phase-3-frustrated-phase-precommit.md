# FROZEN PRE-COMMIT — Phase-3 frustrated-phase oracle (LEAD 1, the novel-in-channel mechanism)

*Frozen 2026-06-02 BEFORE running. Branch `experiment/tem-local-reachability-oracle`. Harness:
`experiments/78_frustrated_phase_oracle.py` (reuses exp61/63/65/68 via importlib; substrate-free
screen, FHRR D=4096 port is rung-2/fenced). Origin: the 2026-06-02 invent-a-faithful-mechanism
workflow — the ONE genuinely-novel survivor (dedup canonical of the phase-oscillator cluster:
TGPM ⊕ PHASE-DEFLATION ⊕ frustrated-XY). It is the only candidate that (a) uses FHRR's PHASE channel
the entire 121-129 arc left idle, and (b) reaches the subdominant mode by a route the bound does not
cover — it INVERTS the target (frustration drives to the BOTTOM of the signed spectrum, not the
dominant mode every banked null power-iterates to).*

## 0. Preamble (CLAUDE.md experiment-preamble)

> **Active capability:** Codebook-growth (P3 structure), substrate-free oracle (IN-SCOPE; the FHRR
> D=4096 port is the user's build-gate, rung-2).
> **Headline metric:** within-paradigmatic-SET **label-shuffle B-KILL** on the settled-phase complex
> cosine, **CONJOINED** with `spec_ci_lo > 0` (the degenerate-code guard from the capacity grounding),
> CI-lo>0 in **≥8/10 seeds**. g3 is DEAD at n≤40 — diagnostic-only (the g3-misapplication bug fired
> twice, exp65/exp76; no auto-verdict may gate on it).
> **Required controls:** (a) ATTRACTIVE coupling (no +π) must NULL; (b) η=0 must give spec≈0;
> (c) collocational pairs must score LOWER than paradigmatic; (d) the inverse-co-occurrence-frequency
> phase arm (Report 128 §8) — frustrated-phase must BEAT it; (e) CEILING-GUARD: reach-but-NOT-exceed
> the global-NMF/SVD anchor; (f) within-set label-shuffle (the 126/129 hubness arbiter); + the
> calibration anchor (+0.1092/0.222 or INVALID).
> **Last verified:** invent-workflow adversary toy (+0.498±0.025 n=8; attractive-control +0.04 null;
> 0.50 < SVD-subdominant 0.59 = does NOT exceed ceiling). Toy ≠ real build_S (where 126/127/128
> toy-positives all died).
> **Why now:** the only novel-in-channel lead from the invent pass; cheap, substrate-free, in-scope,
> runs on the MacBook (D=64-256, no Colab).

## 1. The mechanism (the only new dynamic)

Per token i, only the PHASE `θ_i ∈ [0,2π)^D` evolves (modulus pinned to 1 — stays on the FHRR torus).
Offline replay-pass update (sleep-phase; same batch-offline class as grow_G), per dimension d:

    θ_i[d]  ←  θ_i[d]  +  η · Σ_j A_ij · sin( θ_j[d] − θ_i[d] + π )

`A = build_S` rows (the SAME fixed batch statistic grow_G uses). **The +π lag is load-bearing:** it
makes coupling REPULSIVE/FRUSTRATED — high-A (co-occurring/collocational) pairs are driven ANTI-phase
(the phase-domain image of H_anti pushing co-occurring tokens apart), so the dominant collocational
mode is *expelled* from the phase channel and the subdominant paradigmatic structure becomes the
phase-coherent survivor. This is gradient descent on the frustrated-XY energy
`E = −Σ_{ij,d} A_ij·cos(θ_i[d]−θ_j[d]+π)` (a sum of LOCAL pairwise terms; pre/post phase-difference +
a fixed weight = three-factor-clean). **Read:** para-vs-random + label-shuffle B-KILL via complex
cosine on the settled phasors `exp(i·θ)` (phase coherence IS the similarity; reuse `exp68.read_specificity`
on the real/imag-stacked phasor rows). The screen runs on real angle arrays mod 2π (substrate-free);
the FHRR D=4096 port is rung-2.

**Anti-homunculus:** (a) what moves = the per-dimension phase θ_i[d], by a fixed sinusoidal coupling;
(b) the "decision" (which tokens phase-cohere) lives in the FIXED POINT of the frustrated-XY energy —
no controller picks anything; (c) the frustration is a FIXED sign (+π), not a metric-triggered branch;
(d) control = the attractive (no +π) arm must null. Clean. (Drop the elaborate theta-gamma carrier —
inessential and risks a global-carrier homunculus reading; the minimal frustrated-XY is the mechanism.)

## 2. THE DECISIVE KILL-GATE (mandatory, pre-registered) — spectral-reduction probe

The frustrated-XY fixed point tracks the most-NEGATIVE-k eigenvectors of A — i.e. it may be a LOCAL
implementation of a signed-graph-Laplacian SPECTRAL RELAXATION = the global SVD by another name (the
Sanger/GHA ambiguity, and the user's "global = diagnostic only" rule). **Probe:** compute the
bottom-k eigenvectors of build_S directly; measure whether the settled-phase read reaches structure
the bottom-k eigenvectors alone do NOT.
- If the dynamic **merely recovers** the bottom-k spectrum → it is a **spectral flashlight, DEMOTE**
  it exactly as Report 124 §4 demoted the SVD-spectral +0.531 artifact / 125 §3 scoped Oracle-E. The
  conclusion then: the single-layer bound holds even in the phase channel under frustration → strong
  evidence the bound is FUNDAMENTAL for local fixed-capacity single-projection systems → the frontier
  is a genuine LOCAL MULTI-LAYER writer (Report 129 §56). *(A spec-tightening result — banked, not a defeat.)*
- If it reaches **MORE** than the bottom-k AND passes the §0 headline → the **FIRST local writer to
  break the single-layer bound through a channel no prior oracle used.** Escalate to the rung-2 FHRR
  D=4096 port BEFORE banking; honest ceiling = a LOCAL ROUTE to what global methods find, NOT "beats SVD."

## 3. PASS / NULL (frozen)

**PASS = ALL of:** B-KILL ≥8/10 ∧ spec_ci_lo>0 (§0 conjunction); beats the inverse-frequency phase
arm (d); attractive-arm NULLs (a) ∧ η=0 NULLs (b); collocational < paradigmatic (c); reach-but-NOT-
exceed the global ceiling (e); **AND the kill-gate: reaches MORE than bottom-k (§2).**
**NULL sub-cases:** (i) **hubness** — label-shuffle kills it (para-set hubness, the 126/129 mode; the
single likeliest failure — a Kuramoto field is prone to hub-slot domination); (ii) **deflates-to-PMI**
— the inverse-frequency arm matches it (the 5th narrowing of the year, after 126/127/exp70/128 §8);
(iii) **spectral-flashlight** — merely recovers bottom-k (§2 → demote + spec-tighten). **INVALID** =
anchor misses +0.109/0.222.

## 4. Build checklist

- [ ] Reuse via importlib: `exp61.{build_S, build_cooccurrence, build_sppmi, pick_k_by_density,
      load_corpus}`, `exp63.{select_pairs, raw_sppmi_svd_anchor}`, `exp65.nmf_slots` (ceiling),
      `exp68.{read_specificity, derange_partners}`.
- [ ] New code only: `frustrated_xy(A, D, eta, iters, sign=+π|0)` (the settling), `phase_read`
      (complex-cosine specificity), `spectral_reduction_probe` (bottom-k eigvecs of build_S +
      reach-beyond test), the inverse-frequency phase control arm.
- [ ] Screen substrate-free: real θ arrays mod 2π, D∈{64,128,256}; FHRR D=4096 port = rung-2 (fenced).
- [ ] Anchor every run (INVALID guard); planted-corpus smoke BEFORE WikiText (apparatus must recover a
      planted lift; attractive-control must null on planted too).
- [ ] Differential reporting ONLY (para−rand, label-shuffle, frustrated−attractive, frustrated−freq);
      never absolute cosine (inflation caught 3× R125/127/129).
- [ ] Freeze: η, iters, the +π sign, D-grid, k for the bottom-k probe, n=10, B-KILL≥8/10 ∧ spec_ci_lo>0,
      the 4 PASS/NULL sub-cases, SimLex≥5 sha — all before the WikiText run.
- [ ] Adversarial 3-lens verification BEFORE banking any positive (the 129 pattern).
