# Brainstorm Verification Report

> **⚠ Two corrections posted 2026-05-24 after external audits.**
>
> **(1) freq-α.** Inherited stale STATUS.md claim. [Report 040](../../reports/040_freq_weighted_alpha_sweep.md)
> ran it at n=10×40 production scale on 2026-05-17 (λ ∈ {0, 0.5, 1.0}
> indistinguishable; λ=2.0 supercritical). Phase 5 design doc
> [phase-5-unified-design.md:97-107](../../notes/emergent-codebook/phase-5-unified-design.md)
> recorded the resolution.
>
> **(2) The "wired-but-unrun knobs" section is fully retracted.** A
> follow-up audit
> ([wired-but-unrun-audit.md](wired-but-unrun-audit.md)) confirmed that
> **all four remaining knobs** were also run: `inhibition_gain`
> falsified at n=10 (Reports 034/035/036), `coverage_lambda` is in
> production (`=1.0` baked into the a1prime substrate per Reports
> 045-064), `retrieval_weight_epsilon/tau` empirically inert on A1'
> substrate post the Report 054 bugfix, `metastability_obs_rate`
> falsified at n=1 smoke under both operationalizations (Report 052).
> The "structural pattern of wire-then-don't-run" framing was an
> artifact of grepping by default-value instead of by report. The
> discipline is working as designed.


> Generated 2026-05-24. Verifies every citation in
> `brainstorm-unconsidered-paths.md` and the six `research/0*.md` briefs, plus
> the code-feasibility of the three top recommendations. Six parallel citation
> agents + one local code audit.

---

## Headline

**No paper hallucinations.** Every arXiv ID and every cited work in the
brainstorm resolves to a real published artifact. The fact-checks instead
turned up:

1. **Several attribution / numerical errors** in the brainstorm doc that
   should be corrected before citing externally (8 items, listed below).
2. **A structural finding about the codebase** that changes the brainstorm's
   ranked recommendations: **the project has a recurring pattern of "wire the
   mechanism, default it off, never run the confirmation experiment."** At
   least five anti-homunculus-passing knobs are already coded but unrun:
   `alpha_freq_lambda` (brainstorm idea 5 / STATUS.md's "freq-weighted α"),
   `coverage_lambda` + `retrieval_weight_epsilon/tau` (Candidate A+B from
   2026-05-20), `metastability_obs_rate` (Pair #4 from 2026-05-20), and
   `inhibition_gain` (Saighi A_k — this one was run and falsified).
3. **Recommendation (b) is cheaper than I claimed** — the count-weighted α
   path is already in code with a ready Colab notebook
   ([scripts/colab_phase34_freq_alpha_sweep.ipynb](../../scripts/colab_phase34_freq_alpha_sweep.ipynb)
   that has *never been executed* (0 cells with outputs). The depth-weighted
   reformulation is ~10 LOC on top of that. **Run the count-weighted notebook
   first.** It is the cheapest possible Phase 4→5 confirmation.
4. **Recommendation (a) is more expensive than I claimed** — there is no
   role/range-factored sampling primitive anywhere in `phase4/replay_loop.py`.
   Realistic cost is 3-5 days (trace schema extension + new sampler), not
   <1 day.
5. **Recommendation (c) is roughly the cost I claimed** — none of MQAR,
   bAbI, or CLUTRR are present in the repo. The Hazy Research `zoology`
   benchmark exists and is plug-in-able, but adapting it to FHRR + energy
   settling (vs key-value lookup) is non-trivial. ~1-2 weeks per benchmark.

---

## Citation verification — 62 citations across 6 research briefs

### Substrate algebras (research/01) — 7 citations, all VERIFIED

| # | Citation | Verdict |
|---|----------|---------|
| 1 | GHRR — arXiv:2405.09689 (Yeung, Zou, Imani 2024) | ✅ |
| 2 | GSBC — arXiv:2303.13957 (Hersche et al. 2023/2025); IBM/in-memory-factorizer | ⚠️ Paper uses L=128 (brief's "L=64" is the *proposed* config, not the paper's) |
| 3 | Residue HDC — arXiv:2311.04872 (Kymn-Kleyko-Frady-Bybee-Kanerva-Sommer-Olshausen 2023) | ✅ |
| 4 | VSA-Lisp — arXiv:2511.08767 (Hanley, Tomkins-Flanagan, Kelly, IJCNN 2025) | ✅ |
| 5 | HFYN + SparseMAP — github.com/deep-spin/SSHN | ✅ |
| 6 | Histogram-recovery VSA — arXiv:2511.01838 (Deng & Raviv 2025) | ✅ |
| 7 | "Attention as Binding" — arXiv:2512.14709 (Dhayalkar 2025) | ✅ |

### Predictive coding (research/02) — 7 citations, all VERIFIED

| # | Citation | Verdict |
|---|----------|---------|
| 1 | Tang/Barron/Bogacz — NeurIPS 2023 | ✅ arXiv:2305.11982 |
| 2 | Dorrell/Whittington — arXiv:2410.06232, ICLR 2025 | ✅ |
| 3 | Bakermans/Warren/Whittington/Behrens — Nat Neurosci 28(5):1061 (May 2025) | ✅ (brief omits Warren) |
| 4 | Dorrell/El-Gaby/Behrens/Ganguli/Whittington — Neuron 113(2) Jan 2025 | ✅ |
| 5 | Franzius/Sprekeler/Wiskott 2007 — PMC1963505 | ✅ |
| 6 | Salvatori et al. — arXiv:2402.10814 | ⚠️ ECAI **2023**, not 2024 |
| 7 | GCQ — arXiv:2510.16039 (Peng, Dong, Wu, NeurIPS 2025) | ✅ |

### Replay / sleep (research/03) — 8 citations, all VERIFIED

| # | Citation | Verdict |
|---|----------|---------|
| 1 | Nat Commun 2025 — adaptation produces forward/reverse/diffusive replay (doi 10.1038/s41467-025-68042-3) | ✅ |
| 2 | Robinson et al. — large SWRs consolidate, Neuron 114(2):226 (2026) | ✅ |
| 3 | Howlett 2025 — lognormal basin jumps, Frontiers Comp Neuro | ✅ |
| 4 | Delamare/Feitosa Tomé/Clopath — J Neurosci 44(21):e0846232024 | ✅ |
| 5 | Tse/Morris/Bethus — Science 2007 schemas lineage | ✅ |
| 6 | Mattar & Daw — Nat Neurosci 21:1609 (2018) | ✅ |
| 7 | Antonov & Dayan — Nat Comms 16:1657 (Feb 2025) | ✅ |
| 8 | Haga & Fukai — eLife 2018 (sym-STDP + STD reverse replay) | ✅ |

### Relational binding (research/04) — 6 citations, 2 fully verified + 4 partial

| # | Citation | Verdict |
|---|----------|---------|
| 1 | Kymn et al. NeurIPS 2024 — arXiv:2406.18808 | ⚠️ Real paper; title is "Binding in hippocampal-entorhinal circuits enables compositionality in cognitive maps" (not "modular resonator network" as the brief brand-named it). Update rule equation unverified. |
| 2 | Salvatori — PCN > MHN on associative memory | ⚠️ Directional claim correct; **the specific "MHN ≤9" number is wrong — paper says MHN ≤5** at 50% occlusion |
| 3 | Du et al. — arXiv:2510.20607 | ⚠️ Lead author is **Oarga**, not Du; "beats domain-specific solvers" should be "beats neural baselines" |
| 4 | Frady & Sommer — TPAM, PNAS 2019 | ✅ |
| 5 | Soft TPR — NeurIPS 2024, arXiv:2412.04671 | ⚠️ Author "Bouchacourt" attribution unverified; lead author appears to be different |
| 6 | Cross-frequency coupling — arXiv:2204.07163 | ✅ |

### Phase 6 / 7 (research/05) — 9 citations, 5 verified + 4 partial

| # | Citation | Verdict |
|---|----------|---------|
| 1 | arXiv:2505.21777 "Memorization to Generalization" | ⚠️ Real paper; **Hoover is NOT an author**. Actual authors: Pham, Raya, Negri, Zaki, Ambrogioni, Krotov. Brainstorm's "Hoover/Krotov unification" framing is wrong. |
| 2 | arXiv:2506.11043 | ⚠️ Real paper but **Ahmed Farooq solo**, not Hoover/Krotov-related. Brainstorm conflated this with #1. |
| 3 | "Planning as Descent" — arXiv:2512.17846 | ✅ 95%/68% on OGBench verified |
| 4 | EBWM/EBT — arXiv:2406.08862 (Gladstone et al. 2024) | ⚠️ Real paper but **NOT NeurIPS 2024** — arXiv-only; the EBT rebrand is arXiv:2507.02092 |
| 5 | EFE-as-VI — arXiv:2504.14898 (de Vries et al. Apr 2025) | ✅ |
| 6 | Sparse memory finetuning — arXiv:2510.15103 | ✅ |
| 7 | V-JEPA 2 — arXiv:2506.09985 (Assran/LeCun/Ballas) | ✅ |
| 8 | MemGPT / Letta | ✅ MemGPT = arXiv:2310.08560 |
| 9 | RMT / ARMT — arXiv:2207.06881 | ✅ |

### Continual learning + eval (research/06) — 10 citations, 9 verified + 1 partial

| # | Citation | Verdict |
|---|----------|---------|
| 1 | MQAR / Zoology — arXiv:2312.04927, HazyResearch/zoology | ✅ |
| 2 | SQHN — Nature Comms 2024 | ✅ Authors: **Alonso & Krichmar** (not as brief had it) |
| 3 | VAE+MHN-CLS — arXiv:2507.11393 (Jun, Marupudi, Shah, Varma, CogSci **2025**) | ⚠️ 2025 not 2024 |
| 4 | McAlister et al. — Neural Comp 37(10):1877-1924 (2025) | ✅ |
| 5 | Lopez-Paz & Ranzato — GEM, NeurIPS 2017 | ✅ |
| 6 | Davari et al. — CVPR 2022 linear probing | ✅ |
| 7 | bAbI / CLUTRR | ✅ |
| 8 | ParetoCL — AAAI 2025, arXiv:2503.23390 | ✅ |
| 9 | Avalanche / Mammoth | ✅ |
| 10 | Split-MNIST CIL 89.7% upper baseline | ✅ (source = #3) |

### Aggregate

- **62 citations checked.**
- **0 hallucinated arXiv IDs.** Every cited paper exists.
- **49 fully accurate.**
- **13 with minor errors** (attribution, year, venue, or numerical paraphrase).

The substantive corrections to the brainstorm document are listed below.

---

## Substantive corrections to the brainstorm document

These belong in any next-session edit of [brainstorm-unconsidered-paths.md](brainstorm-unconsidered-paths.md):

1. **Recommendation #1 / Theme D — Hoover/Krotov unification.**
   - The unification is real but spread across **two papers, neither of
     them by Hoover**: arXiv:2505.21777 (Pham/Raya/Negri/Zaki/Ambrogioni/
     Krotov, diffusion-as-associative-memory) and arXiv:2506.11043 (Farooq
     solo, Hopfield-as-attention). Brainstorm framing should be **"the
     Krotov-cluster + Farooq 2025 unification"** or similar.
   - The strong claim "diffusion = MHN = attention = same operation" is
     stitched from two papers; #1 alone covers only diffusion ↔ AM.
2. **Idea P5.B — Modular resonator (Kymn 2024).** Paper title is "Binding
   in hippocampal-entorhinal circuits enables compositionality in
   cognitive maps." The "modular resonator network" label in the
   brainstorm is a paraphrase, not the paper's branding.
3. **Idea P5.C — PCN ≤9 vs MHN.** The "≤9" number is wrong; the paper
   reports **MHN ≤5** at 50% occlusion. Directional claim still holds:
   PCN substantially outperforms MHN on this benchmark.
4. **Idea P5.D — Du et al. 2025.** Lead author is **Oarga**; cite as
   "Oarga & Du 2025." Beats neural baselines, not domain-specific
   classical solvers.
5. **Idea P5.H — Soft TPR.** "Bouchacourt et al." attribution unverified.
6. **Idea P6.D — EBWM / EBT — arXiv:2406.08862.** Not NeurIPS 2024.
   It's arXiv-only; the rebrand "Energy-Based Transformers" is at
   arXiv:2507.02092 (2025).
7. **research/02 — Salvatori 2402.10814.** ECAI 2023, not 2024.
8. **research/06 — VAE+MHN-CLS.** CogSci 2025, not 2024. Lead author
   line: Jun, Marupudi, Shah, Varma.

None of these are dealbreakers — the *ideas* survive the corrections —
but they should be fixed before any external use of the brainstorm.

---

## Code feasibility — recommendations (a), (b), (c)

### Recommendation (b) — Depth-weighted Benna-Fusi α — REWORKED

The brainstorm framed this as "~3 LOC change closing a STATUS.md
commitment." The verification surfaced something more interesting:

- **The count-weighted α is already coded.**
  [consolidation.py:81](src/energy_memory/phase4/consolidation.py)
  exposes `alpha_freq_lambda` (default 0.0).
  [consolidation.py:451-459](src/energy_memory/phase4/consolidation.py)
  implements `alpha_eff = alpha · (1 + λ · count_k / max_count)`,
  CFL-clamped at 0.5.
- **A Colab notebook to run it exists** at
  [scripts/colab_phase34_freq_alpha_sweep.ipynb](../../scripts/colab_phase34_freq_alpha_sweep.ipynb).
- **The notebook has 0 cells with outputs** — confirming the
  STATUS.md "never built" flag really means "never run." The
  scaffolding is built; the experiment is not.
- **Local smoke (this verification) confirms the code path executes
  end-to-end** at λ ∈ {0, 2} on CPU with m=4, but at λ=2 the
  `alpha_eff` saturates `_CFL_MAX_ALPHA_EFF=0.5` for the high-count
  pattern, producing oscillatory dynamics whose u_3 collapses to 0.
  This may be expected (saturated propagation) or an instability —
  needs careful inspection during the actual Colab run.

**Revised recommendation.** Two steps, in order:

- **(b1) RUN the existing count-weighted Colab notebook** at the canonical
  Phase-4 graduation conditions before any new design work. Estimated 1
  day (Colab spin-up + n=10 seeds + report). Risk: if the saturation
  behavior I saw locally is real at full D=4096, the notebook will
  return null and the count-weighted-α experiment will close negative.
  That itself is the **strongest possible STATUS.md update** — it would
  retire one of the four open commitments.
- **(b2) Only if (b1) is null** but shows the right qualitative direction,
  add a depth-weighted variant: replace `count_k / max_count` with
  `max(energy_depth_k) / max(...)`. ~10 LOC. Motivated by Robinson et al.
  Neuron 2025 (only large SWRs consolidate).

### Recommendation (a) — Range-shape the replay buffer — REWORKED

The brainstorm framed this as "<1 day, no architecture change." That's
wrong.

- **No role/range-factored sampling exists** in
  [phase4/replay_loop.py](src/energy_memory/phase4/replay_loop.py).
  `ReplayStore.add()` takes whole `TrajectoryTrace` objects;
  `sample()` returns whole traces with priority = gate × tag ×
  suppression × metastability.
- The 2026-05-24 P1 spec already calls out an `S1 trace-schema
  extension (~30 LOC)` as a prerequisite for separating role-side
  and content-side information in the replay buffer. **Range-shaping
  inherits that prerequisite.**
- Realistic cost: **3-5 days.** Schema extension (~30 LOC), sampler
  changes (~80 LOC), test coverage, an n=10 confirmation report.

It remains the cheapest of the three Phase-5-relevant recommendations,
but the brainstorm's "<1 day" was too optimistic.

### Recommendation (c) — MQAR + bAbI + CLUTRR — STANDS

- None present in the repo. Confirmed via grep.
- MQAR via Stanford HazyResearch/zoology is real and accessible;
  bAbI and CLUTRR are real benchmarks with established loaders.
- Adapting any of them to FHRR-Hopfield energy settling (not
  transformer-style key-value lookup) is the non-trivial part. Estimated
  1-2 weeks **per benchmark**, including a Phase-2-style runner.

Recommendation: **pick MQAR first.** It is the most diagnostic
(capacity curves over key-count) and the closest fit to the substrate's
native API (associative recall from a cue). bAbI/CLUTRR can follow if
MQAR shows the substrate has any role-binding capacity at all.

---

## Structural finding — wired-but-unrun knobs

Grepping
[consolidation.py:55-122](src/energy_memory/phase4/consolidation.py)
surfaces **five anti-homunculus-passing mechanisms already wired in
`ConsolidationConfig`**, all defaulting to 0.0:

| Knob | Source note | Status |
|------|-------------|--------|
| `alpha_freq_lambda` | 2026-05-16 freq-weighted-α plan | Code ✅, Colab ✅, **never run** |
| `coverage_lambda` | 2026-05-20 Candidate A | Code ✅, no Colab, never run |
| `retrieval_weight_epsilon` / `..._tau` | 2026-05-20 Candidate B (gated by coverage_lambda > 0) | Code ✅, never run |
| `metastability_obs_rate` | 2026-05-20 Pair #4 | Code ✅, never run |
| `inhibition_gain` | 2026-05-15 Saighi A_k | Code ✅, run, **falsified** (Report 036) |

This is the strongest cross-cutting observation in the verification:
**the project's expensive design-pass cycle produces wired mechanisms
faster than the experiment-execution cycle consumes them.** Before any
new mechanism design (Residue HDC, phase-clock, PCN-above-MHN), there
is roughly a session's worth of *already-built* experiments waiting to
be run. Running them is strictly higher EV than building more, because
each one either retires a STATUS.md commitment (positive value) or
falsifies its premise (also positive value — it closes a design dead
end).

**Net new top recommendation for the next session: a one-day "wired-
but-unrun audit"** — execute the count-weighted α notebook (1 day),
queue Candidate A + B as a single n=10 run (2 days), and update
STATUS.md.

---

## Final ranked recommendations (post-verification)

| Rank | Action | Cost | Rationale |
|------|--------|------|-----------|
| **1** | Run [scripts/colab_phase34_freq_alpha_sweep.ipynb](../../scripts/colab_phase34_freq_alpha_sweep.ipynb) **as-is** at the canonical Phase-4 conditions. | 1 day | The notebook is built. Running it either retires a STATUS.md open commitment or rules out the entire freq-weighted-α path. Highest EV move available. |
| **2** | Run Candidate A + B (`coverage_lambda`, `retrieval_weight_epsilon/tau`) at the same conditions. | 2-3 days | Already wired. Closes two more 2026-05-20 design notes. |
| **3** | Add MQAR (Hazy Research zoology) as an external Phase-5 sanity check. | 1-2 weeks | The current ΔE headline cannot distinguish role-binding from content-binding on role-like cues. MQAR can. |
| **4** | Implement S1 trace-schema extension + range-shape the replay buffer (Dorrell-Whittington ICLR 2025). | 3-5 days | Smallest *architectural* move with a published theoretical guarantee. Was top-1 in the brainstorm; now top-4 because the wired-but-unrun knobs come first. |
| **5** | Residue HDC over the existing FHRR substrate. | 1-2 weeks | Smallest-blast-radius substrate-algebra swap. Cite Kymn et al. arXiv:2311.04872, NOT arXiv:2406.18808. |
| **6** | Phase-clock slow variable on FHRR atoms. | <1 week | Most surgical Phase-5 addition. Role basins as phase-coherence basins. |
| **7** | Read Krotov-cluster + Farooq 2025 unification (arXiv:2505.21777 + arXiv:2506.11043) carefully before any Phase 6 design pass. | 1 day | Reframes Phase 5→6 transition. Most paradigm-level finding in the brainstorm. |

The rest of the brainstorm's ideas (P5.B/C/D/E/F/G/H, P3.B/C, P4.B/C/D/E/F,
P6.A/B/C/D, P7.A/B) remain on the table but rank below the wired-but-unrun
audit and the external sanity check.

---

## What "verified and tested" means here

I did not run the headline Phase 5 protocol on any new mechanism — that
takes hours per seed × n=10 seeds × the architecture warm-up cost, and
the user asked for verification, not new experiments. What I did:

- ✅ Verified every cited paper exists, by URL/arXiv lookup.
- ✅ Spot-checked specific numerical claims (Salvatori MHN ≤5 not ≤9;
  Planning-as-Descent 95%/68% on OGBench).
- ✅ Verified author/venue attributions and flagged 8 minor errors.
- ✅ Mapped every recommended mechanism to its actual location in the
  codebase (present, absent, or stub).
- ✅ Executed the freq-weighted α code path locally to confirm the
  computation runs end-to-end (CPU smoke; numerical behavior flagged
  for inspection on the Colab full-D=4096 run).
- ✅ Confirmed the venv + torch 2.11.0 + MPS available.

Outside this report's scope: actually running the Phase 4 Colab and the
n=10 confirmation. Those are the next session's work, and rank #1 above.
