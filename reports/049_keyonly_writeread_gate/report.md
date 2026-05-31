# Key-only write-then-read gate (G-0 / G-B / G-C) — the fork adjudicator

> **Status:** decision-relevant drill-down. The headline finding (a write rescues
> the key-only null where store-as-is fails) is **robust across N and D, 3 seeds**.
> Caveats below bound the claim. Harness: `experiments/49_keyonly_writeread_gate.py`.
> Spec: `notes/emergent-codebook/phase-3-consolidation-write-design.md`.

## Experiment preamble

- **Active phase:** Phase 3 — consolidation-write sub-program (Stage-1, pre-Phase-5′, fence down).
- **Headline per [phase-3-consolidation-write-design.md §Headline]:** value-codebook `top_index_hits` Selectivity-Δ + the vs-no-write anchor `Δ(write) − Δ(store-as-is)`.
- **Required controls per [§Required controls]:** shuffled-key selectivity, random-codebook collapse, perfect-cue upper bound, the N/D capacity surface (G-C).
- **Last verified result:** Reports 065/066 — `bind(k,v)` has no key-only basin for any operator (`tix→chance` with N); bundle-first the sole rescue at small N.
- **Why now:** this is the binding-algebra adjudicator the grill (q4) promoted to a co-equal run-first gate — a null here forces the rebuild branch.

## Setup

Key-only recall (the exact 065/066 read): store N `(key, value)` pairs; **cue a key alone**, recover its value by basin-membership (`top_index_hits`) over the C-atom value codebook. `store_as_is` = bundle `B=Σ bind(k_i,v_i)`, recover via `cleanup(unbind(B,k_i))` — the project's current structure. Write arms build a heteroassociative map and recover via `cleanup(W k_i / D)`. D=512, C=32 (chance 0.031), 3 seeds.

## Result 1 — capacity surface (N-sweep, G-C)

| N/D | store-as-is (bundle) | associative write (any rule) |
|---|---|---|
| 0.25 | 0.583 | 1.000 |
| 0.50 | 0.342 | 1.000 |
| 1.0 | 0.225 | 1.000 |
| 1.5 | 0.141 | 0.998 |
| 2.0 | 0.122 | 0.998 |
| 3.0 | 0.097 | 0.983 |
| 4.0 | **0.086** | **0.943** |

Store-as-is decays toward chance (reproducing the 065/066 key-only null — and consistent with bundle-first working only at small N, 066). The associative **write holds ≥0.94 even at N=4×D.** Controls clean: shuffled-key ≈ chance (0.023), random-codebook collapses (0.008).

## Result 2 — D-sweep at fixed N=1024 (G-C direction)

| D | N/D | store-as-is | write |
|---|---|---|---|
| 256 | 4.0 | 0.089 | 0.924 |
| 512 | 2.0 | 0.122 | 0.998 |
| 1024 | 1.0 | 0.188 | 1.000 |
| 2048 | 0.5 | 0.301 | 1.000 |

**D↑ helps, D↓ hurts** — the lever is **N/D**, not D alone. This confirms the grill q5 correction: "lower-D rescues" is **mechanistically backwards** (lowering D at fixed N raises crosstalk). The capacity-wall fix is dimension/orthogonalization (↑D) or fewer items (↓N), never ↓D.

## Result 3 — correlated-key stress (does the write SHAPE matter?)

Random keys are near-orthogonal, so Result 1's writes were indistinguishable. Blending a shared component into the keys (`--key-rho`) raises pairwise key cosine and stresses the write. N=256, D=512, 3 seeds:

| key cosine | store | hebbian | delta | swap-contrastive |
|---|---|---|---|---|
| 0.025 | 0.342 | 1.000 | 1.000 | 1.000 |
| 0.028 | 0.322 | 1.000 | 1.000 | 1.000 |
| 0.126 | 0.171 | **0.595** | **0.947** | **0.956** |
| 0.770 | 0.052 | 0.060 | 0.225 | **0.309** |
| 0.969 | 0.048 | 0.051 | 0.066 | 0.064 |

**The write shape matters once keys correlate.** At cosine 0.126 plain Hebbian drops to 0.60 while the error-correcting (delta) and contrastive (swap) writes hold ≥0.95. At cosine 0.77 (heavily correlated) store-as-is and Hebbian are both at chance (~0.05–0.06) but the contrastive/margin writes still extract **0.23–0.31** (5–7× chance) — and **swap-contrastive > delta** (0.31 vs 0.23), i.e. the *precommitted* swap-negative adds value beyond pure error-correction. (At cosine 0.97 the keys are near-degenerate and everything collapses — expected.)

This **vindicates the plan's margin/contrastive emphasis** and resolves the phase3b narrative: the contrastive *idea* was right; phase3b failed because its negative was **self-mined via `sims.argmax` (a runtime thermostat)** plus a top-1 readout — the **precommitted swap-negative** (anti-homunculus-clean) is the fix, and it genuinely helps under correlation.

## Result 4 — decorrelation rescues the correlated-key collapse (validates the redirect)

Report 050 found the write collapses under real key correlation and redirected to a
decorrelating objective. Testing decorrelation in the limit (ZCA **whitening** of the
keys — the upper bound of any learnable decorrelator), `--whiten-keys`, on the
correlated regime (rho=0.6, key cosine ~0.77):

| | store | hebbian | delta | swap |
|---|---|---|---|---|
| no-whiten | 0.052 | 0.060 | 0.225 | 0.309 |
| **+ whiten** | 0.279 | **1.000** | **1.000** | **1.000** |

Whitening **fully rescues** the write. N-sweep at rho=0.6 (delta_W) pins the reach:

| N/D | no-whiten | whiten |
|---|---|---|
| 0.25 | 0.44 | 1.00 |
| 0.50 | 0.22 | 1.00 |
| 1.00 | 0.12 | 1.00 |
| 1.50 | 0.07 | 0.99 |
| 2.00 | 0.07 | 0.89 |

**The collapse was ill-conditioning, not a rank wall** — decorrelation recovers full
performance up to **N≈D** (the rank capacity), degrading gracefully past it. Two
load-bearing distinctions:
- This is **key/cue-space** decorrelation. The project's existing `repulsion_force`
  (`torch_fhrr.py:147-182`, a `d_eff`-maximizing **atom-space** spread, already in the
  Frame-B baseline at `repulsion_step_size=0.05` and run at `100.0` in the Phase-5
  pilots 045-053) is a **different** lever — it spreads stored atoms, not the
  correlated context cues. So cue-space decorrelation is **genuinely untested**, not
  redundant with the baseline repulsion.
- It is **rank-bounded**: it rescues ill-conditioned (full-rank correlated) keys up to
  N≈D; it cannot manufacture capacity when keys are genuinely rank-deficient (N≫D) or
  ill-posed (single-role cue, rank ≤ K roles).

**Open question for the real-data gate:** does the real role-binding bottleneck fall in
the ill-conditioned (decorrelation-helpable, N≤D) regime, or the rank-bound/ill-posed
regime (where nothing helps and rich-context store-as-is already wins, Report 050)?
This is what a learnable cue-space decorrelator (FEP self-orthogonalizing, the learnable
approximation of whitening) must be tested against next.

## Verdict (the fork)

Combined with G-A (exp 048: frozen-refit shows **no readout defect**):

- **The "no key-only basin for any operator" wall (065/066) is a STORE-AS-IS wall, not a binding-algebra wall.** A heteroassociative write *carves* the key→value basin the bundle lacks, robustly and at high capacity. → **the FHRR substrate CAN hold the basin; the defect is the WRITE STRUCTURE.**
- This is strong, consistent evidence for **surgical-in-place** and **against rebuild-from-Phase-1**: the algebra is not the wall, the readout is not the wall (G-A), the capacity scales with D (G-C). None of the rebuild-trigger conditions fired.
- **Connects to MESH** (`pdf:mesh-2022`, in corpus: "fixed scaffold + heteroassociation avoids the content-addressable-memory cliff") — exactly the structure that rescues here. The surgical direction is: **replace store-as-is bundle / MHN-over-bound-pairs with a heteroassociative write** for role/key→value recall.

## Caveats (bound the claim)

1. **The write SHAPE matters under correlation (Result 3 resolves this).** With random/near-orthogonal keys all write rules saturate the task and look identical; but once keys correlate (cosine ≥ 0.13, the realistic regime), plain Hebbian collapses while error-correcting (delta) and contrastive (swap) writes hold, and the precommitted swap-negative beats delta at high correlation. So the surgical direction is **an error-correcting / contrastive heteroassociative write with a precommitted swap-negative — not plain Hebbian, and not the self-mined `argmax` negative phase3b used.**
2. **A dense D×D weight W is a new mechanism** vs the project's FHRR-codebook+MHN. Anti-homunculus-clean (fixed offline rule, no runtime arbitration), but integrating heteroassociation into the architecture (and at D=4096 the 16M-param cost) is the surgical design question — see MESH for the fixed-scaffold form.
3. **BTSP (G-0) used a magnitude-bounded proxy**, not the faithful sparse-binary rule (card `biorxiv:2025.05.15.654220` = transfers-with-caveats, substrate mismatch). The faithful port is unnecessary for *this* verdict (plain heteroassociation already rescues) but remains open if a one-shot/online variant is wanted.

## Next

- **Surgical design note** for Phase 4: an **error-correcting / contrastive heteroassociative** role→target write (MESH-style fixed scaffold; delta-rule + precommitted swap-negative), with the anti-homunculus check and the integration cost at D=4096. The verdict is set; this is the build.
- Port the write into the project's actual FHRR-codebook+MHN consolidation path (the toy used a standalone W); confirm the rescue survives integration on a real corpus slice.
- Optional: the faithful sparse-binary BTSP port (G-0) if a one-shot/online variant is wanted — not needed for the verdict.
