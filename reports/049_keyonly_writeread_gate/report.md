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

## Verdict (the fork)

Combined with G-A (exp 048: frozen-refit shows **no readout defect**):

- **The "no key-only basin for any operator" wall (065/066) is a STORE-AS-IS wall, not a binding-algebra wall.** A heteroassociative write *carves* the key→value basin the bundle lacks, robustly and at high capacity. → **the FHRR substrate CAN hold the basin; the defect is the WRITE STRUCTURE.**
- This is strong, consistent evidence for **surgical-in-place** and **against rebuild-from-Phase-1**: the algebra is not the wall, the readout is not the wall (G-A), the capacity scales with D (G-C). None of the rebuild-trigger conditions fired.
- **Connects to MESH** (`pdf:mesh-2022`, in corpus: "fixed scaffold + heteroassociation avoids the content-addressable-memory cliff") — exactly the structure that rescues here. The surgical direction is: **replace store-as-is bundle / MHN-over-bound-pairs with a heteroassociative write** for role/key→value recall.

## Caveats (bound the claim)

1. **The write SHAPE doesn't differentiate here.** Hebbian = delta = swap-contrastive = bounded-one-shot to ~3 decimals, because random keys are near-orthogonal so plain heteroassociation already saturates the task. The contrastive/margin/BTSP advantages (and the phase3b "self-mined negative is a thermostat" story) would only surface with **correlated keys**, a small value codebook with collisions, or at the capacity edge. So the surgical recommendation simplifies: it is **"do an associative write at all (vs store-as-is)," not "find a clever contrastive negative."**
2. **A dense D×D weight W is a new mechanism** vs the project's FHRR-codebook+MHN. Anti-homunculus-clean (fixed offline rule, no runtime arbitration), but integrating heteroassociation into the architecture (and at D=4096 the 16M-param cost) is the surgical design question — see MESH for the fixed-scaffold form.
3. **BTSP (G-0) used a magnitude-bounded proxy**, not the faithful sparse-binary rule (card `biorxiv:2025.05.15.654220` = transfers-with-caveats, substrate mismatch). The faithful port is unnecessary for *this* verdict (plain heteroassociation already rescues) but remains open if a one-shot/online variant is wanted.

## Next

- Stress the write SHAPE: correlated-key / colliding-value regime where Hebbian fails and a margin/contrastive write should separate (the regime where the phase3b lesson bites).
- Carry the verdict to a surgical design note: heteroassociative role→target write for Phase 4 consolidation (MESH-style fixed scaffold), with the anti-homunculus check.
