# FROZEN PRE-COMMIT (grilled via /grill-with-docs — Q1–Q6 resolved inline 2026-06-01) — CE-1 ⊗ 127: emergent Replay-interleaving × an online NONLINEAR-PARTITION writer

*Drafted 2026-06-01 (branch `audit/coupled-null-reeval`) as the Stage-2 lead from the coupled
null-audit ([null-audit-coupled.md](null-audit-coupled.md)). **Status: FROZEN (grilled 2026-06-01;
Q1–Q6 resolved inline in §6). The rung-1 planted smoke + WikiText screen are SUBSTRATE-FREE (no
build-gate); only the rung-2 FHRR-port + the A100 run need the Abstraction build-gate (the user's to
clear).** This precommit MERGES two short-list entries the audit's completeness critic showed are the
**same live channel**: CE-1 (Codebook-growth × emergent Replay-interleaving;
[combination-experiments.md](combination-experiments.md)) and the 127 online-streaming-k-WTA next-move
([Report 127 §0.5](../../reports/127_nonlinear_competition/report.md)). Substrate-free oracle; floor
(055-058) untouched. Reuses `experiments/61/63/65/68` verbatim via importlib.*

## 0. Preamble (CLAUDE.md experiment preamble — mandatory)

> **Active capability:** **Codebook-growth ⇄ Replay (a COMBINATION)** — CONTEXT.md §4 capability-DAG;
> first-class per [combination-experiments.md](combination-experiments.md).
> **Headline metric per [combination-experiments.md:52-54]:** the within-set **label-shuffle B-KILL**
> pair-specific residual (real − within-para-shuffled, bootstrap CI-lo > 0 in ≥4/5 seeds) of the
> **INTERACTION** — Arm B (replay-interleaved stream → online k-WTA) must beat **both** Arm A (as-is
> stream → same writer) **and** the emergent-priority must be operative (the random-reorder gauge must
> NOT reproduce it). This is a **DRILL-DOWN oracle, NOT a phase-graduation experiment** (per CLAUDE.md:
> the design-spec graduation headline is paradigmatic "similar tokens → similar hypervectors"; this is a
> feasibility probe of one coupling) — specifically it is **rung 1 of the fidelity ladder (§4.5)**: a
> screen-pass licenses escalation up the ladder, NOT a graduation claim.
> **Required controls per [combination-experiments.md:48-54] + the 127 lesson:** within-set
> label-shuffle B-KILL (headline); **random-reorder gauge** (reordering stripped of emergent priority);
> the **competent OFFLINE global partition** (k-means / batched k-WTA on the full `build_S` — the 127
> ceiling, the control that caught 127's overclaim); a **static-surprise-reweight** control (isolates the
> two-timescale interleaving from a one-shot reweight — the Oracle-C / surprise≈PMI guard); the LINEAR
> `grow_G` floor (+0.021) and the GLOBAL NMF ceiling (Oracle E, +0.19), both in-harness; the
> raw-SPPMI-SVD calibration anchor (+0.1092 CI[0.082,0.137] / kq 0.222 — INVALID otherwise); 5 seeds;
> SimLex≥5 n=40 frozen pairs.
> **Last verified result:** Report 127 (k-WTA on `build_S` clears the gate but a global k-means reproduces
> it → nonlinear-PARTITION, not competition); the PM-7b throwaway prototype (online-local k-WTA **loses**
> the B-KILL margin to offline k-means: +0.088 < +0.208) — so **online-ness alone does NOT buy the
> margin**, which is exactly why this probe isolates **emergent interleaving** as the distinct ingredient.
> **Why this experiment now:** the coupled null-audit found the live set collapses to **one channel** —
> emergent replay-interleaving feeding a nonlinear-partition writer — and that a *linear*-writer CE-1 is
> predicted to NULL (it collapses into the 123 bound). This is the cheapest decisive test of whether the
> 121-127 nulls are **false negatives of isolation-testing** or **replay-invariant**.

## 0.5 The CE-1 = 127 merge (why this is one experiment, not two)

The audit's load-bearing correction: **CE-1 only escapes the bound in its NONLINEAR-PARTITION form.**
123 proved the *linear* channel (`grow_G` power-iteration) reaches only the dominant collocational mode
regardless of stream reordering — so a CE-1 that reorders the stream feeding a *linear* writer just
reweights inputs to the same fixed operator and reproduces 123 (= NO in the audit). 127 proved the
*nonlinear-partition* channel (k-WTA / k-means on `build_S`) **does** reach paradigmatic structure
(+0.25) — but only demonstrated it **offline/global**, and the PM-7b prototype showed an **online-local**
k-WTA underperforms the offline partition. **The merged question:** does an **emergent, pattern-separated
replay interleaving** of the windows the *online* k-WTA integrates close the online→offline gap — i.e.
does replay manufacture the effective statistics that only a global pass otherwise sees? This is the
audit's #1 entry, and it is the one place the 121-127 wall could be a false negative of isolation.

## 1. Anti-homunculus / discipline (must pass before building)

The combination is **not** a license to hand-wire. Per [combination-experiments.md:18-28, 63-66]:

- **Who decides the replay order?** A **local priority scalar per stored window = novelty × surprise**
  (combined *multiplicatively*, not a hard AND-gate — keeps it a smooth fixed dynamic, not a thresholded
  branch; grill Q1): **novelty** = how *thinly the window's tokens are represented in the system's current
  codes* (under-consolidated — a property of the system's state, NOT raw corpus rarity), and **surprise**
  = how poorly the *current* codes complete/resolve the window (the settling-residual). Replay what is
  *both* under-represented AND unresolved. A **dissimilarity-spacing** rule adds **pattern separation**
  (interleave windows whose tokens currently share a code). NO supervisor reads "is this king/queen?"; NO
  if-X-then-Y branch; NO hand-set king/queen curriculum. The interleaving is a **measurement of a local
  dynamic**, not an arbitration. **If the only pass requires hand-picking the schedule, the result is VOID.**
- **The random-reorder gauge is the homunculus tripwire.** A *random* replay reorder (same priority
  distribution shuffled, stripped of the local signal) must **NOT** reproduce Arm B. If it does, the
  "emergent priority" was inert and any Arm-B lift is a reordering artifact → NULL.
- **The writer is the 127 nonlinear-partition k-WTA** (`kwta_features` lineage): a FIXED PRECOMMITTED
  top-`cap` rank-threshold applied uniformly every step — a fixed dynamic, not a metric-read branch
  (admissible per the 127 grill Q3). A **SOM-BMU writer remains BANNED**. `cap`, `k`, learning rates are
  **FROZEN, never swept-to-pass** (the metric-fishing invariant-violation).
- **Batch-offline (sleep/wake):** replay is offline replay over a **frozen window buffer**; the
  settling-residual is computed against the *current* (slow) codes each replay epoch — a two-timescale
  loop, **no runtime error-driven writes**. The writer updates by local Hebbian expansion + a fixed cap
  (no error signal, no reward) — the same compliance class as 127's k-WTA.
- **Surprise≈PMI guard (the Oracle-C trap, [whats_missing.md §1a]):** a one-shot surprise/novelty
  *reweight* of `build_S` is already close to what SPPMI's PMI term does. The **static-surprise-reweight
  control** is mandatory: Arm B must beat **both** Arm A **and** the static-reweight control, or the
  result is "a reweighting in disguise" and collapses to 123. The load-bearing new claim is the
  **two-timescale INTERACTION** (slow codes → per-window surprise → reordered fast replay → updated slow
  codes), not a single re-weighting.

## 2. The arms (everything else held fixed)

Same WikiText-2 frozen windows (V≈2002, W=6, γ=0.9 — exp68 defaults), same operator `build_S`
(SPPMI 2nd-order = the +0.021 `grow_G` floor operator AND the operator 127's k-WTA partitioned), same
n=40 SimLex≥5 non-cooc pairs (`exp63.select_pairs`), same gauge-free means-based para-vs-random
specificity + label-shuffle (`exp68.read_specificity`, `exp68.derange_partners`), same calibration anchor.

The writer is the **online/streaming** k-WTA: process windows (or window-derived `build_S` row
contributions) in a given **replay order/sampling**, updating `W` online per replay epoch (vs exp68's
`kwta_features`, which batches the static full-corpus operator — that batched form is the **offline
ceiling control**, not an arm).

- **Arm A — isolation baseline (reproduce the prototype's losing arm):** online k-WTA fed the **as-is
  corpus-order** window stream. Expected ≈ the PM-7b +0.088 (well below the offline +0.208). This is the
  121-127 isolation null, reproduced with the nonlinear-partition writer.
- **Arm B — the combination (the bet):** the **same** online k-WTA fed a **replay-reordered /
  pattern-separated** window stream, where replay **priority** = **novelty × surprise** per window
  (novelty = how thinly the window's tokens are represented in the *current* codes; surprise = how poorly
  the current codes resolve the window) and **pattern separation** = a dissimilarity-spacing interleave
  (windows whose tokens currently co-occupy a k-WTA assembly are spaced apart). Priority is recomputed
  each replay epoch against the current (slow) codes — the **two-timescale loop** ("what's surprising"
  changes as the system learns; a one-shot reweight is the static-reweight *control*, not this).
- **Gauge — random-reorder:** the **same** online k-WTA fed a stream reordered by a *random* permutation
  drawn from the same sampling-rate marginal (priority distribution preserved, the **local signal
  destroyed**). Isolates "did the *emergent priority* do the work, or did *any* reorder?"
- **Ceiling control — offline global partition:** batched k-WTA (`exp68.kwta_features`) **and** a plain
  Lloyd **k-means** one-hot on the full `build_S` (the 127 competent control — the thing the online-local
  writer must approach). The online-vs-offline gap is the quantity Arm B must close.
- **Static-surprise-reweight control:** offline partition of a **one-shot** surprise-reweighted
  `build_S` (reweight rows/cols by the per-window surprise aggregated once, then partition). Isolates the
  two-timescale interleaving from a single reweight.
- **Anchors (in-harness, every run):** LINEAR `grow_G` floor (+0.021); GLOBAL NMF ceiling
  (`exp65.nmf_slots`, +0.19); raw-SPPMI-SVD calibration (+0.1092 / kq 0.222 or INVALID).

## 3. Frozen gate (PASS = ALL; the INTERACTION + the random-reorder gauge are the headline)

A run PASSES iff **ALL** of:

- **g-interaction (the headline, [combination-experiments.md:52-54]):** Arm B's within-set
  **label-shuffle B-KILL** pair-specific residual bootstrap **CI-lo > 0 in ≥4/5 seeds**, **AND** Arm B's
  para-vs-random specificity beats Arm A by **Δ_AB ≥ +0.05** (FROZEN grill Q5; anchored between the +0.021
  floor / ≈+0.088 online-A and the ≈+0.208 offline ceiling — meaningful but hittable), **AND** Arm B beats
  the **static-surprise-reweight**
  control by **≥ +0.02** (the two-timescale interaction is the active ingredient, not a reweight).
- **g-gauge (the anti-homunculus tripwire):** the **random-reorder** gauge's B-KILL ≈ Arm A (gauge does
  **NOT** reproduce Arm B; gauge − Arm-A specificity < +0.02). If the gauge reproduces Arm B, the result
  is a reordering artifact → **NULL**.
- **g-close (does replay close the online→offline gap):** Arm B closes a **pre-set fraction f** (FROZEN
  grill Q5: **≥ 0.5**) of the Arm-A → offline-partition-ceiling gap. (A PASS that also *reaches* the
  offline ceiling would say replay fully substitutes for the global pass.)
- **g4 no-collapse:** d_eff_ratio ≥ 0.5 for the Arm-B codes.
- **calibration valid:** the raw-SPPMI-SVD anchor hits +0.109 / 0.222 (INVALID otherwise).

**NOT g3** (corr(cooc,drift)<0.15 is dead at n=40 for static reads — even the +0.109 SVD anchor fails it;
the label-shuffle B-KILL is the pair-specificity headline instead, per Report 126).

- **Trajectory drill-down (DIAGNOSTIC, not a pass criterion — grill Q5).** The binary B-KILL says "pair-
  specific *yet*? y/n" but cannot tell a *coarse-to-fine first step* (clumpiness now → pair-specificity
  later) from a *ceiling* (clumpiness is terminal). So log the pair-specific signal (real − pairing-
  scrambled) at **each replay epoch**, for **both** arms. Interpretation: **rising in B relative to A**
  over epochs = the coarse "royal clumpiness" is the **first rung of a real climb** → justifies the ONE
  pre-registered escalation (the heavier A100 / longer-budget run, Q4); **flat-at-zero or declining while
  clumpiness saturates** = the ceiling/contraction case (the bound). This is a diagnostic that informs
  escalation, **never** a pass-substitute (anti-treadmill). *(126 found the clumpiness was "para-set
  hubness" — a degree artifact, not obviously a first step; the trajectory + the scramble test together
  separate "coarse semantic category climbing" from "hubness stuck.")*

All hyperparameters (k-WTA `cap`, `k` ∈ {8,16,32} matched to Oracle E, learning rate, replay epochs, the
surprise-residual definition, the dissimilarity-spacing rule, Δ_AB, f, the ≥4/5-seed bar, the SimLex sha)
are **FROZEN before the run — NONE swept-to-pass.** If parallelized across priority/spacing variants, it
**MUST** be a pre-registered controlled set with **family-wise correction + one shared adjudication** —
never "iterate over reorder schemes and accept whichever shows a signal" (the false-positive generator
the audit's two caught false positives warn against).

## 4. Disposition (frozen) — decisive either way

- **SCREEN-PASS** (Arm B clears g-interaction ∧ g-gauge ∧ g-close, anchor valid): the replay-interleaving
  signal is **reachable by the faithful local dynamic in the idealized read** — emergent pattern-separated
  replay manufactures paradigmatic structure the as-is stream + the same writer cannot, and the emergent
  priority (not a hand-set schedule) is operative. **This is a screen-pass, NOT a graduation** — it
  *licenses escalation up the fidelity ladder (§4.5)*, and only the full-system rung (3) may claim
  "replay-as-structure-generator is real." It makes **Codebook-growth ⇄ Replay** the live Phase-4
  *candidate* → escalate per §4.5 (FHRR port → full integrated confirm), each rung behind the
  Abstraction build-gate (the user's to clear); write the emergent-priority dynamic's grounding + a
  faithful, carded precommit before the port.
- **NULL-BUT-RISING** (Arm B fails the pass gate *at the cheap screen's frozen budget*, BUT the trajectory
  drill-down shows the pair-specific signal **rising in B relative to A** over epochs — coarse clumpiness
  refining toward pair-specificity, grill Q5): **escalate ONCE** to the pre-registered heavier run (A100,
  WikiText + 2nd dataset, the frozen *larger* epoch budget, Q4). If that crosses the gate → SCREEN-PASS;
  if it still does not → bank the NULL **at power** (clumpiness *was* the ceiling, now confirmed not
  assumed). **ONE pre-registered escalation, then commit — not an open-ended "give it more" treadmill.**
- **NULL** (Arm B ≈ Arm A on the B-KILL **and the trajectory is flat/declining**, OR the random-reorder
  gauge reproduces it, OR Arm B fails to beat the static-surprise-reweight control): the
  **flat-code-from-small-text program is CLOSED** — the
  121-127 bound is banked as **capability-level AND replay-invariant**, and we learn the load-bearing
  thing: the **global/offline nature of the partition is irreducible** (you must see all the data at once;
  emergent local replay cannot substitute). → the next move is an **Abstraction-node build-gate decision**
  (a substrate/data change — the user's to make), **NOT another oracle**. *A null is a SIGNPOST, not a
  dead end:* this fork coincides with the project's north star (CONTEXT.md §2 sense 3, **Abstraction /
  "conceptualization"**) — a token-stream null doesn't doom the goal, it signposts the move toward the
  conceptualization substrate that learns from **richer-than-token** input (the king/queen *cosine* here
  is a token-altitude PROXY for the idea-altitude capability, NOT the goal itself). **Concretely, the
  indicated "substrate change" is a latent/HIERARCHICAL (layered) code** (Report 124's named escalation):
  layer 1 = the coarse "royalness" clump a flat code already reaches; layer 2 = the differentiating axis
  (king = royal-*man* vs queen = royal-*woman*); higher layers = more abstract factors — exactly what the
  one method that recovered the signal (Oracle E's 8–32-slot factorization) encodes, and whose LOCAL
  writer is audit short-list **#2**. *Not free:* the layering must be grown by a **LOCAL** rule with a
  **between-layer NONLINEARITY** (127) — stacked *linear* layers stay inside the bound. The layered
  *architecture* (layers, wiring, ops) is a **legal structural prior** — an evolution-style scaffold like
  the precommitted k-WTA cap or D, **NOT** a homunculus (grill, 2026-06-01: scaffold ≠ arbitration); the
  live constraints are only (i) the *learning* filling it must be **LOCAL** (Hebbian/competitive/
  equilibrium-prop/predictive-coding, Millidge 2022 — avoiding **global backprop**, the PM-6 tension), and
  (ii) the scaffold must stay **problem-GENERIC**, never hand-shaped to the king/queen answer (= a
  design-time homunculus). Open empirical Q: does local learning in a fixed generic scaffold actually
  *recover* the generalizing factorization (legal ≠ guaranteed).
- **INVALID:** the calibration anchor misses +0.109/0.222 → fix the harness, re-run.

*Convergence note:* this and the **Oracle-E TEM local writer** (short-list #2) are two local-writer
candidates for the same global positive (the nonlinear partition / the NMF factorization). A NULL on
**both** = the global optimization is irreducibly load-bearing → the bank-the-bound decision is firm.

## 4.5 Fidelity ladder + inter-rung agreement (a screen-pass is NOT a behavior claim)

**Rationale (the load-bearing methodological point).** A substrate-free oracle tests a *necessary*
condition — *is the signal reachable by the faithful local dynamic, in idealized numbers?* — **not the
behavior of the full FHRR + Hopfield system.** The asymmetry that makes it useful: a **NULL on a faithful
screen is a trustworthy kill** (the noisier real substrate, which can only do *less*, won't do better),
but a **PASS is only a feasibility result** and must be *confirmed by climbing the ladder*. **No rung's
pass may be reported as a higher rung's claim.** This section makes that escalation a frozen rule so a
screen-pass cannot be over-claimed as system behavior.

| Rung | What it runs | Fence | What a PASS here means |
|---|---|---|---|
| **0 — Planted smoke** | the loop + gauge on a planted king/queen corpus (§5) | substrate-free | plumbing + the gauge discriminates; **not verdict-bearing** |
| **1 — Faithful substrate-free SCREEN** (this precommit) | the nonlinear-partition **k-WTA** writer (the *same local dynamic the substrate runs*), Euclidean code-row reads, full control battery, anchor + `grow_G` floor + NMF ceiling reproduced in-harness | substrate-free (fence-clear) | the signal is **reachable by the faithful local dynamic** → license to escalate; **NOT graduation** |
| **2 — FHRR port** | re-run Arm A vs Arm B (+ gauge) on the **actual `TorchFHRR` vectors** (the Report-122 single-shot-port pattern), at a D where the anchor/floor stay resolvable; **D-sweep if it disagrees with rung 1** | **behind the build-gate** | the **real substrate expresses** the mechanism → license to integrate |
| **3 — Full integrated confirm** (graduation) | wire the emergent replay-interleaving into the **real consolidation/replay path** (actual replay buffer + `OnlineCodebookUpdater`), confirm contextual-completion end-to-end | **behind the build-gate** | **graduation-style claim licensed** (the 055-058 standard: headline + CI + control on the same test set, multi-seed) |

**Inter-rung agreement checks (the over-claim guards):**

- **Rung 1 internal validity:** the raw-SPPMI-SVD calibration anchor hits **+0.109 / 0.222**, AND the
  in-harness `grow_G` linear floor reproduces **+0.021** and the NMF ceiling **+0.19** — else the harness
  itself is off (INVALID), not the mechanism.
- **Rung 1 → Rung 2 = VERDICT-LEVEL agreement (NOT bit-identical).** Different number system + expected
  global contraction (cf. Report 122: FHRR-port **+0.101** vs SVD **+0.109** — same verdict, attenuated
  magnitude). The port PASSES iff it reproduces **(a)** the *sign* + the *≥4/5-seed B-KILL pair-specificity*,
  and **(b)** the Arm-B − Arm-A interaction margin *surviving within the port's known contraction factor*.
- **Rung 2 → Rung 3 = BIT-IDENTICAL / byte-consistent** through the integrated path (the Report-058
  e2e-wiring standard: *driving identical data through the integrated path reproduces the mechanism
  bit-identically*). A graduation claim requires this exact reproduction.

**Divergence protocol (your exact worry, made a frozen rule).** If a **higher rung NULLs where a lower
rung passed**, that divergence **is itself a finding to surface** (CONTEXT.md), **never** a quiet drop and
**never** a re-tune to force agreement. The prime suspect is the substrate's **1/√D crosstalk noise floor
(audit bound family #2)** *masking* a real mechanism — so the response is a **D-sweep / lower-D run to
separate mechanism from substrate artifact** (the 052 Pair-#2 lesson), not abandoning the mechanism.
Conversely, the opposite failure — a lower rung that passes only via an *over-powerful idealized read* (the
Report-124 SVD-contraction-artifact trap) — is caught **at rung 1 by the faithfulness requirement**: rung
1 uses the **k-WTA local writer, not a global SVD flashlight**, precisely so a screen-pass already reflects
the local dynamic the substrate would run.

**Reporting rule:** every STATUS/report line names the **rung**. "Reachable by the faithful dynamic"
(rung 1) is never written as "the system does it" (rung 3). The 121-127 nulls are already banked at rung
1 fidelity; a CE-1 rung-1 NULL banks the bound as replay-invariant at the same fidelity (a trustworthy
kill). A CE-1 *behavioral* claim requires rung 3.

## 5. Build checklist (`experiments/69_ce1_replay_nonlinear_partition.py` — proposed)

- [ ] Reuse via importlib: `exp61.{load_corpus,build_cooccurrence,build_sppmi,build_S,pick_k_by_density,
      d_eff,grow_G,row_center}`, `exp62.{_cos_real,_fcos,_boot_diff}`,
      `exp63.{select_pairs,raw_sppmi_svd_anchor}`, `exp65.{nmf_slots,faithful_read}` (linear floor + NMF
      ceiling), `exp68.{kwta_features,read_specificity,derange_partners,l2rows,random_nonneg}`.
- [ ] New code: `kwta_stream(windows, order, k, cap, lr, epochs, ...)` (online k-WTA consuming windows in
      a given replay order/sampling, W updated online); `settling_residual_priority(codes, windows, ...)`
      (the LOCAL surprise scalar per window against the current codes); `pattern_separated_order(priority,
      codes, windows, ...)` (priority-sampled + dissimilarity-spaced interleave); `random_reorder(priority,
      seed)` (the gauge); `static_surprise_reweight(build_S, priority)` + offline partition (the reweight
      control); `kmeans_onehot(build_S, k, seed)` (the competent offline ceiling, Lloyd, no sklearn).
- [ ] Calibration anchor EVERY run; INVALID if it misses +0.109/0.222.
- [ ] Arms wired: A (as-is), B (replay-interleaved), gauge (random-reorder), offline ceiling (batched
      k-WTA + k-means), static-reweight control, grow_G floor, NMF ceiling — all read by
      `exp68.read_specificity` with the label-shuffle B-KILL.
- [ ] Planted/small SMOKE: a planted king/queen corpus where pattern-separated replay SHOULD help (and
      the as-is order should not) — validates the loop + the gauge discriminates BEFORE the WikiText run.
- [ ] FREEZE before the run: `cap`, `k`∈{8,16,32}, lr, replay epochs, the settling-residual definition,
      the dissimilarity-spacing rule, **Δ_AB**, **f**, the ≥4/5-seed bar, the SimLex sha — all
      precommitted, NONE swept-to-pass. Substrate-free verdict; FHRR-port is the deferred Stage-1 (fence).

## 6. Open questions to resolve in the grill (before freeze)

1. ~~The surprise signal.~~ **RESOLVED (grill Q1, 2026-06-01):** priority = **novelty × surprise**,
   combined *multiplicatively* over a two-timescale loop — **novelty** = how thinly a window's tokens are
   represented in the system's *current* codes (under-consolidated; a state property, NOT raw corpus
   rarity), **surprise** = how poorly the current codes *resolve* the window (settling-residual). Replay
   what is *both* new-to-the-system AND unresolved. The two-timescale loop (current understanding → what's
   unresolved now → replay → updated understanding) is the load-bearing new ingredient; the
   **static-surprise-reweight control** isolates it from a one-shot novelty/PMI reweight (the 123 trap).
   *Whether the surprise term is read cheaply off the codes (rung 1) or needs the FHRR settle (rung 2) is
   the fence question — see §6.6 / grill Q4.*
2. ~~Δ_AB and f.~~ **RESOLVED (grill Q5, 2026-06-01):** headline = the **label-shuffle B-KILL**
   (pairing-scramble, magnitude-immune) must hold in **≥4/5 seeds** (the bar that caught the last two
   false wins); supporting rails, anchored between the measured floor (+0.021 / online-A ≈ +0.088) and the
   offline ceiling (≈ +0.208) so they're meaningful-but-hittable: **Δ_AB ≥ +0.05** (B beats dumb-order A)
   and **f ≥ 0.5** (B closes ≥ half the A→offline-ceiling gap), plus B beats the static-reweight control by
   **≥ +0.02** (proves the *loop*, not a one-shot reweight). Magnitudes are inflated → the **B-KILL, not
   the raw number, is the verdict-bearer**. PLUS the trajectory drill-down + the NULL-BUT-RISING escalation
   outcome (§3, §4) that your "is clumpiness the first step?" challenge added.
3. ~~Pattern separation operationalization.~~ **RESOLVED (grill Q2, 2026-06-01):** YES — spacing similar
   windows apart is a core *second* ingredient (the CLS interleaving that prevents king/queen smushing,
   [whats_missing.md Reframe A]). "Similar" is judged by the **system's own current codes** (do the
   window's tokens currently land in the same assembly?), compared **locally** against a sliding window of
   recently-replayed snippets — no global all-pairs pass, and it re-judges as the system learns. (Whether
   *spacing* vs *priority* is the load-bearing half = an optional later teardown, not a headline arm.)
4. ~~Online k-WTA recency bias / dataset & epoch budget.~~ **RESOLVED (grill Q4, 2026-06-01):**
   (a) **Dataset:** rung-1 screen on **WikiText-2** (signal present — the global flashlight found +0.109;
   comparability + the validity anchor). IF the screen justifies it, escalate to **Colab A100** with
   **WikiText + a SECOND dataset** (a *generality* check — the effect must not be WikiText-specific); the
   second corpus needs its **OWN calibration anchor** (the +0.109/0.222 band is WikiText-specific —
   re-derive the SVD floor/ceiling per corpus or the run is INVALID against the wrong yardstick). *Note:
   data-scale and substrate-fidelity are SEPARATE axes — a bigger-corpus run can still be a cheap rung-1
   substrate-free screen; it does not by itself need the build-gate.*
   (b) **Epochs:** a **generous, FROZEN** replay-epoch budget so the two-timescale loop can actually turn
   (CE-1's novelty IS the multi-pass loop; too few turns starves it). The **d_eff collapse-guard is the
   circuit-breaker** — more epochs of a local rule drives *contraction* toward the dominant collocational
   mode (the 122 trap: "20 epochs iterated into contraction"); if d_eff collapses, more epochs is moot and
   the run stops. Arm A vs Arm B are compared at the **SAME** epoch budget (so "more time" never favors
   B). The budget is frozen in advance — **NOT** extended-until-it-passes (no unfalsifiable "just give it
   more time").
5. **Smoke design.** What planted corpus makes the as-is/interleaved distinction sharp enough to validate
   the gauge discriminates (the planted-smoke discipline from exp61)?
6. **Fence check (resolved *structurally* by §4.5; the *choice* is still the grill's).** If the surprise
   priority can be read off the current k-WTA **code rows** (Euclidean, like 127) → CE-1 stays **rung 1**
   (substrate-free, fence-clear, runnable now). If it genuinely needs the **FHRR Hopfield settle** → that
   is simply **rung 2** (behind the build-gate), not a blocker — the ladder absorbs either answer. The
   grill decides *which surprise signal* (and therefore *which rung* CE-1 opens at).

## 7. Run log (rung-0 rehearsal — appended; does not alter the frozen spec)

- **2026-06-01 — Rung-0 planted-smoke rehearsal #1 (`experiments/69`, kind+trap, 3 seeds) → SMOKE-FAIL
  (the gate did its job).** Numbers: KIND offline +0.88 / linear-floor −0.013 / **Arm-A +0.74 ≈ Arm-B
  +0.74** / gauge +0.88 / static +0.88, B-KILL 3/3; TRAP **offline +0.85 / B-KILL 3/3** (should null).
  Three diagnoses, separating apparatus from the question:
  1. *(apparatus)* **negative control not null** — per-pair sentence *clumping* + global sliding windows
     + shared filler leak **locality** into within-window co-occurrence, so the TRAP's "disjoint contexts"
     aren't disjoint in practice → trap shows a spurious pair signal. Fix: **within-sentence co-occurrence**
     (no cross-sentence windows) so the only pair signal is the intended shared-vs-disjoint context.
  2. *(apparatus)* **para−rand is hubness-inflated** (the Report-126 trap) — `rand` not matched to target
     hubness. Fix: **B-KILL (within-target shuffle) is the headline** (hubness-immune by construction);
     `rand` = hubness-matched cross-pair target pairs.
  3. *(the question)* **no replay effect as-implemented** — the streaming writer *accumulated* co-occurrence
     monotonically, so the effective operator converged regardless of order (B ≈ A; more-streaming arms
     drifted to the offline ceiling). Fix: make replay change the **effective operator** persistently —
     a **per-epoch evolving multiplicity reweight** (priority recomputed from current codes), and a
     **headroom corpus** where the paradigmatic pairs are RARE and buried under dominant collocational
     blobs (uniform consolidation misses them; only reweighting — or a subdominant-mode method like NMF —
     surfaces them). Ceiling-of-record for "signal exists" = **NMF** (subdominant modes), not k-WTA-on-
     uniform (which, like Arm A, is dominated by the collocational mode — *that's the headroom*).
  **Discipline:** these are apparatus/mechanism-shape fixes (debugging the test), NOT tuning toward a pass.
  `mult_scale`/epochs are frozen-by-principle; the result of rehearsal #2 is reported either way. Honest
  status: **the mechanism is NOT shown to work; rehearsal #1's read was "replay ≈ as-is," uninterpretable
  until the test is valid.** Latent deeper hypothesis (would be a real, deflationary-for-CE-1 finding): for
  a nonlinear-partition writer, the *partition* may do the work and replay-order may be a weak lever.
- **2026-06-01 — Rung-0 rehearsal #2 (`experiments/69`, redesigned: within-sentence cooc, common-mode
  headroom corpus, hubness-immune B-KILL headline, per-epoch evolving-multiplicity reweight) → APPARATUS
  FIXED, but SMOKE-FAIL on "Arm B beats Arm A" (NO HEADROOM).** KIND: NMF ceiling **+0.95**, linear floor
  **−0.013** (brackets like the real bound), but **Arm-A as-is +0.96 ≈ Arm-B +0.96 ≈ NMF** — the online
  k-WTA *already* reaches the planted signal as-is, so replay adds nothing. TRAP: NMF **+0.003**, B-KILL
  **negative** → the negative control is now VALID (the rehearsal-#1 locality leak is fixed). **Finding:**
  this **converges with the PM-7b prototype + rehearsal #1** → for a nonlinear-PARTITION writer, the
  *partition* does the work and **replay-reweight is a weak lever** (the deflationary-for-CE-1 hypothesis,
  now 3 independent probes). **Structural limit (charter-relevant):** a planted toy CANNOT pose "does
  replay beat as-is" without **hand-shaping the corpus to the answer** = a *design-time homunculus*
  (CONTEXT.md §3, the clause added today) — because SPPMI frequency-normalization un-buries any
  NMF-recoverable planted signal, making it reachable by the as-is partition. So the decisive, un-rigged
  headroom test is **real WikiText** (rung-1, no gate): does online-as-is k-WTA already reach 127's +0.25,
  or fall short (the PM-7b hint) — and does replay close any gap? Discipline held: NOT tuned to pass;
  apparatus fixes applied, result reported as-is. **NEXT = a strategic fork (the user's): WikiText screen
  vs re-rank toward the Oracle-E TEM local writer (short-list #2, the partition-itself lever).**
- **2026-06-01 — exp70 WikiText partition head-to-head (rung-1, anchor-valid +0.1092/0.222) → STRONG
  SCREEN POSITIVE (NOT banked; adversarial verification pending).** On real WikiText-2 (V=2002, 40 SimLex
  pairs, hubness-immune B-KILL headline): a **LOCAL, ONLINE, BOUNDED-MEMORY, LEARNED k-WTA partition
  reaches +0.26 B-KILL (3/3 seeds lo>0), matching the offline-global k-WTA ceiling (+0.27, ratio 0.94)**,
  where the LINEAR local read (grow_G) gets **≈0**. Survives the **frozen-random no-learning control**
  (+0.26 vs +0.02 → learning-driven, not partition-inflation). The **converged** accumulating writer
  (1.03×) confirmed my first-pass "global is load-bearing" was a **STEP-COUNT confound**, not locality.
  → **OVERTURNS the PM-7b prototype's online-local downgrade** (naive/3-seed; proper bounded-streaming
  matches global). Magnitudes are partition-inflated — **NOT** "beats SVD/NMF"; the apples-to-apples is
  vs offline-kwta. **RUNG-1 SCREEN, 3 seeds — NOT graduation.** NEXT (before banking): **adversarial
  verification** (the 127 6-agent discipline) — proper multi-restart k-means (here k-means +0.125 < k-WTA
  +0.27, inconsistent with 127's "k-means reproduces k-WTA" — must resolve), single-pass locality, n≥10
  multi-seed, code-leak audit, B-KILL breadth. **Replay stays parked** (it'd be the gap-closer only if a
  locality cost emerges; the screen shows none — strengthens "the PARTITION is the lever").
- **2026-06-01 — Adversarial verification of the exp70 positive (7-agent workflow) → NARROWED, NOT
  refuted. The discipline caught a THIRD overclaim before banking (after 126/127).** All 6 skeptics
  returned NARROWED; no code leak. **SURVIVES (clean):** the LEARNED bounded-input k-WTA beats its
  frozen-random control on the IDENTICAL representation (+0.258 vs +0.019, no leak — verified line-by-line)
  and the input Π is genuinely recency-bounded (24.8× newest/oldest at read-time; never equals the full
  operator). Learning is load-bearing; input-locality is real; magnitude-inflation scoping ("NOT beats
  SVD/NMF") honestly held; anti-homunculus PASSES (fixed k-WTA cap + local Hebbian = legal scaffold).
  **DOES NOT survive (RETRACTED framing):** (1) **CROSS-OPERATOR defect** — exp70's partition arms run on
  `l2rows(sppmi)` (1st-order) while the grow_G floor AND NMF ceiling run on `build_S` (2nd-order), so
  "past the LINEAR bound / where grow_G gets ≈0" is NOT apples-to-apples (violates this precommit §2's
  same-operator invariant). (2) **Anomalous ceiling** — offline k-means B-KILL CI-lo = −0.067 (FAILS),
  contradicting 127's "k-means reproduces k-WTA"; the ceiling silently became same-dynamic offline-kwta →
  re-opens, not confirms, 127. (3) **Normalization mismatch** (online cooc-degree vs offline
  window-presence marginals, per-row cos ~0.73) → ratio 0.94 not strictly same-representation. (4)
  **Breadth UNVERIFIED** (none of 127's per-pair / drop-top-K / permutation-co-membership battery). (5)
  **"3/3 seeds"** is a within-seed fixed-boot_seed pair-bootstrap at a relaxed 2/3 gate — NOT across-seed,
  below the n≥10 / 4-5-of-5 bar; per-seed values not even persisted. (6) **Decay knob near-inert**
  (bounded 0.94 vs converged 1.03) — signal lives in W (20 passes), so "bounded-memory" is fair about Π
  but the mechanism is many-pass; single-pass control pending. **NOT licensed to overturn PM-7b.**
  **Mandatory controls before banking (ranked):** ① SAME-OPERATOR re-run (all arms on `build_S`; k-means
  must re-converge to k-WTA = 127 replication, AND online_bounded/offline ratio survive); ② same-
  representation ceiling (cooc-degree-marginal offline arm); ③ n=10 + independent bootstrap seeds + persist
  per-seed + paired across-seed CI of (bounded−frozen)>0 + ≥8/10 bar; ④ single-pass (epochs=1)
  bounded-vs-converged gap; ⑤ 127 breadth battery (per-pair, drop-top-10, permutation-p co-membership);
  ⑥ multi-restart k-means. **Banked NOW = only the narrow "learning matters + input-bounded" claim.**
- **2026-06-01 — exp71 same-operator/representation diagnostic (settles verification red flags #1/#2) →
  RESOLVED.** On the CORRECT 2nd-order operator **build_S**: k-means (+0.192, lo>0) ≈ k-WTA (+0.227,
  lo>0), **gap +0.035 → REPRODUCES 127's "k-means reproduces k-WTA"**; the linear grow_G read on build_S
  is the ≈0 floor; k-WTA breadth 22-23/40 pairs positive, drop-top10 stays >0. exp70's anomalous NEGATIVE
  k-means (lo −0.067) was a **1st-ORDER-REPRESENTATION artifact** (on L1=l2rows(sppmi): k-WTA +0.274 ≫
  k-means +0.108, gap +0.165; k-means fails the B-KILL on 1st-order profiles, k-WTA stays broad 28/40).
  **NET:** (a) the harness is consistent with 127 on the right operator; (b) exp70's headline ran on the
  **WRONG operator** → the "past the linear bound / matches global ceiling" framing is an artifact,
  **RETRACTED**; (c) what is SOLID = **127 replicated at the B-KILL + breadth level on build_S** — a real
  consolidation, **NOT a new finding**; (d) the genuinely-NEW **LOCALITY** claim (online-local-bounded
  reaching the build_S ceiling AND beating frozen-random, *on build_S*) is **STILL UNTESTED** (exp70 ran
  online on L1). **NEXT decisive control = online-on-build_S; then n=10 + single-pass + permutation-breadth
  (verification battery ①b/③/④/⑤).** The discipline caught a 3rd overclaim before banking — the session's
  durable win is the methodology, not a graduation.
- **2026-06-01 — exp72 DECISIVE control: online writer re-run on the CORRECT operator build_S → LOCALITY
  COST (exp70's "locality ~free" was a 1st-order-representation artifact, now CONFIRMED retracted).** On
  build_S (anchor-valid +0.1092/0.222): offline ceiling k-WTA +0.227 ≈ k-means +0.192 (127 replication
  holds), grow_G floor ≈0. **online_CONVERGED** (decay 1.0, accumulating → effectively sees the full
  operator) reaches the ceiling (+0.228, ratio **1.01**). But the genuinely **BOUNDED-memory** writer
  (decay 0.7, recency-limited, never holds the full operator) **FALLS SHORT: +0.173, ratio 0.76, B-KILL
  CI-lo NEGATIVE in 0/3** (doesn't clear significance). It beats frozen-random (+0.072, learning_matters
  =TRUE) but not enough. converged vs bounded differ ONLY in decay → the gap IS the locality cost (clean:
  same structure/steps/init). **DECISIVE: on the correct operator the GLOBAL/accumulated input is
  LOAD-BEARING; a genuinely-local bounded-memory writer does NOT reach the nonlinear partition.** →
  **REPLAY RE-ENTERS as the gap-closer with a MEASURED gap (bounded 0.76 → global 1.0):** the CE-1
  question sharpens to "can an emergent replay schedule make a bounded-memory writer close the 0.76→1.0
  gap on build_S?" — now well-motivated (the user's "replay will matter" instinct, vindicated by a number).
  Oracle-E TEM local writer = the alternative gap-closer. **NET ARC:** partition-on-build_S = 127
  (consolidated, NOT new); locality NOT free; replay re-motivated by a measured gap. 3 seeds (n=10 to bank
  the locality-cost finding).
- **2026-06-01 — exp72 n=10 BANKING run → LOCALITY COST BANKED at the n≥10 bar (CE-1 arc CLOSED).**
  build_S, anchor-valid, 10 seeds, INDEPENDENT bootstrap seeds, per-seed persisted, ACROSS-seed CIs:
  **converged−bounded [+0.042, +0.055] (lo>0 → locality cost ROBUST across seeds)**; **bounded−frozen lo
  +0.104 (learning matters ROBUST)**; bounded across-seed B-KILL CI [+0.174, +0.184] vs ceiling ~+0.224
  (ratio 0.80, bounded_reaches_ceiling=FALSE, within-seed lo>0 in only 1/10); converged matches the
  ceiling (1.01). **DECISIVE + BANKED: on the correct operator the genuinely bounded-memory LOCAL writer
  falls short of the global/accumulated writer → the GLOBAL pass is LOAD-BEARING; locality is NOT free.**
  exp70's "locality ~free" RETRACTED at n=10 (1st-order-representation artifact). **CE-1 arc CLOSED:**
  (i) 127 replicated on build_S (a consolidation, not new); (ii) locality-cost banked (n=10); (iii) replay
  / Oracle-E TEM re-enter as the gap-closers with a MEASURED gap (bounded +0.179 → ceiling/global
  +0.224–0.228). **NEXT (next session) = the gap-closer:** the emergent-replay-schedule experiment (does a
  local replay-priority + pattern-separation schedule let a bounded-memory writer close +0.179→+0.224 on
  build_S?) OR the Oracle-E TEM local writer (short-list #2). The remaining verification controls
  (single-pass, 127 breadth battery) are now OPTIONAL — the headline is a locality-COST, not a positive
  needing defense.
