# Report 128 — CE-1 emergent-replay gap-closer on `build_S` → a LOCAL pair-reweight closes the locality gap (real, substrate-confirmed, domain-general), but the operative signal is INVERSE-FREQUENCY (≈ PMI) — NOT codes-derived surprise, NOT the two-timescale loop (both deflated by controls)

*(rung-1 NULL-BUT-RISING → rung-2 FHRR port PASS (§6) + Phase-B generality (§7) → the §8 frequency control DEFLATES the mechanism. The distinctive CE-1 hypothesis is not supported; the gap-close is a frequency/PMI effect. Full arc resolved — §9.)*

**Status:** Phase-3 / capability **Codebook-growth ⇄ Replay** (a COMBINATION, CONTEXT.md §4 DAG). A
substrate-free **rung-1 DRILL-DOWN oracle** (NOT graduation; precommit §0/§4.5). Frozen, grill-hardened
pre-commit (with the user-approved §3.5 gate re-anchoring):
[phase-3-ce1x127-replay-nonlinear-partition-precommit.md](../../notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md).
Harness: `experiments/73_ce1_emergent_replay_gapcloser.py`. 10 seeds, WikiText-2 V=2002, W=6, n=40
SimLex≥5 non-cooc pairs, operator = `build_S` (2nd-order SPPMI). Floor (055-058) untouched.

**RESOLVED verdict (after the §6 escalation):** an emergent, LOCAL, surprise-*targeted* **pair-replay** is
a **real, substrate-confirmed local gap-closer** — it closes the +0.045 locality gap on `build_S` (rung-1,
10/10, gauge-discriminated) AND the gap-close **survives the rung-2 FHRR port** at D≥1024 (verdict-level
agreement with rung-1; the 1/√D crosstalk floor only bites at D=512) → it is **not an idealized-Euclidean
artifact**. The **operative lever is a ONE-SHOT surprise-targeted reweight**, NOT the two-timescale loop:
`g-static` fails at n=10 (20 epochs) AND still fails at the heavier 40-epoch budget (B even dips), so the
loop is **inessential** (its only edge is stability/anti-contraction, not a mean shift). Honest deflation:
a one-shot surprise-weighted reweight is, per §1's surprise≈PMI guard, close to a PMI-style reweight —
so the banked claim is the **modest** one (a local surprise-targeted reweight recovers what the recency-
bounded writer drops; magnitudes partition-inflated, NOT "beats SVD/NMF"). **First time a LOCAL mechanism
closes a measured gap to the global pass on the paradigmatic axis, confirmed real on the substrate.**
**Generality (§7): the gap-close GENERALIZES to a different domain** (TinyStories narrative fiction, ~3×
stronger) and **one-shot-suffices generalizes cleanly** — with one honest scope caveat: the anti-homunculus
*gauge* (random replay must be inert) is fully discriminating only on **sparse-signal** corpora (WikiText);
on a **paradigmatically-saturated** corpus (TinyStories) random replay also helps, though **targeting still
wins ~3:1**. So "targeting beats random" + "gap closes" hold across domains; "*only* targeting matters" is
corpus-dependent. **DEFLATION (§8, the frequency control): the operative targeting signal is INVERSE
CO-OCCURRENCE FREQUENCY (≈ PMI), not the codes-derived "surprise."** A codes-independent `freq` arm closes
the gap as well as B (B−freq CI [−0.009, +0.012]; freq−A CI [+0.034, +0.052]) → the surprise/novelty/loop/
pattern-separation machinery is ALL inessential; the gap-close is "up-weight the rare pairs the recency-bias
drops" ≈ re-applying SPPMI's PMI. **So the honest banked claim is: the gap-close is real, substrate-confirmed,
and domain-general, but its MECHANISM is a mundane frequency/PMI reweight — the distinctive "emergent surprise
replay" hypothesis is NOT supported (§9).**

---
**Rung-1 verdict (NULL-BUT-RISING, n=10) — what the screen alone found:** Two findings, separated honestly:
**(1) a clean POSITIVE** — an emergent, LOCAL, surprise-*targeted* replay schedule **closes the measured
locality gap** on `build_S`: Arm B = +0.225 vs Arm A = +0.179 (closes **102%** of the +0.045 gap, reaching
the offline ceiling +0.224), B-KILL lo>0 in **10/10** seeds, robust over baseline ((B−A) CI [+0.036,+0.056]),
and the **random-reorder gauge does NOT** close it (gauge +0.170 ≈ A; (gauge−A) CI [−0.017,−0.002]). So it
is the *which-pairs targeting* (not replay mass) that recovers the structure the recency-bounded writer
otherwise loses — the first time a **local** mechanism has closed a measured gap to the global pass on the
paradigmatic axis. **(2) the CE-1-distinctive claim is NOT established** — Arm B does **not** robustly beat
the **static one-shot surprise-reweight** ((B−static) CI **[−0.010,+0.055]**, mean +0.021): B is better on
average and far more *stable*, but the frozen `g-static` (the precommit §1 surprise≈PMI guard: "B must beat
both A *and* static, or it's a reweighting in disguise") **fails at n=10**. The loop's benefit is *stability*
(self-correcting away from the init-dependent epoch-1 priority) and *anti-contraction* (the trajectory shows
A decays 0.215→0.179 over epochs while B holds 0.215→0.225, B−A rising 0.000→+0.046), **not a higher mean**.
Per the frozen disposition (§4), trajectory-rising → **ONE pre-registered escalation** (heavier/longer
budget + 2nd dataset), then commit. **NOT a graduation; NOT banked as a two-timescale-loop pass.**

## 0. Preamble (CLAUDE.md experiment preamble)

> **Active capability:** Codebook-growth ⇄ Replay (a COMBINATION) — CONTEXT.md §4.
> **Headline metric per [combination-experiments.md:52-54]:** within-set **label-shuffle B-KILL**
> pair-specific residual of the INTERACTION (Arm B replay-interleaved vs Arm A as-is + the emergent
> priority operative / gauge inert). DRILL-DOWN oracle, rung-1 of the fidelity ladder — NOT graduation.
> **Required controls:** random-reorder gauge; competent offline partition (k-means + batched k-WTA on
> `build_S`); static-surprise-reweight; linear `grow_G` floor; global NMF ceiling; raw-SPPMI-SVD
> calibration anchor (+0.1092 / kq 0.222, INVALID otherwise); 10 seeds; independent bootstrap seeds.
> **Last verified result:** Report 127 (nonlinear partition reaches paradigmatic structure, not
> competition-specific) + exp72 §7 banking (on `build_S`, the bounded-memory LOCAL writer = +0.179 falls
> short of the global ceiling +0.224 → **locality is NOT free; the global pass is load-bearing**).
> **Why now:** exp72 turned the locality cost into a **measured target** — can an emergent, LOCAL replay
> schedule make the bounded-memory writer close +0.179 → +0.224 on `build_S` without becoming the global
> pass in disguise? This is the audit's #1 short-list lead (the one place the 121-127 wall could be a
> false negative of isolation-testing).

## 1. The mechanism (faithful, anti-homunculus — precommit §1)

The bounded writer's operator is `sppmi(S_decayed)` — a **recency-biased** co-occurrence (decay forgets
the early corpus). Because SPPMI cancels per-**row** frequency reweights (rehearsal #1's lesson), replay
must re-present specific **pairs** (episodes), not tokens. **Arm B re-adds OBSERVED co-occurrence pairs**,
weighted by a LOCAL per-pair priority recomputed **each epoch** against the **current codes**:

- `priority(i,j) = novelty(i)·novelty(j) · surprise(i,j)` (multiplicative — grill Q1).
- `novelty(t)` = code-spread `(1 − peak assembly fraction)` × **assembly-balance** (pattern-separation,
  grill Q2: down-weight tokens in crowded dominant assemblies = dissimilarity-spacing).
- `surprise(i,j) = 1 − cos(code_i, code_j)` (settling-residual proxy: the partition has NOT yet bound i,j).

**Anti-homunculus check:** no supervisor reads "is this king/queen?"; no if-X-then-Y branch; the boost is
a smooth function of a local scalar over **observed pairs only** (no hallucinated episodes). Two-timescale:
immature codes → broad boost; mature codes → boost concentrates on the still-unresolved (paradigmatic-
context) pairs. `replay_strength = 1.0` FROZEN by principle (one unit of replay mass per unit observed
mass, scaled by max-normalized priority), NOT swept.

## 2. Arms (everything else held fixed = exp72 defaults; operator = `build_S`)

| Arm | What it is | Role |
|---|---|---|
| **A — online_bounded** | `exp72.online_build_s(decay 0.7)` — bit-exact banked path | the +0.179 baseline |
| **B — replay(priority)** | emergent codes-derived pair boost, recomputed each epoch | the bet |
| **gauge — replay(random)** | same boost mass/distribution, pair-assignment SHUFFLED | anti-homunculus tripwire |
| **static — replay(one-shot)** | boost frozen at first computed priority | isolates the two-timescale LOOP |
| **ceiling** | offline batched k-WTA + multi-restart k-means on `build_S` | 127 replication, ~+0.224 |
| **converged / frozen** | `online_build_s(decay 1.0)` / `(decay 0.7, learn=False)` | accumulating / no-learning controls |
| **floor / nmf / anchor** | `grow_G` / NMF / raw-SPPMI-SVD | +0.021 / +0.19 / +0.109 (calibration) |

*Fast path:* the base co-occurrence stream is seed- and arm-independent (`Pi_E = decay^m·Pi_{E-1} + S`);
the corpus is streamed once and the `mode="none"` writer reproduces `exp72.online_build_s` to <1e-9
(asserted in `--smoke`), so the per-epoch trajectory is trustworthy.

## 3. Re-anchored gate (precommit §3.5, user-approved 2026-06-01 — across-seed CIs)

§3's absolute rails (`Δ_AB ≥ +0.05`, `beat-static ≥ +0.02`) were frozen against a *believed* +0.120 gap;
exp72 measured the real gap as **+0.045** (2.7× smaller), so the absolute rails were superseded by
across-seed-CI forms anchored to the measured gap (the B-KILL headline + gauge tripwire UNCHANGED). PASS
iff ALL of: **g-headline** (Arm B B-KILL CI-lo>0 ≥4/5 seeds) · **g-close** ((B−A)/(ceiling−A) ≥ 0.5) ·
**g-dab** ((B−A) across-seed CI-lo > 0) · **g-static** ((B−static) across-seed CI-lo > 0) · **g-gauge**
((gauge−A) across-seed CI-lo NOT > 0) · **g4** (d_eff(B)/d_eff(A) ≥ 0.5) · calibration valid.

## 4. Results (10 seeds; anchor +0.1092 / kq 0.222, calib_ok ✓)

Data: `reports/_exp73_emergent_replay_gapcloser.json` (gitignored); log: `reports/_exp73_run.log`.

**Across-seed B-KILL means (hubness-immune within-set label-shuffle, operator = `build_S`):**

| arm | B-KILL mean | role / read |
|---|---|---|
| floor — `grow_G` (linear local) | **+0.0002** | the linear bound (~0) reproduced ✓ |
| online_frozen (no-learning) | +0.059 | learning matters: A−frozen ≈ +0.12 |
| gauge — replay(random) | +0.170 | **≈ A** — random reorder does NOT close the gap |
| kmeans_multi (offline) | +0.170 | noisy across seeds (+0.05…+0.275) |
| **A — online_bounded (decay 0.7)** | **+0.179** | **bit-exact to the exp72 banked baseline** |
| nmf (global subdominant) | +0.118 | global ceiling-of-record (low here at k=32) |
| static — replay(one-shot) | +0.204 | one-shot surprise reweight; **high variance** (+0.091…+0.278) |
| **B — replay(priority, two-timescale)** | **+0.225** | **closes 102% of the gap; stable (+0.191…+0.250)** |
| offline_kwta (ceiling) | +0.224 | the 127 build_S partition ceiling |
| online_converged (decay 1.0) | +0.228 | accumulating/global online |

**Gap arithmetic:** ceiling +0.224, Arm A +0.179 → **gap +0.045** (reproduces exp72's measured gap exactly);
B closes **(0.225−0.179)/0.045 = 102%**.

**Across-seed bootstrap CIs (the re-anchored §3.5 rails):**
- `(B − A)` = **[+0.036, +0.056]** (lo>0 → **g-dab PASS**; replay robustly beats the bounded baseline)
- `(gauge − A)` = **[−0.017, −0.002]** (lo NOT>0 → **g-gauge PASS**; random reorder does NOT close it →
  the *targeting* is the lever, not the replay mass — the anti-homunculus tripwire fires correctly)
- `(B − static)` = **[−0.010, +0.055]**, mean +0.021 (lo NOT>0 → **g-static FAIL**; B beats the one-shot
  reweight on average + is far more stable, but not significantly in the across-seed mean)
- d_eff(B)/d_eff(A) = **1.50** (→ **g4 PASS**, no collapse)

**Gate:** g-headline ✅ (10/10 ≥ 8) · g-close ✅ (102%) · g-dab ✅ · g-gauge ✅ · g4 ✅ · **g-static ❌**.
6 of 7. The single failure is the criterion that the precommit §1 surprise≈PMI guard makes load-bearing.

**Trajectory drill-down (per-epoch point B-KILL, mean over seeds):** Arm A **contracts** monotonically over
the 20 epochs (0.215 → 0.179 — the 122 "iterated-into-contraction" signature on the recency-bounded
operator); Arm B **holds/rises** (0.215 → 0.225). `B−A` rises 0.000 → +0.046 (`rising = True`). The
gauge contracts like A. Reading: emergent surprise-targeted replay acts as an **anti-contraction force** —
re-injecting under-resolved pairs each epoch stops the bounded writer decaying toward the recency bound.

## 5. Verdict & disposition (rung-1; NOT graduation)

**NULL-BUT-RISING.** The frozen gate fails on `g-static` → this is **not** a screen-pass: the
CE-1-distinctive **two-timescale LOOP** is not shown to beat a **one-shot surprise-reweight** at n=10, so
the load-bearing new claim ("the two-timescale interaction, not a single reweighting") is unproven. But two
things are real and survive: **(a)** an emergent, LOCAL, surprise-*targeted* replay **closes the measured
+0.045 locality gap to the global ceiling** (10/10, robust over A), and **(b)** the **gauge discriminates**
(random reorder fails) → it is the *which-pairs* signal, not replay mass — the anti-homunculus check passes.
What the loop *adds* over the one-shot is **stability + anti-contraction**, visible in the trajectory but not
in the across-seed mean.

**Mechanistic summary:** the bounded writer's operator is recency-biased; both a one-shot and an evolving
surprise-weighted **pair** re-presentation up-weight the under-resolved (paradigmatic-context) pairs that the
recency bias drops, recovering the global structure. SPPMI cancels per-*row* reweights, so this is genuinely
a *pair/episode* lever (not a frequency trick). The k-WTA codes mature within ~1 epoch, so the epoch-1
priority (static) already captures most of the targeting → the loop's marginal value is small-but-positive
(stability), not a mean shift.

**Disposition (frozen §4, NULL-BUT-RISING path):** trajectory-rising licenses **ONE pre-registered
escalation** — the heavier/longer-budget run (more epochs → more A-contraction → larger loop advantage; a
2nd dataset with its OWN calibration anchor → generality), Q4. If it crosses `g-static` → SCREEN-PASS; if
not → bank the NULL at power (the one-shot reweight is the operative lever; the loop is inessential). **This
is the user's call** (it is heavier compute, likely Colab/A100, and a one-shot is the alternative to bank).
*No open-ended treadmill — one escalation, then commit.* No adversarial-verification workflow is run here
because the verdict is NULL-leaning (no positive to defend); the gate itself caught the g-static shortfall.

**Banked side-findings (don't relitigate):**
- A **local** mechanism CAN close a measured gap to the global pass on the paradigmatic axis (first time) —
  via surprise-targeted **pair** replay; the gauge proves it's targeting, not mass.
- The two-timescale **loop** is NOT (yet) shown to beat a **one-shot** surprise-reweight (n=10) — the loop's
  value is stability/anti-contraction, not mean.
- The bounded writer **contracts** over epochs on the recency-biased operator (0.215→0.179); replay resists it.
- Magnitudes remain partition-inflated — NOT "beats SVD/NMF" (B/static exceeding the offline ceiling is
  reweight-up-weighting of rare pairs, a known inflation, not a structural claim over the global methods).

## 6. Escalation (Phase A) — rung-2 FHRR port + heavier budget (`experiments/74`, n=10)

Pre-registered (frozen) in [precommit §8](../../notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md)
BEFORE running (user-approved "escalate the working mechanism", build-gate pressure off). Data:
`reports/_exp74_escalation_fhrr_port.json`; log: `reports/_exp74_run.log`. Anchor +0.1092/0.222 ✓.

**(A) Rung-2 FHRR single-shot port → PASS (the substrate-reality headline).** Each arm's final 1st-order
SPPMI profiles were bundled into real FHRR hypervectors (`G_i = normalize(Σ_c sppmi[i,c]·E_c)`, E = random
FHRR atoms); `build_S` re-emerges as the substrate's FHRR-cosine Gram + O(1/√D); the SAME k-WTA partition
re-read. **B robustly beats A through the port:**

| D | ported B-KILL | ported A | (B−A) across-seed CI | B-KILL lo>0 |
|---|---|---|---|---|
| ∞ (exact, rung-1 anchor) | +0.226 | +0.191 | [+0.016, +0.051] | 10/10 |
| 4096 | **+0.227** | +0.204 | **[+0.011, +0.036]** | **10/10** |
| 2048 | +0.205 | +0.167 | [+0.020, +0.055] | 8/10 |
| 1024 | +0.208 | +0.171 | [+0.013, +0.058] | 10/10 |
| 512 | +0.173 | +0.153 | [−0.013, +0.050] | 8/10 |

At D=4096 the ported B-KILL (+0.227) ≈ the exact (+0.226) — **near-zero degradation; verdict-level agreement
(§4.5 rung-1→2) holds.** (B−A) CI-lo > 0 at D ∈ {1024, 2048, 4096}; only at **D=512** does the CI straddle 0
— the **1/√D crosstalk floor** (audit bound family #2) starting to mask the signal. Per the §4.5 divergence
protocol this is a *finding* (degrade-at-low-D), **not** a mechanism kill: the substrate genuinely expresses
the gap-close at sufficient D. → **the rung-1 positive is real on the substrate, not an idealized-Euclidean
artifact.** *Side-note (not load-bearing):* the ported global **ceiling** drops below ported-A at D=4096
(+0.197 < +0.204) — FHRR bundling differentially degrades the **denser** full-corpus operator (more contexts
bundled → more crosstalk); it does not affect the same-density B−A comparison.

**(B) Heavier budget (40 epochs) g-static re-test → STILL FAILS.** A=+0.179, B=+0.213, static=+0.203,
gauge=+0.167. **(B−static) CI [−0.023, +0.044]** (straddles 0 → `g-static` fails again); B at 40ep (+0.213)
is ≤ B at 20ep (+0.225), so **more epochs do NOT rescue the loop**. (B still robustly beats A: [+0.022,
+0.047]; gauge still inert.) → the **two-timescale loop is genuinely inessential**; the one-shot
surprise-targeted reweight is the operative lever.

**Escalation disposition (frozen §8): rung-2 PASS + g-static still fails → bank the ONE-SHOT
surprise-targeted pair-replay as a real, substrate-expressible LOCAL gap-closer (the loop is inessential).**

## 7. Phase B — generality on a 2nd domain (`experiments/73` on TinyStories, `experiments/75` probe, n=10)

Per §8 Phase B (user-chosen). The corpus PROBE (`experiments/75`) screened genuinely-different domains for
whether they INDEPENDENTLY carry the paradigmatic signal (PTB = dead, HF loading-script removed; ag_news =
VALID but weak own-anchor +0.062; **TinyStories = VALID, strong own-anchor +0.234 / kq 0.350, 49 SimLex
pairs > WikiText's 40**). Ran the rung-1 gate on **TinyStories** (narrative-fiction domain vs WikiText's
encyclopedic; its OWN calibration band; 1.18M windows; `--fast-arms` = the proven-equivalent fast path).

| arm | TinyStories B-KILL | WikiText (ref) |
|---|---|---|
| floor (grow_G) | +0.001 | +0.0002 |
| A (bounded) | +0.231 | +0.179 |
| **B (targeted replay)** | **+0.395** | +0.225 |
| gauge (random replay) | +0.289 | +0.170 |
| static (one-shot) | +0.379 | +0.204 |
| ceiling (offline kwta) | +0.262 | +0.224 |

**What GENERALIZES (the core finding, confirmed + strengthened):**
- **The gap-close generalizes** — `(B−A)` across-seed CI **[+0.134, +0.194]**, B-KILL lo>0 **10/10**; B
  closes 527% of the (smaller, +0.031) gap, far exceeding the ceiling. The effect is **~3× larger** than
  WikiText's +0.045 → not WikiText-specific.
- **One-shot-suffices / loop-inessential generalizes cleanly** — `g-static` fails again ((B−static) CI
  [−0.025, +0.051]); the two-timescale loop adds nothing over the one-shot reweight on this domain too.
- No collapse (d_eff ratio 0.90); floor ≈ 0 (the linear local read fails here too).

**The wrinkle (an honest scope finding) — the gauge's INERTNESS is WikiText-specific.** On WikiText the
random-reorder gauge was inert (gauge ≈ A → "it's the *targeting*, not the mass"); on TinyStories the gauge
**also closes part of the gap** ((gauge−A) CI **[+0.042, +0.074] > 0**), so `g-gauge` fails. **But targeting
still wins decisively:** B−A (+0.165) is ~3× gauge−A (+0.058) — the emergent priority captures most of the
effect; random replay captures a baseline fraction. **Mechanistic reading:** the gauge tracks the corpus's
*paradigmatic density*. WikiText's gap is the recency-bias dropping *specific, rare* paradigmatic-context
pairs → only targeting recovers them → gauge inert. TinyStories is **paradigmatically saturated** (anchor 2×
stronger; king/queen/princess patterns pervasive), so even random re-presentation re-injects paradigmatic
mass → the gauge is non-zero. *The "only-targeting-matters" claim is thus corpus-dependent (clean on sparse-
signal corpora, partial on saturated ones); the "targeting-beats-random" and "gap-close" claims hold on both.*

**Predicted clean test (not yet run):** ag_news (weak own-anchor +0.062, *sparse* like WikiText) should show
the gauge **inert** again — confirming gauge-inertness tracks signal sparsity, not WikiText specifically. A
cheap decisive follow-up.

**Phase B verdict:** the banked claim — *a real, substrate-confirmed LOCAL surprise-targeted pair-replay
closes the locality gap; one-shot suffices* — **GENERALIZES across domains** (stronger on TinyStories), with
one honest scope caveat: the anti-homunculus *gauge* is fully discriminating only on sparse-signal corpora
(targeting still beats random everywhere). Remaining sharpening if pushed harder: the **frequency-targeting
control** (is "surprise" doing more than inverse-frequency up-weighting?) + the ag_news gauge-inertness test.
RUNG-1 generality; NOT graduation; magnitudes partition-inflated.

## 8. Sharpening (the caveat-closing controls) → the mechanism DEFLATES to a frequency reweight (`experiments/73`, n=10)

Two cheap pre-identified controls (user-chosen "cheap sharpening, then commit"), added as additive arms.

**(a) Frequency-targeting control — DECISIVE + DEFLATIONARY.** A codes-INDEPENDENT arm `freq` re-weights
OBSERVED pairs by **inverse co-occurrence frequency** (rare pair → higher boost), no novelty/surprise/codes.
On WikiText (n=10): `freq` = **+0.222** vs A +0.179 vs B +0.225 vs ceiling +0.224.
- `freq − A` CI **[+0.034, +0.052]** → **pure inverse-frequency up-weighting closes the gap by itself** (as
  much as the full surprise mechanism).
- `B − freq` CI **[−0.009, +0.012]** (mean +0.003) → **B ≈ freq**: the codes-derived "surprise" adds NOTHING
  over plain rare-pair up-weighting.
- **So every distinctive CE-1 ingredient is inessential:** not the two-timescale loop (`g-static`, §4/§6),
  not the codes-derived surprise/novelty (this control), not pattern-separation. The gap-close is achieved by
  **re-weighting toward rare observed pairs** — which is, per the §1 surprise≈PMI guard (now empirically
  confirmed), essentially **re-applying the PMI the recency-bounded operator under-weights**. The recency-bias
  drops rare pairs (and paradigmatic-context pairs are rarer than collocational); ANY rarity-targeted reweight
  (frequency or surprise) recovers them. The gauge (random, non-rarity) still fails → it IS *targeting* (rare
  pairs), just not "surprise."

**(b) ag_news gauge test — INCONCLUSIVE (corpus too weak).** ag_news (sparse news domain, SVD anchor +0.062,
calib_ok) was meant to test "sparse signal → gauge inert (like WikiText)". But its **partition signal is
degenerate**: offline ceiling = **−0.017**, all arms ≈ 0 (A +0.031, B +0.033, gauge +0.027). With no gap,
the gauge test is meaningless here. *Side-finding:* a valid raw-SPPMI-**SVD** anchor (+0.062) does NOT
guarantee a **partition** signal — consistent with the project-wide theme that the partition and the SVD pick
up different structure; ag_news's weak paradigmatic content survives the (linear) SVD read but not the
(nonlinear) partition. The gauge-sparsity story (from §7) thus rests on the WikiText(inert)-vs-TinyStories
(partial) contrast, not on ag_news.

## 9. Final verdict on the CE-1 thread (fully controlled)

**What is real (survives every control):** the recency-bounded LOCAL writer leaves a measured locality gap
to the global pass on the paradigmatic axis; a **LOCAL pair-reweight closes it** — robustly (B−A 10/10),
**real on the FHRR substrate** (rung-2 port, §6), and **domain-general** (TinyStories, §7). First local
mechanism to close such a gap.

**What is NOT supported (deflated by controls):** CE-1's *distinctive* hypothesis — that an **emergent,
codes-derived, two-timescale, pattern-separated replay schedule** is what closes the gap. It isn't: the
operative signal is **inverse co-occurrence frequency (≈ PMI)**; the loop, the codes-derived surprise, and
pattern-separation are all inessential (§6, §8). So the honest mechanism is **"up-weight the rare
co-occurrence pairs the recency-bias drops"** — mundane, close to what SPPMI already does. The exciting
"emergent surprise replay" framing does not hold. **Banked accordingly: the gap-close is real but the
mechanism is a frequency/PMI reweight, NOT an emergent replay-schedule effect.** The discipline (the freq
control) caught the over-attribution — the 4th narrowing this year (after 126, 127, exp70).
