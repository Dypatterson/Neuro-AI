# Option-B Grounding Brief — Can a latent/hierarchical layer ABOVE the flat code escape the subdominant-modes bound?

**Deliverable type:** GROUNDING (not a build). Phase fence intact: this routes toward Option B / Phase-5 architecture and recommends a substrate-free oracle; it does NOT authorize a Phase-5 build. Lifting the build fence remains the user's call. Floor (055-058) untouched throughout.

## 1. The bound this must clear (re-verified this session, not assumed)

Report 123 §4 ([reports/123_growth_redesign_sweep_null/report.md:56-72]) and Report 124 §6 ([reports/124_r3_directional_oracle/report.md:101-117]) establish a **mechanically exact local-vs-global bound**. The faithful local dynamic is `exp61.grow_G` (verified at experiments/61:334): `G <- normalize(alpha * normalize(M@G) + (1-alpha)*G)` — a **damped power iteration on a fixed operator M**. Power iteration provably converges to M's DOMINANT eigenspace. The finding restated: **paradigmatic (substitutability/king-queen) structure lives in the SUBDOMINANT eigenvectors of M; the dominant eigenvectors are collocational/frequency.** Global SVD reaches it (+0.109, kq 0.222); local iteration cannot (≤+0.021 symmetric, +0.0067 directional). Report 124 proved this is **invariant to directional asymmetry**. Every FLAT-code growth rule inherits it.

**The make-or-break question is singular and precise:** does a latent layer whose objective selects for substitutability have its DOMINANT (locally-reachable) modes carry the paradigmatic axis? The corpus does NOT settle this — that is exactly what the oracle must test.

## 2. The decisive adjudication — the escape is operator-class-specific, not 'latent-vs-flat'

The two steelmen are **asymmetric**, and that asymmetry is the key finding (Memo C, ratified):

- **NO-ESCAPE evidence (Report 119 collapse, washout, AAR) is all about PREDICT-CONTEXT / CONTRASTIVE objectives** (InfoNCE, PAM L_assoc, masked-token). These are second-order co-occurrence factorizers in disguise — SGNS == implicit SPPMI factorization (Levy-Goldberg) == LITERALLY the operator exp62 already SVDs. **The bound EXTENDS to them.** WS-InfoNCE (Report 119) is the proof: a predict-context latent ALREADY failed by collapse (d_eff 341->165, negative held-out).
- **ESCAPE evidence (SFA, SR-reweighting) is about a STRUCTURALLY DIFFERENT operator** — the time-derivative covariance (SFA) or the inverted transition operator (SR). SFA extracts the SLOWEST features == minor-eigenvector of the input but DOMINANT-eigenvector of the derivative operator. This is the ONLY lineage with a principled, non-circular reason to escape.

**Crux:** Option B escapes the bound IF AND ONLY IF the latent objective is a 'slowness / minor-of-co-occurrence / inverted-operator' objective (SFA, SR-reweight), NOT a 'predict-context / contrastive-co-occurrence' objective (InfoNCE, PAM, masked-prediction, Gamma1 context-residual). The first changes the operator; the second re-factorizes the same one. Every prior null (119, the whole flat family) is the SECOND kind. The genuinely untested question is the FIRST kind.

This resolves the survey's two latent candidates: **Gamma1 (context-residual PC consolidation) is a predict-context objective in atom-space -> SAME-OPERATOR -> predicted-NULL; DEMOTE it** despite its current Leader ranking (written pre-123/124). **Gamma3 (SFA), recast as a separate slow LATENT state, is the derivative-operator objective -> the LEAD escape candidate.**

## 3. GO/NO-GO — CONDITIONAL GO

**Pursue Option B, but ONLY via the slowness/SR operator-class, and ONLY through the cheap oracle first.** The honest read is NOT 'latent layers are magic': three of five candidates (Gamma1, tPC, value-codebook-JEPA) are predicted-NULL or already-killed, and even the lead (SFA) has a sub-50% escape prior dominated by the unproven FHRR port. But there IS a genuine, non-circular escape hypothesis (slowness == dominant-of-a-different-operator) that the project has NEVER exercised, and a ~1-day substrate-free oracle can kill-or-greenlight it before any build. That asymmetric payoff is a GO — conditional on the oracle and on the build fence staying the user's.

## 4. The ≤3 strongest genuinely-distinct candidates

| Rank | Candidate (latent-ABOVE) | Why it might escape | FHRR-port risk | Primary to open+card |
|---|---|---|---|---|
| **B1 LEAD** | **SFA / slow-latent layer** (Franzius-Sprekeler-Wiskott; Sprekeler SFA~=Laplacian-eigenmaps; survey Gamma3 RECAST as separate slow state) | Slowness extracts the slowest factor = dominant-of-the-derivative-operator; substitutable tokens -> same slow latent. The one principled, non-circular exemption from the power-iteration bound. | **HIGH/UNPROVEN** — slow z is Euclidean; native unbind/MHN recovery is the Stage-1 go/no-go | **pmc:PMC1963505** (link_only, NO card/PDF) — the whole escape argument rests on it |
| **B2 ALTERNATE** | **SR-reweighted / successor-eigenvector latent** (Stachenfeld 2017; Machado eigenoptions) | Psi=(I-gamma*T)^-1 inflates the slow/low-freq (subdominant) modes by 1/(1-gamma*lambda); its TOP eigenvectors ARE the Laplacian low-freq structure | **HIGH + a 2nd risk:** global inverse, TD-on-FHRR BANNED online -> a LOCAL batch-offline Psi-estimate is unsolved | **SR canon (Stachenfeld 2017)+TCM CONFIRMED ABSENT** from manifest — zero grounding, heaviest open |
| **B3 CONTINGENCY** | **PCN-above-MHN hierarchy** (Salvatori line; Rao-Ballard; tPC whitening) | Higher PC layers represent slow/abstract causes; whitening flattens the dominant collocational mode. AH-clean PC settling. | **HIGHEST/MOST DISTANT** — PC latents are Euclidean error-min units | arxiv:2402.10814 is the **WRONG paper** (feature-space autoassoc, not a PC hierarchy); genuine source **arxiv:2509.01987** abstract-only + the **KILLING washout caution** — must open before build |

**Excluded by the bound + Report 119 (Option B must NOT reinvent):** tPC latent head (REINVENTS-KILLED — Report 124 directional null + already-adjudicated: asymmetric AHN write encodes order which within-scene lacks, whitening already == the L2 decorrelator in H); standalone PAM-L_assoc / WS-InfoNCE on the VALUE codebook (literally Report 119: d_eff 341->165); Gamma1 context-residual (predict-context atom-space update -> same-operator NULL). The two reusable discriminators the project already paid for: the **SEPARATE-G discipline** (experiments/61 §0) keeps every candidate off the Report-119 floor-collapse path; the **subdominant-modes bound** means floor-safety is NOT enough — a separate table fed a collocational objective just relocates the collocational-dominant-mode problem one layer up.

## 5. The recommended cheap feasibility oracle — ORACLE B (exp65)

**Classification:** DRILL-DOWN feasibility oracle (NOT a graduation experiment), the latent-layer analog of exp62 §10 / exp63. Substrate-free, CPU, ~1 day, near-clone of experiments/63 (importlib-reuse, one new statistic). Author the FROZEN precommit FIRST: `notes/emergent-codebook/phase-3-option-b-latent-oracle-precommit.md`.

**The one question:** does changing the operator from co-occurrence to slowness/SR move paradigmatic structure into the DOMINANT modes the FAITHFUL local `grow_G` dynamic reaches?

**Verdict-bearing read = exp61.grow_G on row_center(S_lat), NOT the SVD spectral read.** Report 124 §4 PROVED the low-rank spectral 'reachability' proxy is a CONTRACTION ARTIFACT (+0.53 at r=5 vanished to +0.008 at full rank; the control scored the same). The SVD read is retained ONLY as a contraction-contaminated upper-bound diagnostic. (This demotes Memo A's 'read where on the SFA spectrum' framing on banked evidence.)

**Three operators on the frozen WikiText-2 windows** (V=2002, W=6, gamma=0.9 — same as exp63, calibrates against +0.1092/kq-0.222):
- `M_cooc` = exp61 SPPMI operator [known-subdominant CONTROL].
- `M_slow` = one-pass SFA-surrogate = lag covariance `<(x_{t+1}-x_t)(x_{t+1}-x_t)^T>` over the window stream, read at its SMALL-eigenvalue end (SFA == bottom of Delta). Closed-form, NO training, cheapest + most-principled.
- `M_SR` = `Psi=(I-gamma*T)^-1`, `T=rownorm(directional-cooc F)` [flashlight-only, labeled].

New code (~30-60 lines, the single conceptual addition): `build_lag_covariance(windows, context_vectors)` for `M_slow`, mirroring exp63's `build_directional_cooccurrence`. Everything else reused verbatim: exp61.{build_cooccurrence, build_sppmi, pick_k_by_density, row_center, grow_G, d_eff, load_simlex_pairs, random_matched_pairs, corr_bootstrap_ci}; exp62.{_cos_real, _fcos, _boot_diff}; exp63.{spectral_read, local_iteration_read, raw_sppmi_svd_anchor, select_pairs}.

**FROZEN GATE (all four for PASS, anchored to global +0.109 / local +0.021 / directional +0.0067):**
- **g1** best COLLAPSE-FREE (d_eff_ratio>=0.5) `grow_G` specificity on row_center(M_slow): **CI-LO > +0.04** (~2x flat-local bound, ~1/3 of global; clears the +0.0067 directional ceiling by margin).
- **g2** M_slow local-read beats the matched-r flat-SPPMI baseline S_flat at the same r by **>= +0.02** (prevents 'did nothing the global SVD already did').
- **g3** `corr(log cooc, drift)` CI-HI **< 0.15** via exp61.corr_bootstrap_ci on the grow_G drift — the NON-NEGOTIABLE collocational discriminator that killed all 11 collapse-free positives in 123 and every cell in 124.
- **g4** floor-guard: latent Gram eff-rank **>= 0.25*d** AND d_eff_ratio>=0.5 (rules out the WS-InfoNCE 341->165 collapse signature).

**MANDATORY calibration anchor (INVALID-vs-NULL bright line, verbatim from exp63 raw_sppmi_svd_anchor):** raw-SPPMI-SVD must reproduce **+0.1092 CI[0.082,0.137], kq 0.222** in the SAME run; miss -> INVALID (corpus/pairs/build wrong), not NULL, nothing else read. Collocational sanity floor (collo-random CI-lo>0 on every operator) must pass -> a real NULL is a MECHANISM verdict, not a corpus-scale one.

**Outcomes:** PASS = M_slow's dominant modes ARE paradigmatic AND a local dynamic reaches them -> greenlight an Option-B SFA-latent GROUNDING+PRECOMMIT (NOT a build; FHRR-port Stage-1 still downstream). NULL = the slow/SR operator ALSO hides paradigmatic in subdominant modes OR the signal is collocational -> bound is OPERATOR-INVARIANT for single-projection latents -> escape (if any) needs a genuine MULTI-LAYER hierarchy or re-scope. Either outcome decisive for ~zero build cost — exactly exp63's job.

**Why SFA is oracled first:** it has the closed-form global solve (trivial substrate-free build) AND the strongest principled claim its dominant modes are paradigmatic — so a NULL here is the most informative kill, a PASS the most credible greenlight.

## 6. The three hard cautions, folded into the design

1. **FHRR-native port UNPROVEN:** oracle is SUBSTRATE-FREE and explicitly PRE-PORT — it tests the operator question before the port is attempted (exactly as exp63 killed flat R3 pre-port). grow_G uses the FHRR substrate's normalize/repulsion_force so the question IS asked in FHRR space, but the full latent-LAYER port is a downstream Stage-1 gate the oracle does NOT cross. PASS greenlights GROUNDING ONLY.
2. **Corpus-specific transfer / AAR:** folded by g3 (corr<0.15 separates corpus-specific cooc clustering from paradigmatic) AND surfaced as an unresolved fence — the oracle is in-sample; the downstream precommit MUST carry a held-out paradigmatic arm. SEPARATE-G discipline is the structural guard so a corpus-specific latent cannot regress the floor.
3. **Overfit washout (arxiv:2509.01987):** the oracle reads at a frozen d-grid (d in {32,64,128}); a PASS that only appears at the smallest d is the washout fingerprint and routes to NULL-disposition, not greenlight. Flagged in the report as not-yet-transferable, never banked as structure. This is also why B3 (PCN) is demoted — its predict-context core is the washout-vulnerable family.

## 7. Calibrated prior on the oracle's outcome

**~60-65% NULL, ~25-30% PASS, ~10% INVALID.** Two operator-swaps already failed under the harsh faithful grow_G read (123 symmetric +0.021, 124 directional +0.0067, both collocational); the lag-covariance/SR operator is a THIRD swap and the prior a third re-weighting survives where two did not is bounded below ~30%. PASS mass concentrates on M_slow (principled) over M_SR (global-inverse re-weighting least likely to survive grow_G). INVALID is lower than exp63's because all S_lat share the same symmetric Phi at the same k — the directional-density-mismatch INVALID cannot arise (a genuine simplification); residual INVALID is numerical (generalized-eig conditioning -> regularize-and-rerun). Mirrors the domain-expert's ~80-85%-NULL directional prior, adjusted UP on the PASS side because M_slow has a genuine principled-escape argument the directional operator lacked. Most likely outcome: a NULL banking 'the bound is operator-invariant for single-projection latents -> escape needs a genuine multi-LAYER hierarchy' — decisive either way.

## 8. Disposition

Recommend: (1) author `phase-3-option-b-latent-oracle-precommit.md` with the frozen gate above; (2) run Oracle B (exp65) — ~1 day, substrate-free; (3) on PASS, open+card pmc:PMC1963505 and write the SFA-latent grounding+precommit (build still fenced); on NULL, bank the operator-invariance finding and escalate to the genuinely-multi-layer-hierarchy question (heavier, separate). Flag for user: ratify SFA-first over PCN-first; ratify grow_G-as-verdict over spectral-read; agree the survey's Gamma1/Gamma3 entries need re-flagging; confirm the build fence stays the user's.