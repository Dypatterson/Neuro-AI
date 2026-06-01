# Report 127 — Nonlinear-competition kill-test → GENUINE-BUT-NARROW (a nonlinear-partition escape, NOT a competition escape)

**Status:** Phase-3. A substrate-free **DRILL-DOWN oracle** (NOT graduation). Frozen pre-commit
(grill-hardened): [phase-3-nonlinear-competition-kill-test-precommit.md](../../notes/emergent-codebook/phase-3-nonlinear-competition-kill-test-precommit.md).
Harness: `experiments/68_nonlinear_competition_kill_test.py`. 5 seeds, D=2048, WikiText-2 V=2002,
n=40 SimLex≥5 non-cooc pairs. Floor (055-058) untouched.

**Verdict: GENUINE-BUT-NARROW (downgraded from the provisional QUALIFIED POSITIVE by a 6-agent
adversarial verification — §0.5).** k-WTA on `build_S` clears the gate — BUT a **plain global Lloyd
k-means one-hot of the same `build_S` rows reproduces it almost exactly** (spec +0.250 vs k-WTA
+0.248; B-KILL comparable), so the **headline para-vs-random specificity is a property of ANY hard
nonlinear PARTITION of `build_S`, NOT of local competition.** What genuinely survives: (1) the faithful
LINEAR local read (`grow_G` +0.002) fails on the SAME operator while *every* nonlinear partition
succeeds (+0.25) — the structure is only +0.011 in the raw rows, so it is *produced* by the partition,
build_S-specific (k-means on a random Gram = +0.010), cosine-independent (argmax co-membership +0.300,
perm-p 0.0000, broad across 30/40 pairs); (2) competition buys only a *modest* label-shuffle B-KILL
bonus over k-means. **So the operative bound is LINEAR-PROJECTION-vs-NONLINEAR-PARTITION — not
global-vs-local, not competition-vs-not.** Magnitude uninterpretable (low-dim inflation, recoverE 1.67;
k-means inflates identically — NOT "beats SVD"). NSM (soft competition) genuinely null. M_trans
inconclusive. Literature motivating-only (Sengupta-2018 uncarded; sparse-Hopfield link-only).

## 0.5 VERIFICATION UPDATE (6-agent adversarial workflow — supersedes the provisional verdict)

The provisional read (below, §1-§5) called this a QUALIFIED POSITIVE for *local hard competition*. The
adversarial verification **refuted that mechanism-specific claim** and salvaged a narrower real finding:
- **The decisive control (the one §6.2 flagged but did not run before banking):** a *competent-but-non-
  assembly* clustering — plain spherical/Lloyd **k-means one-hot** of the `build_S` rows — hits spec
  +0.246–0.260 (k=8/16/32), **statistically indistinguishable from the k-WTA writer** (+0.248), with the
  same k-robustness. Reproduced by two independent agents (5-seed and 20-seed). The frozen-random
  no-learning control in §2 was an *incompetent* (noise) clustering; it never excluded k-means. ⇒ the
  gate headline is a partition property, not a competition property.
- **What survives (corroborated, multi-seed):** grow_G (+0.002) fails where any partition wins (+0.25);
  raw rows hold only +0.011 (so the partition *produces* it); random-Gram k-means = +0.010 (build_S-
  specific); the geometry is cosine-independent + broad (survives dropping the top-10 pairs at +0.126);
  k-WTA's B-KILL margin is *modestly* cleaner than k-means' (a small competition-specific bonus).
- **NSM (soft competition) is genuinely null** — every de-collapsing fix fails the g4 anti-inflation
  gate (learned − no-learning ≤ 0 at all k); de-collapsing requires rank-1 common-mode removal which
  *itself* manufactures a no-learning artifact. The escape is hard-partition-specific.
- **The corrected bound + next move:** the project's genuine differentiator over global Lloyd k-means is
  **LOCALITY + ONLINE STREAMING** (no global Gram pass, brain-analogous), NOT "partition vs projection"
  (k-means already proves that). **NEXT = an ONLINE/STREAMING local k-WTA vs OFFLINE global k-means
  head-to-head on `build_S`, headlined on the label-shuffle B-KILL** (the only surviving
  competition-specific signal) — if online-local preserves the B-KILL margin that offline k-means lacks,
  *that* is the load-bearing novel result worth carding the assembly primaries for. Anti-homunculus on
  the cap = clean (fixed precommitted top-k). BUILD remains the Phase-5 fence.

*(The provisional analysis §1-§5 below is retained for provenance; read §0.5 as its correction.)*

---

## 1. The grid (5 seeds, calibration anchor +0.1092 / kq 0.222 valid)

| operator | writer | best k | spec (para−rand) | CI-lo | no-learn ctrl | linear floor | recover/E | **B-KILL** | PASS |
|---|---|---|---|---|---|---|---|---|---|
| **build_S** (SPPMI 2nd-order) | **k-WTA** | 8 | **+0.278** | +0.098 | +0.003 | +0.002 | 1.67 | **5/5** | **✓** |
| build_S | k-WTA | 16 | +0.221 | +0.055 | +0.046 | +0.002 | 1.33 | 4/5 | ✓ |
| build_S | k-WTA | 32 | +0.238 | +0.074 | +0.056 | +0.002 | 1.44 | 5/5 | ✓ |
| build_S | NSM | — | +0.000 | — | +0.017 | +0.002 | 0.00 | — | ✗ (COLLAPSED) |
| **M_trans** (transition) | k-WTA | 8 | +0.083 | **−0.106** | **+0.108** | +0.005 | 0.41 | 1/5 | ✗ |
| M_trans | NSM | — | +0.003 | +0.001 | +0.07–0.11 | +0.005 | 0.01 | 0/5 | ✗ (COLLAPSED) |

`build_S = rownorm(SPPMI@SPPMIᵀ)` (the Report-123 +0.021 operator); `M_trans = (T+Tᵀ)/2` (Oracle E's
operator, NMF ceiling +0.19). k-WTA cap = k/4 (precommitted); NSM = the rectifying `[Wx−My]₊` network.

## 2. build_S — the genuine, controlled pass

k-WTA clusters the SPPMI-similarity rows by hard competition (each token's top-k/4 units; competitive
Hebbian move toward the won inputs). Para pairs (king/queen, river/sea — non-co-occurring, cooc≤2) land
in **shared assemblies** (para cos 0.66 vs shuffled-para 0.39 ≈ random 0.38), collo highest (0.93). The
result is robust:
- **All k pass** (8/16/32), B-KILL 4–5/5 each — not a single-cell fluke.
- **The no-learning control is clean** (frozen-random k-WTA spec ≈0.003–0.056 ≪ the learned 0.22–0.28):
  the architecture/sparsity alone produces ~nothing; the signal is the **competitive learning**.
- **The label-shuffle is pair-specific** (para ≫ shuffled-para): NOT para-set hubness (the failure mode
  that killed Reports 119/126 and exp67). Shuffled-para ≈ random.
- **The linear local read fails on the SAME operator** (`grow_G` +0.0023) — so this is precisely the
  bound's escape: power-iteration → dominant collocational mode; hard competition → the subdominant
  part-based cluster structure (grill Q1 / Sengupta et al. NeurIPS 2018 manifold-tiling).

## 3. Caveat A — the magnitude is low-dim-INFLATED (do NOT claim "k-WTA > SVD")

k-WTA spec +0.278 > E (NMF) +0.166 > SVD anchor +0.109 (recover/E = 1.67). This is the Report-125 §3
low-dim-inflation tell: sparse k-dim nonneg codes inflate absolute cosines (para 0.66 baseline). **The
absolute magnitude is NOT comparable to the +0.109 SVD or a D=4096 substrate.** The verdict rests
ENTIRELY on the DIFFERENTIAL controls (label-shuffle para−shuffled; learned−no-learning), which net out
the uniform inflation — and those are clean. Stated correctly: k-WTA produces a pair-specific
paradigmatic *clustering* that the linear local read misses; its *magnitude* is uninterpretable.

## 4. Caveat B — M_trans is control-CONTAMINATED (inconclusive, not a contradiction)

On `M_trans`, the **no-learning control itself inflates** (frozen-random k-WTA spec +0.096–0.108): the
transition-operator rows are peaky enough that a random cap already manufactures structure, so the
learned writer cannot beat its own architecture (g4 fails; spec CI-lo negative at k=8/16). M_trans
therefore **cannot adjudicate** — it is neither a clean replication nor a clean failure. The build_S
pass is the trustworthy cell precisely because its no-learning control is clean. *(Why the operators
differ: build_S is a smooth dense Gram → random cap = noise; M_trans is a peaked transition matrix →
random cap = structure. This is itself worth understanding before a build.)*

## 5. Caveat C — NSM (soft competition) COLLAPSED (implementation, not a valid null)

The NSM rectifying network `y←[Wx−My]₊` collapsed to **rank-1** (all output rows identical, cos=1.000)
on both operators, even with W-row normalization — a degenerate fixed point on the common-mode-dominated
operator, NOT a valid null. So **only the HARD competition (k-WTA) was validly tested**; whether SOFT
competition (NSM) escapes is UNRESOLVED and needs a stabilized implementation (stronger lateral
inhibition / whitening / a different settling schedule). Flagged for the follow-up; do not read the NSM
rows as evidence.

## 6. Disposition — greenlight GROUNDING, but adversarially verify FIRST

Per the frozen disposition, a PASS → **open + CARD the assembly / sparse-Hopfield primaries**
(Papadimitriou-Vempala Assembly Calculus; arxiv:2411.08590, arxiv:2309.12673, pdf:sqhn-2024; the
manifold-tiling Sengupta et al. 2018) **before** any build; then a competitive-writer grounding +
precommit. **A BUILD remains the Phase-5 fence (the user's to lift).**

**BUT this is the first positive after 121/123/124/125/126 — the session's discipline (adversarially
verify every surprising result; multi-seed; no single-operator banking) binds hardest HERE.** Before
treating "hard local competition breaks the linear bound" as established, run the adversarial pass:
1. **Why does build_S pass but M_trans contaminate?** Is the build_S signal a genuine paradigmatic
   escape or a `build_S`-specific clustering artifact? (The no-learning control says learning-driven,
   but a deeper operator-structure check is warranted.)
2. **Is k-WTA-clustering-of-build_S more than "cluster the SPPMI Gram"?** build_S[king,queen] is already
   high (similar context profiles); competitive clustering trivially co-locates them. The non-trivial
   claim is that the LINEAR local read can't reach it (true: +0.0023) — but a skeptic should check the
   result isn't an artifact of reading clustering-assignment overlap as cosine.
3. **Fix + re-run NSM** (soft competition) to see if the escape is k-WTA-specific (hard cap) or general
   to nonneg competition.
4. **A frequency-matched / collocational deep-dive** on which para pairs drive it (king/queen itself?).

## 7. Artifacts
- `experiments/68_nonlinear_competition_kill_test.py`; `reports/127_nonlinear_competition/_kill_test_wikitext.json`.
- New mechanism code: `nsm_features` (rectifying NSM, COLLAPSED — needs fix), `kwta_features` (assembly
  competitive learning, the validated writer), per-writer no-learning control, the static read +
  label-shuffle + random-nonneg controls.
- 3a CORRECTION applied (precommit §2): row-streaming on a 2nd-order operator (no global eig);
  symmetric-NSM-via-sqrt(M_trans) is disallowed (smuggles the banned global eig).
- DEFERRED (precommit-committed, not in this run): the faithful H-coupled READ-2 (exp67 closed its
  reduction); the 3b fully-online window-stream input mode; the FHRR-port (Stage-1, fence).
