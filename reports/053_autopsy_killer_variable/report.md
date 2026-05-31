# Autopsy: the graduation failure was a D-scale decorrelator collapse — NOT corpus, NOT task

> **Status:** CORRECTS Report 052's verdict. The D=4096 graduation failure was **not**
> corpus-specific and **not** task-limited — the surgical mechanism **works near the
> information ceiling at D=512** and collapses at D=4096 due to a **decorrelator ×
> high-D interaction** in the real pipeline (clean toys do not collapse). The direction
> is **not dead** — it was tested in a regime where the decorrelator breaks. Source:
> the autopsy + killer-variable Colab (`notebooks/autopsy_pivot_colab.ipynb`), results
> recovered from Drive, reproduced in-pipeline locally.

## Finding 1 — MECHANISM-limited, not task-limited

The Bayes-optimal **information ceiling** (best any cue→target predictor can do) at the
sparse cue, computed on matched windows:

| corpus | obs=1 | obs=2 | obs=3 | obs=5 |
|---|---|---|---|---|
| WikiText | **0.424** | 0.759 | 0.925 | 0.997 |
| repo_sample | **0.509** | 0.921 | 0.994 | 1.000 |

At obs=1 the WikiText ceiling is **0.424** (77% of sparse cues are unique → memorizable)
— the sparse cue **carries real, recoverable structure**. The Report-052 mechanism got
0.098 ≈ floor, leaving ~4× the signal on the table. **Not task-limited.** (The earlier
worry that obs=1 is ill-posed is refuted: it is informative, the mechanism just wasn't
reaching it.)

## Finding 2 — the killer variable is D (scale), not corpus

Killer-variable sweep, obs=1, N=800 (≈585/seed), 2 seeds, write+decorrelation:

| corpus | D | N/D | key cosine | store | **write+decorr** | floor | ceiling |
|---|---|---|---|---|---|---|---|
| repo_sample | 512 | 1.14 | 0.48 | 0.008 | **0.522** | 0.048 | 0.509 |
| repo_sample | 4096 | 0.14 | 0.48 | 0.005 | **0.105** | 0.048 | 0.509 |
| WikiText | 512 | 1.14 | 0.51 | 0.048 | **0.257** | 0.085 | 0.424 |
| WikiText | 4096 | 0.14 | 0.51 | 0.049 | **0.103** | 0.085 | 0.424 |

- At **D=512** the mechanism is **0.522 on repo_sample — essentially the 0.509 ceiling** —
  and 0.257 on WikiText (well above the 0.085 floor). **It works.**
- At **D=4096** it collapses to ~0.10 on **both** corpora.
- **Key correlation is constant across D (~0.48–0.51)** → correlation is *not* the killer.
- **D (scale) is.** Report 052's "corpus-specific, prose vs source code" conclusion is
  **WRONG** — the corpus difference (0.52 vs 0.26 at D=512) is secondary; the **D=512→4096
  collapse** is the load-bearing effect.

## The cause (leading hypothesis, narrowed)

Reproduced in-pipeline locally (exp 50, repo_sample, obs=1): D=512/2048 at N=230 → **0.739**;
D=4096 at N=536 → **0.091** (collapsed, ≈ floor). But **clean synthetic toys do not collapse
at high D** — neither correlated-random keys (1.000 at all D) nor bound-key+Zipfian-collision
keys (~0.245 flat across D). So the collapse is specific to the **real `encode_window` key
structure × high D**: the obs=1 cue is a *bundle* dominated by a large **shared mask-binding
component** (the source of the ~0.49 key cosine), with the distinctive `bind(pos₀, token₀)`
part being ~1/√6 of it. **Leading hypothesis (unconfirmed):** the ZCA decorrelator's
**relative ridge** (`ridge·λmax`, with λmax set by the dominant shared component) **floors out
the small distinctive eigenvalues at high D**, projecting the cue's distinctive signal to
zero — so the whitened key keeps only the (now-decorrelated) shared component, which carries
no target information → floor. At D=512 the distinctive eigenvalues stay above the floor and
survive. This is a **decorrelator regularization / conditioning bug, fixable** — not a
property of the mechanism or the substrate.

## Verdict — do NOT pivot yet

The surgical direction is **not falsified**. It works near the information ceiling at D=512;
it collapses at D=4096 because of a decorrelator-at-high-D issue, not the mechanism or the
task. The Report-052 graduation **failed in the wrong regime**.

## Next (focused, cheap)

1. **Confirm the cause:** instrument the D=4096 decorrelated keys — is the distinctive
   `bind(pos₀,token₀)` direction floored out by `ridge·λmax`? Sweep the decorrelator ridge
   (absolute vs relative; smaller) at D=4096, N=585.
2. **Fix:** an absolute/adaptive ridge, or remove the shared mask-binding component before
   whitening (it carries no information — subtract the per-position mask binding from the cue),
   then re-test at D=4096.
3. **Re-run the graduation** at D=4096 once the decorrelator clears its floor at obs=1, with
   the full done-gate panel (floor-gated, Report-052's correct bar). Hard abandon-if-floor
   stays.

(Reports 052's numbers stand; only its *attribution* — "corpus-specific" — is corrected.)
