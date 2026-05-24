"""Verify the Phase 5 magnitude-floor formula's sensitivities to D, N, beta.

Reconciles brainstorm Finding 1 ("D-lever is weak") against the formula in
notes/notes/2026-05-21-phase5-headline-magnitude-floor.md:49-73.

Formula:
    noise_floor = 1.0 / sqrt(D)
    floor       = (1/beta) * log(1 + (N-1) * exp(-beta * (1 - noise_floor)))

Run:
    .venv/bin/python scripts/verify_magnitude_floor.py
"""

from __future__ import annotations

import math


def floor(D: int, N: int, beta: float) -> float:
    noise = 1.0 / math.sqrt(D)
    arg = (N - 1) * math.exp(-beta * (1.0 - noise))
    return math.log1p(arg) / beta


def main() -> None:
    print("=" * 70)
    print("Phase 5 magnitude-floor formula sensitivities")
    print("Formula: (1/beta) * log(1 + (N-1) * exp(-beta * (1 - 1/sqrt(D))))")
    print("=" * 70)

    # Reproduce the spec's table at fixed D=4096, varying N (sanity check).
    print("\n[1] At fixed D=4096, beta=10, varying N (matches spec table):")
    print(f"  {'N':>6} | {'floor':>12} | {'ratio vs N=1064':>16}")
    print("  " + "-" * 42)
    ref = floor(4096, 1064, 10.0)
    for N in (12, 100, 500, 1064, 4000):
        f = floor(4096, N, 10.0)
        print(f"  {N:>6} | {f:>12.4e} | {f/ref:>16.3f}x")

    # Brainstorm Finding 1's claim: D-lever is weak across 16x range.
    print("\n[2] At fixed N=1064, beta=10, varying D (BRAINSTORM FINDING 1):")
    print(f"  {'D':>6} | {'floor':>12} | {'ratio vs D=4096':>16}")
    print("  " + "-" * 42)
    ref = floor(4096, 1064, 10.0)
    for D in (256, 512, 1024, 2048, 4096, 8192, 16384):
        f = floor(D, 1064, 10.0)
        print(f"  {D:>6} | {f:>12.4e} | {f/ref:>16.3f}x")

    # The beta lever (which audit constraint #10 forbids retuning, but is
    # informative for understanding what dominates the floor).
    print("\n[3] At fixed D=4096, N=1064, varying beta (audit-locked at 10):")
    print(f"  {'beta':>6} | {'floor':>12} | {'ratio vs beta=10':>17}")
    print("  " + "-" * 43)
    ref = floor(4096, 1064, 10.0)
    for b in (5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 30.0):
        f = floor(4096, 1064, b)
        print(f"  {b:>6.1f} | {f:>12.4e} | {f/ref:>17.4f}x")

    # Combined D+N: if lower-D produced fewer atoms (because capacity is lower
    # at lower D, so mass-death survives fewer atoms), what's the joint effect?
    #
    # Three ΔE reference points (run each in turn for the ratio table):
    #   - report 053 mean (n=10, n_cues=3000, one cue regime): +0.00130
    #   - report 057 cross-seed β=10 best (n=10×200 cues, K=1): +0.002289
    #     CI [+0.00109, +0.00349], 9/10 seeds positive — pre-Tier-0 best estimate
    #   - report 058 best cue-regime cell (cd=0.6, bns=0.05):  +0.002478
    #     CI [+0.00113, +0.00383], 9/10 seeds positive — best operating point yet
    # Codex audit 2026-05-23 caught that prior version used 053's value; both
    # newer values are presented now so the strategic implication is visible.
    de_refs = {
        "report 053 (n=10, original)": 1.30e-3,
        "report 057 (cross-seed β=10 best)": 2.289e-3,
        "report 058 (best cue cell cd=0.6)": 2.478e-3,
    }

    for de_label, observed_de in de_refs.items():
        print(f"\n[4·{de_label}] Hypothetical Path A trajectories (capacity-proportional, D and N both shrink):")
        print(f"  observed ΔE = {observed_de:.4e}")
        print(f"  {'D':>6} | {'N':>6} | {'floor':>12} | {'ratio vs floor':>15}")
        print("  " + "-" * 50)
        # Best-case assumption: N scales linearly with capacity ~ D (so smaller D
        # ends up with proportionally fewer atoms after mass-death).
        for D in (256, 512, 1024, 2048, 4096):
            N_scaled = int(round(1064 * D / 4096))
            if N_scaled < 2:
                N_scaled = 2
            f = floor(D, N_scaled, 10.0)
            ratio = observed_de / f
            marker = " <-- current" if D == 4096 else ""
            above = " ABOVE FLOOR" if ratio >= 1.0 else ""
            print(f"  {D:>6} | {N_scaled:>6} | {f:>12.4e} | {ratio:>15.3f}x{marker}{above}")

    # Worst-case assumption: D shrinks but N stays at 1064 (mass-death dynamics
    # might not scale capacity with D). Use the newest (058) ΔE here.
    observed_de = 2.478e-3
    print(f"\n[5] Pessimistic Path A trajectories (D shrinks, N stays 1064; ΔE = report 058 best cell {observed_de:.4e}):")
    print(f"  {'D':>6} | {'N':>6} | {'floor':>12} | {'ratio vs floor':>15}")
    print("  " + "-" * 50)
    for D in (256, 512, 1024, 2048, 4096):
        N_const = 1064
        f = floor(D, N_const, 10.0)
        ratio = observed_de / f
        marker = " <-- current" if D == 4096 else ""
        print(f"  {D:>6} | {N_const:>6} | {f:>12.4e} | {ratio:>15.3f}x{marker}")

    # Cross-check with the STATUS.md claim: "at D=512 the noise floor would
    # be ~8× higher; at D=1024 it would be ~4× higher."
    print("\n[6] STATUS.md claim check ('D=512 is ~8x higher floor at fixed N=1064'):")
    f_4096 = floor(4096, 1064, 10.0)
    f_1024 = floor(1024, 1064, 10.0)
    f_512  = floor( 512, 1064, 10.0)
    print(f"  D=4096 floor = {f_4096:.4e}")
    print(f"  D=1024 floor = {f_1024:.4e}  ratio = {f_1024/f_4096:.3f}x  (STATUS said ~4x)")
    print(f"  D= 512 floor = {f_512:.4e}  ratio = {f_512/f_4096:.3f}x  (STATUS said ~8x)")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print("""
Spec formula faithfully implemented (matches table 4096 column exactly).

Sensitivities (consistent with brainstorm Finding 1):
  - N-lever:     LINEAR in N for small (N-1)*exp(-beta*(1-1/sqrt(D)))
                 Range observed: 5.8e-5 (N=12) -> 5.5e-3 (N=1064) -> 1.9e-2 (N=4000)
  - D-lever:     WEAK at fixed N (only ~1.3x across 16x D range)
                 Mechanism: (1 - 1/sqrt(D)) -> 1 fast as D grows, so
                 beta*(1-1/sqrt(D)) ~ beta dominates the exponent.
  - beta-lever:  EXPONENTIAL (audit-locked, informational only)

STATUS.md's claim "D=512 is ~8x higher floor" is NOT consistent with the
spec formula at fixed N=1064. The actual factor is ~1.31x. Either:
  (a) the STATUS framing meant N also shrinks with D (capacity-proportional),
      in which case the Path-A trajectory has to commit to deliberate
      substrate downsizing AND argue that the signal does not shrink
      proportionally with N -- a stronger architectural claim than just "lower D",
  (b) the STATUS framing was an off-the-cuff estimate that did not match the
      spec formula, and Path A as currently framed is investing weeks against
      a ~1.3x lever, OR
  (c) a different formula (e.g. Plate unbind-amplitude noise sigma = sqrt(N/D))
      was the implicit basis for the STATUS framing; that formula goes the
      OPPOSITE direction (lower D -> WORSE unbind noise), which would foreclose
      Path A entirely.

Practical implication for the next session:
  - Before committing to a D-sweep diagnostic, decide which formula the
    Path-A premise actually rests on.
  - If it is the floor formula above, Path A needs an architectural argument
    that lower D leads to lower N WITHOUT proportional signal loss.
  - If it is the Plate formula, Path A is foreclosed and the strategic
    decision tree reduces to Path B (close + pivot) or substrate-pure
    mechanism redesign (IDP / log-prior softmax bias).

Note (Codex audit 2026-05-23): prior version of this script used only
report 053's +0.00130 as the observed ΔE. Reports 057 and 058 have
since produced better-grounded estimates (+0.002289 cross-seed β=10
best, +0.002478 best cue-regime cell), each ~1.8x larger. With those
values the Path-A capacity-proportional trajectory is materially MORE
favorable at the cheaper D values:
  - D=512, N=133: ratios rise from 1.40x (053) -> 2.47x (057) -> 2.67x (058)
  - D=1024, N=266: ratios rise from 0.80x (053) -> 1.41x (057) -> 1.52x (058)
At the latest (058) ΔE, even D=1024 with N=266 crosses the floor.
""")


if __name__ == "__main__":
    main()
