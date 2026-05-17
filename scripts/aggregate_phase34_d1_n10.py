"""n=10 aggregation for the D1 graduation run.

Reads per-seed JSON from `reports/phase34_integration_n3k_noak_seed{N}/`
(produced by `scripts/colab_phase34_integration_n10_n3k.ipynb`) and prints
the Phase 4 D1 graduation table.

Headline (per the pivot decision following reports 036, 037):
    D1 = Δ meta_stable_w3 at the final checkpoint, condition C (phase3_phase4)
                                                  minus condition A (baseline_static).

Drill-downs:
    Δ meta_stable_w2, Δ meta_stable_w4 (other scales)
    Δ R@10 (replicate / extend report 026 at the integration regime)
    Δ cap_t_05 (variance-bound per report 037; reported but not gated on)
    Δ top1 (demoted to drill-down per 2026-05-16 discipline note)

Outputs:
    stdout: per-seed table, aggregate CIs, sign breakdown
    reports/phase34_integration_n3k_noak_aggregate.json (machine-readable)
"""
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
RUN_TAG = "integration_n3k_noak"
SEEDS = [17, 11, 23, 1, 2, 3, 5, 7, 13, 19]


# 95% t-critical by df = n-1. Larger df → smaller t.
T95 = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 14: 2.145}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (p, center - half, center + half)


def t_ci(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    m = mean(values)
    if n == 1:
        return m, 0.0, m, m
    v = sum((x - m) ** 2 for x in values) / (n - 1)
    se = math.sqrt(v / n)
    t = T95.get(n - 1, 1.96)
    return m, math.sqrt(v), m - t * se, m + t * se


def load_seed(seed):
    p = ROOT / f"reports/phase34_{RUN_TAG}_seed{seed}/phase34_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def per_seed_row(seed):
    d = load_seed(seed)
    if d is None:
        return None
    a = d["results"]["baseline_static"][-1]
    b = d["results"]["phase3_reencode"][-1]
    c = d["results"]["phase3_phase4"][-1]

    def ms(payload, scale):
        return payload.get(f"meta_stable_w{scale}", float("nan"))

    return {
        "seed": seed,
        "n": a["n"],
        # raw final values, A=baseline, B=phase3, C=phase3+phase4
        "A_R10": a["topk"], "B_R10": b["topk"], "C_R10": c["topk"],
        "A_top1": a["top1"], "B_top1": b["top1"], "C_top1": c["top1"],
        "A_capt5": a["cap_t_05"], "B_capt5": b["cap_t_05"], "C_capt5": c["cap_t_05"],
        "A_ms2": ms(a, 2), "B_ms2": ms(b, 2), "C_ms2": ms(c, 2),
        "A_ms3": ms(a, 3), "B_ms3": ms(b, 3), "C_ms3": ms(c, 3),
        "A_ms4": ms(a, 4), "B_ms4": ms(b, 4), "C_ms4": ms(c, 4),
        # diagnostic counters
        "C_cands": c.get("candidates_total", 0),
        "C_cons": c.get("consolidations", 0),
        "C_deaths": c.get("deaths_total", 0),
        "C_drift": c.get("codebook_drift_from_initial", 0.0),
    }


def report_delta(label, values, headline=False):
    m, sd, lo, hi = t_ci(values)
    pos = sum(1 for x in values if x > 0)
    neg = sum(1 for x in values if x < 0)
    zer = len(values) - pos - neg
    excl = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
    tag = "  [HEADLINE]" if headline else ""
    print(
        f"  {label:<14} mean={m:>+.4f}  std={sd:.4f}  95%CI [{lo:>+.4f}, {hi:>+.4f}]  "
        f"signs {pos}+/{zer}0/{neg}-  {excl}{tag}"
    )
    return {"label": label, "mean": m, "std": sd, "ci_lo": lo, "ci_hi": hi,
            "pos": pos, "neg": neg, "zero": zer, "ci_excludes_zero": (lo > 0 or hi < 0),
            "per_seed": list(values), "headline": headline}


def main():
    rows = []
    missing = []
    for s in SEEDS:
        r = per_seed_row(s)
        if r is None:
            missing.append(s)
        else:
            rows.append(r)

    if missing:
        print(f"WARNING: missing seeds {missing} (no phase34_results.json)")
    if not rows:
        print("No data. Did the Colab run complete and sync?")
        return

    print(f"\n=== Per-seed final-checkpoint values (n={len(rows)}) ===")
    print(f"{'seed':>4} {'n':>4} | "
          f"{'R@10 A→C':>13} {'Δtop1':>8} {'Δcapt5':>8} | "
          f"{'ms_w2 A→C':>13} {'ms_w3 A→C':>13} {'ms_w4 A→C':>13} | "
          f"{'cands':>6} {'cons':>5} {'deaths':>7}")
    for r in rows:
        dR = r["C_R10"] - r["A_R10"]
        dT = r["C_top1"] - r["A_top1"]
        dC = r["C_capt5"] - r["A_capt5"]
        print(
            f"{r['seed']:>4} {r['n']:>4} | "
            f"{r['A_R10']:>5.3f}→{r['C_R10']:>5.3f} {dT:>+8.3f} {dC:>+8.3f} | "
            f"{r['A_ms2']:>5.3f}→{r['C_ms2']:>5.3f} "
            f"{r['A_ms3']:>5.3f}→{r['C_ms3']:>5.3f} "
            f"{r['A_ms4']:>5.3f}→{r['C_ms4']:>5.3f} | "
            f"{r['C_cands']:>6} {r['C_cons']:>5} {r['C_deaths']:>7}"
        )

    dR10 = [r["C_R10"] - r["A_R10"] for r in rows]
    dtop1 = [r["C_top1"] - r["A_top1"] for r in rows]
    dcapt5 = [r["C_capt5"] - r["A_capt5"] for r in rows]
    dms2 = [r["C_ms2"] - r["A_ms2"] for r in rows]
    dms3 = [r["C_ms3"] - r["A_ms3"] for r in rows]
    dms4 = [r["C_ms4"] - r["A_ms4"] for r in rows]
    # C - B is the cleanest architectural test: Phase 4 over phase3-only
    dms3_CB = [r["C_ms3"] - r["B_ms3"] for r in rows]
    dR10_CB = [r["C_R10"] - r["B_R10"] for r in rows]

    print(f"\n=== Aggregate (n={len(rows)}, df={len(rows)-1}, t-CI) ===\n")
    print("D1 graduation headline (Δ meta_stable_w3, C vs A):")
    h_d1 = report_delta("Δms_w3 (CA)", dms3, headline=True)

    print("\nD1 drill-downs (other scales):")
    h_d1_w2 = report_delta("Δms_w2 (CA)", dms2)
    h_d1_w4 = report_delta("Δms_w4 (CA)", dms4)
    h_d1_cb = report_delta("Δms_w3 (CB)", dms3_CB)

    print("\nReadout drill-downs:")
    h_r10 = report_delta("ΔR@10 (CA)", dR10)
    h_r10_cb = report_delta("ΔR@10 (CB)", dR10_CB)
    h_capt5 = report_delta("Δcap_t05 (CA)", dcapt5)
    h_top1 = report_delta("Δtop1 (CA)", dtop1)

    # Pooled-Wilson on R@10 (one event per test sample per seed) for additional context.
    sum_n = sum(r["n"] for r in rows)
    sum_A = sum(round(r["A_R10"] * r["n"]) for r in rows)
    sum_C = sum(round(r["C_R10"] * r["n"]) for r in rows)
    a_p, a_l, a_u = wilson(sum_A, sum_n)
    c_p, c_l, c_u = wilson(sum_C, sum_n)
    print(
        f"\n  Pooled-Wilson R@10:  A {a_p:.4f} [{a_l:.4f}, {a_u:.4f}]  "
        f"C {c_p:.4f} [{c_l:.4f}, {c_u:.4f}]  Δ={c_p - a_p:+.4f}  "
        f"({sum_A}/{sum_n} → {sum_C}/{sum_n})"
    )

    # Diagnostic counters
    cands = [r["C_cands"] for r in rows]
    deaths = [r["C_deaths"] for r in rows]
    drift = [r["C_drift"] for r in rows]
    print(
        f"\n  Diagnostics: candidates {min(cands)}…{max(cands)} (mean {mean(cands):.1f}); "
        f"deaths {min(deaths)}…{max(deaths)} (mean {mean(deaths):.0f}); "
        f"drift {mean(drift):.2e}"
    )

    out = ROOT / f"reports/phase34_{RUN_TAG}_aggregate.json"
    out.write_text(json.dumps({
        "run_tag": RUN_TAG,
        "seeds": SEEDS,
        "missing": missing,
        "rows": rows,
        "headline_d1_ms_w3_CA": h_d1,
        "d1_w2_CA": h_d1_w2,
        "d1_w4_CA": h_d1_w4,
        "d1_w3_CB": h_d1_cb,
        "R10_CA": h_r10,
        "R10_CB": h_r10_cb,
        "capt5_CA": h_capt5,
        "top1_CA": h_top1,
        "pooled_R10": {
            "A": {"k": sum_A, "n": sum_n, "p": a_p, "ci": [a_l, a_u]},
            "C": {"k": sum_C, "n": sum_n, "p": c_p, "ci": [c_l, c_u]},
            "delta": c_p - a_p,
        },
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
