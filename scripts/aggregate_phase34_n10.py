"""n=10 aggregation for exp 19 Phase 3+4 integration (st03 config).

Combines the original 5 seeds {17, 11, 23, 1, 2} from report 029 with the
5 new seeds {3, 5, 7, 13, 19} for a total n=10. Outputs:
- Per-seed deltas table
- 5-seed-original vs 5-seed-new vs n=10 combined summary
- Pooled-Wilson and per-seed-t 95% CIs
- Sign breakdown
"""
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_5 = [17, 11, 23, 1, 2]
NEW_5 = [3, 5, 7, 13, 19]
ALL_10 = ORIGINAL_5 + NEW_5


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (p, center - half, center + half)


def load_seed(seed):
    p = ROOT / f"reports/phase34_integrated_hebbian_st03_seed{seed}/phase34_results.json"
    return json.loads(p.read_text())


def deltas_for(seeds):
    rows = []
    for s in seeds:
        d = load_seed(s)
        a = d["results"]["baseline_static"][-1]
        b = d["results"]["phase3_reencode"][-1]
        c = d["results"]["phase3_phase4"][-1]
        rows.append({
            "seed": s, "n": a["n"],
            "A_topk": a["topk"], "B_topk": b["topk"], "C_topk": c["topk"],
            "A_top1": a["top1"], "B_top1": b["top1"], "C_top1": c["top1"],
            "A_capt5": a["cap_t_05"], "B_capt5": b["cap_t_05"], "C_capt5": c["cap_t_05"],
            "C_cands": c.get("candidates_total", 0),
            "C_cons": c.get("consolidations", 0),
            "C_drift": c.get("codebook_drift_from_initial", 0.0),
        })
    return rows


def t_ci(ds, dof_to_t={4: 2.776, 9: 2.262, 14: 2.145}):
    n = len(ds)
    m = mean(ds)
    v = sum((d-m)**2 for d in ds)/(n-1) if n > 1 else 0
    se = math.sqrt(v/n)
    t = dof_to_t.get(n-1, 1.96)
    return m, m - t*se, m + t*se


def report_set(label, rows):
    print(f"\n{'='*92}\n  {label} (n={len(rows)})\n{'='*92}")
    print(f"{'seed':>5} {'n':>4} | {'A R@10':>7} {'C R@10':>7} {'ΔR@10':>8} {'Δtop1':>7} {'Δcapt5':>8} | "
          f"{'cands':>5} {'cons':>5} {'drift':>9}")
    for r in rows:
        dR = r["C_topk"] - r["A_topk"]
        dT = r["C_top1"] - r["A_top1"]
        dC = r["C_capt5"] - r["A_capt5"]
        print(f"{r['seed']:>5} {r['n']:>4} | "
              f"{r['A_topk']:>7.3f} {r['C_topk']:>7.3f} {dR:>+8.3f} "
              f"{dT:>+7.3f} {dC:>+8.3f} | "
              f"{r['C_cands']:>5} {r['C_cons']:>5} {r['C_drift']:>9.2e}")

    dR = [r["C_topk"] - r["A_topk"] for r in rows]
    dT = [r["C_top1"] - r["A_top1"] for r in rows]
    dC = [r["C_capt5"] - r["A_capt5"] for r in rows]
    cb = [r["C_topk"] - r["B_topk"] for r in rows]

    def line(name, ds):
        m, lo, hi = t_ci(ds)
        signs = (sum(1 for d in ds if d > 0), sum(1 for d in ds if d == 0), sum(1 for d in ds if d < 0))
        excl = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
        print(f"  {name:<10} mean={m:>+.4f}  std={pstdev(ds):.4f}  95%CI [{lo:>+.4f}, {hi:>+.4f}]  "
              f"signs {signs[0]}+/{signs[1]}0/{signs[2]}-  {excl}")

    print()
    line("ΔR@10",   dR)
    line("Δcap_t05", dC)
    line("Δtop1",    dT)
    line("C-B R@10", cb)

    # Pooled-Wilson
    sum_n = sum(r["n"] for r in rows)
    sum_A = sum(round(r["A_topk"]*r["n"]) for r in rows)
    sum_C = sum(round(r["C_topk"]*r["n"]) for r in rows)
    a_p, a_l, a_u = wilson(sum_A, sum_n)
    c_p, c_l, c_u = wilson(sum_C, sum_n)
    print(f"\n  Pooled R@10  A: {a_p:.4f} [{a_l:.4f}, {a_u:.4f}]  "
          f"C: {c_p:.4f} [{c_l:.4f}, {c_u:.4f}]  Δ={c_p-a_p:+.4f} ({sum_A}/{sum_n} → {sum_C}/{sum_n})")

    return {
        "dR10_per_seed": dR, "dtop1_per_seed": dT, "dcapt5_per_seed": dC,
        "cb_R10_per_seed": cb,
        "pooled_R10_delta": c_p - a_p, "pooled_A": sum_A, "pooled_C": sum_C, "pooled_n": sum_n,
    }


orig = deltas_for(ORIGINAL_5)
new = deltas_for(NEW_5)
all10 = deltas_for(ALL_10)

summary_orig = report_set("ORIGINAL 5 (report 029 seeds)", orig)
summary_new = report_set("NEW 5 (this run)", new)
summary_all = report_set("ALL 10 (n=10 combined)", all10)

# Side-by-side
print(f"\n{'='*92}\n  Comparison: original 5 vs new 5 vs combined 10\n{'='*92}")
print(f"  {'metric':<14}  {'orig 5':>14}  {'new 5':>14}  {'all 10':>14}")
print(f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")
print(f"  {'mean ΔR@10':<14}  {mean(summary_orig['dR10_per_seed']):>+14.4f}  "
      f"{mean(summary_new['dR10_per_seed']):>+14.4f}  {mean(summary_all['dR10_per_seed']):>+14.4f}")
print(f"  {'mean Δtop1':<14}  {mean(summary_orig['dtop1_per_seed']):>+14.4f}  "
      f"{mean(summary_new['dtop1_per_seed']):>+14.4f}  {mean(summary_all['dtop1_per_seed']):>+14.4f}")
print(f"  {'mean Δcap_t05':<14}  {mean(summary_orig['dcapt5_per_seed']):>+14.4f}  "
      f"{mean(summary_new['dcapt5_per_seed']):>+14.4f}  {mean(summary_all['dcapt5_per_seed']):>+14.4f}")
print(f"  {'pooled ΔR@10':<14}  {summary_orig['pooled_R10_delta']:>+14.4f}  "
      f"{summary_new['pooled_R10_delta']:>+14.4f}  {summary_all['pooled_R10_delta']:>+14.4f}")

out = ROOT / "reports/phase34_integrated_hebbian_st03_n10_aggregate.json"
out.write_text(json.dumps({
    "original_5": summary_orig, "new_5": summary_new, "all_10": summary_all,
    "all_10_seeds": ALL_10,
}, indent=2))
print(f"\nwrote {out}")
