"""Aggregate exp 19 5-seed run with --success-threshold 0.3 (st03 variant).
Output: pooled-Wilson + per-seed t CIs for ΔR@10, Δcap_t05, Δtop1.
Side-by-side comparison vs report-028 default-threshold run.
"""
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [17, 11, 23, 1, 2]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (p, center - half, center + half)


def load(prefix, seed):
    p = ROOT / f"reports/{prefix}_seed{seed}/phase34_results.json"
    return json.loads(p.read_text())


def collect(prefix):
    rows = []
    for s in SEEDS:
        d = load(prefix, s)
        a = d["results"]["baseline_static"][-1]
        b = d["results"]["phase3_reencode"][-1]
        c = d["results"]["phase3_phase4"][-1]
        rows.append({
            "seed": s, "n": a["n"],
            "A_top1": a["top1"], "A_topk": a["topk"], "A_capt5": a["cap_t_05"],
            "B_top1": b["top1"], "B_topk": b["topk"], "B_capt5": b["cap_t_05"],
            "C_top1": c["top1"], "C_topk": c["topk"], "C_capt5": c["cap_t_05"],
            "C_candidates": c.get("candidates_total", 0),
            "C_consolidations": c.get("consolidations", 0),
            "B_consolidations": b.get("consolidations", 0),
            "C_drift": c.get("codebook_drift_from_initial", 0.0),
            "B_drift": b.get("codebook_drift_from_initial", 0.0),
        })
    return rows


def perseed_t_ci(deltas, t_crit=2.776):
    n = len(deltas)
    m = mean(deltas)
    var = sum((d-m)**2 for d in deltas) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return m, m - t_crit*se, m + t_crit*se


def pooled_wilson_delta(rows, A_key, C_key):
    sum_n = sum(r["n"] for r in rows)
    sum_A = sum(round(r[A_key] * r["n"]) for r in rows)
    sum_C = sum(round(r[C_key] * r["n"]) for r in rows)
    a_p, a_l, a_u = wilson(sum_A, sum_n)
    c_p, c_l, c_u = wilson(sum_C, sum_n)
    return a_p, c_p, c_p - a_p, sum_n, (sum_A, sum_C), (a_l, a_u), (c_l, c_u)


def summarize(label, rows):
    print(f"\n{'='*100}\n  {label}\n{'='*100}")
    print(f"{'seed':>5} {'n':>4} | "
          f"{'A R@10':>7} {'B R@10':>7} {'C R@10':>7} {'ΔR@10':>8} {'C-B':>7} | "
          f"{'A capt5':>8} {'C capt5':>8} {'Δcapt5':>8} | "
          f"{'A top1':>7} {'C top1':>7} {'Δtop1':>7} | "
          f"{'cands':>5} {'cons':>5} {'drift':>9}")
    for r in rows:
        d_topk = r["C_topk"] - r["A_topk"]
        d_top1 = r["C_top1"] - r["A_top1"]
        d_capt5 = r["C_capt5"] - r["A_capt5"]
        c_minus_b = r["C_topk"] - r["B_topk"]
        print(f"{r['seed']:>5} {r['n']:>4} | "
              f"{r['A_topk']:>7.3f} {r['B_topk']:>7.3f} {r['C_topk']:>7.3f} "
              f"{d_topk:>+8.3f} {c_minus_b:>+7.3f} | "
              f"{r['A_capt5']:>8.3f} {r['C_capt5']:>8.3f} {d_capt5:>+8.3f} | "
              f"{r['A_top1']:>7.3f} {r['C_top1']:>7.3f} {d_top1:>+7.3f} | "
              f"{r['C_candidates']:>5d} {r['C_consolidations']:>5d} {r['C_drift']:>9.2e}")

    deltas_topk = [r["C_topk"] - r["A_topk"] for r in rows]
    deltas_capt5 = [r["C_capt5"] - r["A_capt5"] for r in rows]
    deltas_top1 = [r["C_top1"] - r["A_top1"] for r in rows]
    cb = [r["C_topk"] - r["B_topk"] for r in rows]

    def show(label, ds):
        m, lo, hi = perseed_t_ci(ds)
        signs = (sum(1 for d in ds if d > 0), sum(1 for d in ds if d == 0), sum(1 for d in ds if d < 0))
        print(f"  {label:<10} mean={m:>+7.4f}  per-seed-t 95% CI [{lo:>+.4f}, {hi:>+.4f}]  "
              f"signs: {signs[0]}+ {signs[1]}0 {signs[2]}-  "
              f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}")
    print()
    show("ΔR@10",   deltas_topk)
    show("Δcap_t05", deltas_capt5)
    show("Δtop1",    deltas_top1)
    show("C-B R@10", cb)

    # Pooled-Wilson on R@10
    a_p, c_p, dp, sum_n, (sa, sc), a_ci, c_ci = pooled_wilson_delta(rows, "A_topk", "C_topk")
    print(f"\n  Pooled R@10  A: {a_p:.4f} [{a_ci[0]:.4f}, {a_ci[1]:.4f}]  "
          f"C: {c_p:.4f} [{c_ci[0]:.4f}, {c_ci[1]:.4f}]  Δ_pooled = {dp:+.4f}  ({sa}/{sum_n} → {sc}/{sum_n})")

    print(f"\n  Mean Hebbian consolidations C: {mean(r['C_consolidations'] for r in rows):.1f}  "
          f"Mean drift: {mean(r['C_drift'] for r in rows):.2e}")
    print(f"  Mean Phase 4 candidates: {mean(r['C_candidates'] for r in rows):.1f}")
    print(f"  Hebbian fire rate: ~{mean(r['C_consolidations']/1500 for r in rows)*100:.2f}% of cues")

    return {
        "deltas_topk_per_seed": deltas_topk,
        "deltas_capt5_per_seed": deltas_capt5,
        "deltas_top1_per_seed": deltas_top1,
        "cb_topk_per_seed": cb,
        "pooled_delta_topk": dp,
        "pooled_n": sum_n,
        "mean_cons": mean(r["C_consolidations"] for r in rows),
        "mean_drift": mean(r["C_drift"] for r in rows),
        "mean_candidates": mean(r["C_candidates"] for r in rows),
    }


# Default-threshold (report 028)
default_rows = collect("phase34_integrated_hebbian")
default_summary = summarize("DEFAULT (success_threshold=0.5) — report 028 baseline", default_rows)

# st03 run (this report)
st03_rows = collect("phase34_integrated_hebbian_st03")
st03_summary = summarize("ST03 (success_threshold=0.3) — this run", st03_rows)

# Side-by-side at-a-glance
print("\n" + "=" * 100)
print("  Side-by-side: default vs st03 at final checkpoint")
print("=" * 100)
print(f"  {'metric':<22}  {'default (st=0.5)':>22}  {'st03 (st=0.3)':>22}  {'change':>10}")
print(f"  {'-'*22}  {'-'*22}  {'-'*22}  {'-'*10}")
print(f"  {'mean ΔR@10':<22}  {mean(default_summary['deltas_topk_per_seed']):>+22.4f}  "
      f"{mean(st03_summary['deltas_topk_per_seed']):>+22.4f}  "
      f"{mean(st03_summary['deltas_topk_per_seed'])-mean(default_summary['deltas_topk_per_seed']):>+10.4f}")
print(f"  {'mean Δcap_t05':<22}  {mean(default_summary['deltas_capt5_per_seed']):>+22.4f}  "
      f"{mean(st03_summary['deltas_capt5_per_seed']):>+22.4f}  "
      f"{mean(st03_summary['deltas_capt5_per_seed'])-mean(default_summary['deltas_capt5_per_seed']):>+10.4f}")
print(f"  {'mean Δtop1':<22}  {mean(default_summary['deltas_top1_per_seed']):>+22.4f}  "
      f"{mean(st03_summary['deltas_top1_per_seed']):>+22.4f}  "
      f"{mean(st03_summary['deltas_top1_per_seed'])-mean(default_summary['deltas_top1_per_seed']):>+10.4f}")
print(f"  {'mean cons C':<22}  {default_summary['mean_cons']:>22.1f}  {st03_summary['mean_cons']:>22.1f}  "
      f"{st03_summary['mean_cons']/max(default_summary['mean_cons'],0.1):>10.1f}x")
print(f"  {'mean drift C':<22}  {default_summary['mean_drift']:>22.2e}  {st03_summary['mean_drift']:>22.2e}  "
      f"{st03_summary['mean_drift']/max(default_summary['mean_drift'],1e-12):>10.1f}x")

# Trajectory comparison: R@10 mean by checkpoint, st03
print("\n" + "=" * 100)
print("  ST03 trajectory: mean R@10 by checkpoint (5 seeds)")
print("=" * 100)
print(f"{'cues':>6} {'A R@10':>9} {'B R@10':>9} {'C R@10':>9} {'Δ(C-A)':>9} {'Δ(C-B)':>9}")
for i in range(len(load('phase34_integrated_hebbian_st03', 17)["results"]["baseline_static"])):
    cues = load('phase34_integrated_hebbian_st03', 17)["results"]["baseline_static"][i]["cues_seen"]
    a = [load('phase34_integrated_hebbian_st03', s)["results"]["baseline_static"][i]["topk"] for s in SEEDS]
    b = [load('phase34_integrated_hebbian_st03', s)["results"]["phase3_reencode"][i]["topk"] for s in SEEDS]
    c = [load('phase34_integrated_hebbian_st03', s)["results"]["phase3_phase4"][i]["topk"] for s in SEEDS]
    print(f"{cues:>6} {mean(a):>9.4f} {mean(b):>9.4f} {mean(c):>9.4f} "
          f"{mean(c)-mean(a):>+9.4f} {mean(c)-mean(b):>+9.4f}")

# Save
out = ROOT / "reports/phase34_integrated_hebbian_st03_aggregate.json"
out.write_text(json.dumps({
    "default_per_seed": default_rows,
    "st03_per_seed": st03_rows,
    "default_summary": default_summary,
    "st03_summary": st03_summary,
}, indent=2))
print(f"\nwrote {out}")
