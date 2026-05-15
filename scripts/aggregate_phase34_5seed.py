"""5-seed aggregation for exp 19 Phase 3+4 integration sweep (Hebbian, drift via online updates).

Headline (per phase-4-unified-design.md:276-282): ΔR@10 (topk field with decode_k=10) AND Δcap_t05
at the final checkpoint, condition C (phase3_phase4) minus condition A (baseline_static).
Reports 5-seed Wilson CIs (pooled), per-seed deltas, and diagnostic counts.
"""
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [17, 11, 23, 1, 2]
CHECKPOINTS = [300, 600, 900, 1200, 1500]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (p, center - half, center + half)


def load_seed(seed):
    p = ROOT / f"reports/phase34_integrated_hebbian_seed{seed}/phase34_results.json"
    return json.loads(p.read_text())


def main():
    # Per-seed final checkpoint
    rows = []  # one row per seed
    for s in SEEDS:
        d = load_seed(s)
        a = d["results"]["baseline_static"][-1]
        b = d["results"]["phase3_reencode"][-1]
        c = d["results"]["phase3_phase4"][-1]
        rows.append({
            "seed": s,
            "n": a["n"],
            "A_top1": a["top1"], "A_topk": a["topk"], "A_capt5": a["cap_t_05"],
            "B_top1": b["top1"], "B_topk": b["topk"], "B_capt5": b["cap_t_05"],
            "C_top1": c["top1"], "C_topk": c["topk"], "C_capt5": c["cap_t_05"],
            "C_candidates": c.get("candidates_total", 0),
            "C_consolidations": c.get("consolidations", 0),
            "B_consolidations": b.get("consolidations", 0),
            "C_drift": c.get("codebook_drift_from_initial", 0.0),
            "B_drift": b.get("codebook_drift_from_initial", 0.0),
        })

    # Per-seed deltas
    print("=" * 100)
    print("Per-seed deltas at final checkpoint (cues_seen=1500), C (phase3_phase4) − A (baseline_static)")
    print("=" * 100)
    print(f"{'seed':>5} {'n':>4} | "
          f"{'A_R@10':>7} {'B_R@10':>7} {'C_R@10':>7} {'ΔR@10':>8} | "
          f"{'A_top1':>7} {'C_top1':>7} {'Δtop1':>7} | "
          f"{'A_capt5':>8} {'C_capt5':>8} {'Δcapt5':>8} | "
          f"{'cands':>5} {'cons':>4} {'drift':>9}")
    for r in rows:
        d_topk = r["C_topk"] - r["A_topk"]
        d_top1 = r["C_top1"] - r["A_top1"]
        d_capt5 = r["C_capt5"] - r["A_capt5"]
        print(f"{r['seed']:>5} {r['n']:>4} | "
              f"{r['A_topk']:>7.3f} {r['B_topk']:>7.3f} {r['C_topk']:>7.3f} "
              f"{d_topk:>+8.3f} | "
              f"{r['A_top1']:>7.3f} {r['C_top1']:>7.3f} {d_top1:>+7.3f} | "
              f"{r['A_capt5']:>8.3f} {r['C_capt5']:>8.3f} {d_capt5:>+8.3f} | "
              f"{r['C_candidates']:>5d} {r['C_consolidations']:>4d} {r['C_drift']:>9.2e}")

    # Aggregate stats
    print("\n" + "=" * 100)
    print("5-seed summary")
    print("=" * 100)

    def agg(label, deltas):
        m = mean(deltas)
        sd = pstdev(deltas)
        n_pos = sum(1 for d in deltas if d > 0)
        n_zero = sum(1 for d in deltas if d == 0)
        n_neg = sum(1 for d in deltas if d < 0)
        print(f"  {label:<10} mean={m:>+7.4f}  std={sd:.4f}  "
              f"signs: {n_pos}+ / {n_zero}0 / {n_neg}-")

    agg("ΔR@10",   [r["C_topk"]  - r["A_topk"]  for r in rows])
    agg("Δtop1",   [r["C_top1"]  - r["A_top1"]  for r in rows])
    agg("Δcap_t05",[r["C_capt5"] - r["A_capt5"] for r in rows])

    # Pooled Wilson CIs across seeds (paired delta via summed counts)
    print("\n  Pooled (sum across seeds) — paired Δ with Wilson CI on the unpaired counts:")
    sum_n = sum(r["n"] for r in rows)
    sum_A_topk = sum(round(r["A_topk"] * r["n"]) for r in rows)
    sum_C_topk = sum(round(r["C_topk"] * r["n"]) for r in rows)
    sum_A_capt5 = sum(round(r["A_capt5"] * r["n"]) for r in rows)
    sum_C_capt5 = sum(round(r["C_capt5"] * r["n"]) for r in rows)
    a_p, a_l, a_u = wilson(sum_A_topk, sum_n)
    c_p, c_l, c_u = wilson(sum_C_topk, sum_n)
    print(f"  R@10  A: {a_p:.4f} [{a_l:.4f}, {a_u:.4f}]  C: {c_p:.4f} [{c_l:.4f}, {c_u:.4f}]  "
          f"Δ={c_p - a_p:+.4f} ({sum_A_topk}/{sum_n} → {sum_C_topk}/{sum_n})")
    a_p, a_l, a_u = wilson(sum_A_capt5, sum_n)
    c_p, c_l, c_u = wilson(sum_C_capt5, sum_n)
    print(f"  capt5 A: {a_p:.4f} [{a_l:.4f}, {a_u:.4f}]  C: {c_p:.4f} [{c_l:.4f}, {c_u:.4f}]  "
          f"Δ={c_p - a_p:+.4f} ({sum_A_capt5}/{sum_n} → {sum_C_capt5}/{sum_n})")

    # Per-seed Δ_R@10 with 95% normal-approx CI from per-seed deltas
    deltas_topk = [r["C_topk"] - r["A_topk"] for r in rows]
    n_seeds = len(deltas_topk)
    mean_d = mean(deltas_topk)
    if n_seeds > 1:
        # sample SE
        var = sum((d - mean_d)**2 for d in deltas_topk) / (n_seeds - 1)
        se = math.sqrt(var / n_seeds)
        # t(4 df, 0.975) = 2.776
        t_crit = 2.776
        ci_lo, ci_hi = mean_d - t_crit*se, mean_d + t_crit*se
        print(f"\n  ΔR@10 per-seed: mean={mean_d:+.4f}  95% CI (t, 4df) [{ci_lo:+.4f}, {ci_hi:+.4f}]"
              f"   CI {'EXCLUDES' if ci_lo > 0 or ci_hi < 0 else 'INCLUDES'} 0")
        deltas_capt5 = [r["C_capt5"] - r["A_capt5"] for r in rows]
        mean_d2 = mean(deltas_capt5)
        var2 = sum((d - mean_d2)**2 for d in deltas_capt5) / (n_seeds - 1)
        se2 = math.sqrt(var2 / n_seeds)
        ci_lo2, ci_hi2 = mean_d2 - t_crit*se2, mean_d2 + t_crit*se2
        print(f"  Δcapt5 per-seed: mean={mean_d2:+.4f}  95% CI (t, 4df) [{ci_lo2:+.4f}, {ci_hi2:+.4f}]"
              f"   CI {'EXCLUDES' if ci_lo2 > 0 or ci_hi2 < 0 else 'INCLUDES'} 0")

    # Diagnostic: candidates and consolidations across seeds
    print("\n" + "=" * 100)
    print("Diagnostic: candidate counts, Hebbian firing rate, codebook drift")
    print("=" * 100)
    print(f"  C_candidates_total: mean={mean(r['C_candidates'] for r in rows):.1f}  "
          f"per-seed={[r['C_candidates'] for r in rows]}")
    print(f"  C_consolidations:   mean={mean(r['C_consolidations'] for r in rows):.1f}  "
          f"per-seed={[r['C_consolidations'] for r in rows]}")
    drifts_str = [f"{r['C_drift']:.2e}" for r in rows]
    print(f"  C_codebook_drift:   mean={mean(r['C_drift'] for r in rows):.2e}  "
          f"per-seed={drifts_str}")
    print(f"  Hebbian fire rate: ~{mean(r['C_consolidations']/1500 for r in rows)*100:.2f}% of cues")

    # R@10 trajectory by checkpoint
    print("\n" + "=" * 100)
    print("R@10 trajectory by checkpoint (mean across 5 seeds, Δ = C − A)")
    print("=" * 100)
    print(f"{'cues':>6} {'A_R@10':>9} {'B_R@10':>9} {'C_R@10':>9} {'ΔR@10(C-A)':>12}")
    for i in range(len(load_seed(17)["results"]["baseline_static"])):
        vals_a = [load_seed(s)["results"]["baseline_static"][i]["topk"] for s in SEEDS]
        vals_b = [load_seed(s)["results"]["phase3_reencode"][i]["topk"] for s in SEEDS]
        vals_c = [load_seed(s)["results"]["phase3_phase4"][i]["topk"] for s in SEEDS]
        cues = load_seed(17)["results"]["baseline_static"][i]["cues_seen"]
        print(f"{cues:>6} {mean(vals_a):>9.4f} {mean(vals_b):>9.4f} {mean(vals_c):>9.4f} "
              f"{mean(vals_c) - mean(vals_a):>+12.4f}")

    # Save aggregate to JSON
    out = ROOT / "reports/phase34_integrated_hebbian_5seed_aggregate.json"
    out.write_text(json.dumps({
        "per_seed_final": rows,
        "delta_r10_per_seed_mean": mean_d,
        "delta_r10_per_seed_ci_t4df_95": [ci_lo, ci_hi],
        "delta_capt5_per_seed_mean": mean_d2,
        "delta_capt5_per_seed_ci_t4df_95": [ci_lo2, ci_hi2],
        "delta_r10_pooled": (sum_C_topk - sum_A_topk) / sum_n,
        "n_total": sum_n,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
