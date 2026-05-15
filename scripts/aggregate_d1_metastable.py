"""D1 aggregation: per-scale meta-stable rate across the 5-seed
Phase 4 canonical run, baseline vs phase4 conditions, with Wilson CIs.

Inputs: reports/phase4_rt0p85{,_seed1,_seed2,_seed11,_seed23}/phase4_unified_results.json
Output: stdout table + reports/d1_metastable_5seed.json
"""
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [("17", "phase4_rt0p85"),
         ("1",  "phase4_rt0p85_seed1"),
         ("2",  "phase4_rt0p85_seed2"),
         ("11", "phase4_rt0p85_seed11"),
         ("23", "phase4_rt0p85_seed23")]


def wilson(p, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (center - half, center + half)


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    by_scale = {2: {"baseline": [], "phase4": []},
                3: {"baseline": [], "phase4": []},
                4: {"baseline": [], "phase4": []}}
    final_n = None
    for seed, dirname in SEEDS:
        p = ROOT / "reports" / dirname / "phase4_unified_results.json"
        if not p.exists():
            print(f"  [skip] missing {p}")
            continue
        d = load(p)
        for cond in ("baseline", "phase4"):
            ckpts = d["results"][cond]
            final = ckpts[-1]
            if final_n is None:
                final_n = final["n"]
            for s in (2, 3, 4):
                rate = final["per_scale"][str(s)]["metastable_rate"]
                by_scale[s][cond].append((seed, rate, final["n"]))

    out = {"final_n_per_seed": final_n, "by_scale": {}}
    print(f"\nD1 — meta-stable rate at final checkpoint (per-seed n≈{final_n})")
    print(f"{'scale':>5} {'cond':>9} {'n_seeds':>7}  "
          f"{'mean':>8} {'std':>8}  per_seed_rates")
    for s in (2, 3, 4):
        for cond in ("baseline", "phase4"):
            rows = by_scale[s][cond]
            rates = [r for _, r, _ in rows]
            m = mean(rates) if rates else 0.0
            sd = pstdev(rates) if len(rates) > 1 else 0.0
            print(f"{s:>5} {cond:>9} {len(rates):>7}  "
                  f"{m:>8.4f} {sd:>8.4f}  "
                  + " ".join(f"s{sd_id}={r:.3f}" for sd_id, r, _ in rows))
            out["by_scale"].setdefault(str(s), {})[cond] = {
                "mean": m, "std": sd, "n_seeds": len(rates),
                "per_seed": [{"seed": sd_id, "rate": r, "n_eval": n}
                             for sd_id, r, n in rows],
            }

    # Delta per scale: phase4 - baseline, per seed
    print(f"\n{'scale':>5} {'delta_mean':>10} {'delta_std':>10}  per_seed_deltas")
    for s in (2, 3, 4):
        base = {sd: r for sd, r, _ in by_scale[s]["baseline"]}
        ph4 = {sd: r for sd, r, _ in by_scale[s]["phase4"]}
        deltas = [(sd, ph4[sd] - base[sd]) for sd in base if sd in ph4]
        ds = [d for _, d in deltas]
        m = mean(ds) if ds else 0.0
        sd = pstdev(ds) if len(ds) > 1 else 0.0
        print(f"{s:>5} {m:>10.4f} {sd:>10.4f}  "
              + " ".join(f"s{i}={d:+.3f}" for i, d in deltas))
        out["by_scale"][str(s)]["delta"] = {
            "mean": m, "std": sd,
            "per_seed": [{"seed": i, "delta": d} for i, d in deltas],
        }

    out_path = ROOT / "reports" / "d1_metastable_5seed.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
