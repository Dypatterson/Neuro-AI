#!/usr/bin/env python
"""Parallel landscape-size diagnostic via per-(L, seed) fan-out (Report 117).

Cells are order-independent (verified 2026-05-30: arm-A recall is identical whether
a seed runs fresh or after others), so fanning each (L, seed) out to its own CUDA
subprocess is BYTE-IDENTICAL to a single-process run — it just uses the otherwise-
idle GPU (the run is sync-bound, ~0.5 GB, not memory/compute-bound, so a bigger GPU
alone does nothing; PARALLELISM is the lever). Only arms A (consolidated, real) and
C (frozen, real) are run — all the pre-committed read needs (sigma_A, sigma_C,
mean(A-C)). Follows the project's blessed fan-out pattern (colab_c3_followup_v3):
parent stays CPU-only, workers each own a CUDA context, staggered launch.

Orchestrator (parent — never imports torch / touches CUDA):
  python scripts/landscape_sweep_parallel.py --device cuda \
      --landscapes 64,256,512 --seeds 0-9 --out reports/landscape_2026-05-30

Smoke (tiny synthetic, seconds, CPU-ok) — validates the whole pipeline:
  python scripts/landscape_sweep_parallel.py --smoke --device cpu --out /tmp/lsmoke

Worker (one (L, seed); launched by the orchestrator — do not call directly):
  python scripts/landscape_sweep_parallel.py --worker --L 256 --seed 3 \
      --device cuda --out-file <path>
"""
import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Cell params passed to c3._run_single_seed_condition (mirrors gate0_frame_a._cell).
REAL_OP = dict(D=4096, window_size=8, n_test_windows=512, n_train_windows=2048,
               beta=10.0, k=5, n_consolidation_events=1000, alpha_anti=0.01,
               repulsion_step_size=0.05, lr_pull=0.1, lr_push=0.05)
SMOKE_OP = dict(D=256, window_size=4, n_test_windows=24, n_train_windows=48,
                beta=10.0, k=5, n_consolidation_events=40, alpha_anti=0.01,
                repulsion_step_size=0.05, lr_pull=0.1, lr_push=0.05)


def run_worker(L, seed, device, out_file, smoke):
    """One (L, seed): arm A (consolidated, real) + arm C (frozen, real)."""
    for p in (REPO / "experiments", REPO / "src"):
        sys.path.insert(0, str(p))
    import c3_phase3_exit_criterion as c3
    import gate0_frame_a as g
    op = dict(SMOKE_OP if smoke else REAL_OP)
    if smoke:
        corpus = None
        vocab_size = 40
    else:
        corpus = c3._load_wikitext_corpus(
            repo_root=REPO, wikitext_name="wikitext-2-raw-v1", vocab_cap=1000)
        vocab_size = corpus.vocab_size

    def cell(standard_mode):
        row = c3._run_single_seed_condition(
            seed=seed, is_control=False, theta_prime_mode="default",
            standard_mode=standard_mode, control_mode="shuffled-token",
            landscape_size=L, vocab_size=vocab_size, device=device,
            repo_root=REPO, wikitext_corpus=corpus,
            use_context_residual=False, lr_cr=0.1, use_pull_push=True,
            world="real", **op)
        return g._overall_recall(row)

    Path(out_file).write_text(json.dumps(
        {"L": L, "seed": seed,
         "recall_A": cell("consolidated"), "recall_C": cell("frozen")}))


def _read(rows, out_dir):
    """sigma_A(L), sigma_C(L), mean(A-C)(L) + the pre-committed verdict.

    Identical thresholds to scripts/landscape_sweep_read.py / the precommit
    (notes/notes/2026-05-30-landscape-sweep-diagnostic-precommit.md).
    """
    table = {}
    print("\n   L | sigma_A | sigma_C | mean(A-C) | n")
    print("  ----+---------+---------+-----------+--")
    for L in sorted(rows):
        A = [rows[L][s]["recall_A"] for s in sorted(rows[L])]
        C = [rows[L][s]["recall_C"] for s in sorted(rows[L])]
        lift = [a - c for a, c in zip(A, C)]
        table[L] = (st.stdev(A), st.stdev(C), st.mean(lift))
        print(f"  {L:>4} |  {table[L][0]:.3f}  |  {table[L][1]:.3f}  |  "
              f"{table[L][2]:+.3f}   | {len(A)}")
    Path(out_dir, "landscape_sweep_summary.json").write_text(json.dumps(
        {str(L): {"sigma_A": table[L][0], "sigma_C": table[L][1],
                  "mean_lift_A_minus_C": table[L][2]} for L in table}, indent=1))

    top = max(table)
    sA, _, ml = table[top]
    if any(table[L][2] < 0.015 for L in table):
        v = "CAPACITY-WALL (signal collapsed; bigger L not usable even if sigma dropped)"
    elif sA <= 0.10 and ml >= 0.020:
        v = "VARIANCE-REDUCIBLE -> propose a powered run at the best L (op-point sign-off)"
    elif sA >= 0.13:
        v = "VARIANCE-IRREDUCIBLE -> Frame B mechanism / op-point rethink"
    else:
        v = "PARTIAL -> read the sigma_A(L) curve; push L to 1024 or modest L+n"
    base = table.get(64)
    if base is not None and not (0.13 <= base[0] <= 0.18):
        print(f"\n  !! WARNING sigma_A(64)={base[0]:.3f} did NOT reproduce the recovered "
              "~0.157 (expect 0.13-0.18) -> environment/repro problem, STOP and re-derive.")
    print(f"\n  VERDICT: {v}")


def orchestrate(landscapes, seeds, device, out_dir, smoke, stagger, poll, max_concurrent):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(L, s) for L in landscapes for s in seeds]
    procs, pending = [], list(jobs)

    def _launch(L, s):
        of = out / f"cell_L{L:04d}_s{s}.json"
        lg = out / f"cell_L{L:04d}_s{s}.log"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--L", str(L), "--seed", str(s), "--device", device,
               "--out-file", str(of)] + (["--smoke"] if smoke else [])
        fh = open(lg, "w")
        procs.append([L, s, of, lg, subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT), fh])
        print(f"launched L={L} seed={s} "
              f"({len(procs)}/{len(jobs)})", flush=True)
        time.sleep(stagger)  # stagger -> avoid CUDA-init races

    # Windowed launch: keep at most max_concurrent workers alive at once.
    while pending or any(p[4].poll() is None for p in procs):
        alive = sum(p[4].poll() is None for p in procs)
        while pending and alive < max_concurrent:
            _launch(*pending.pop(0))
            alive += 1
        done = sum(p[4].poll() is not None for p in procs)
        print(f"  {done}/{len(jobs)} done, {alive} running, "
              f"{len(pending)} queued...", flush=True)
        time.sleep(poll)

    rows, fail = {}, []
    for L, s, of, lg, p, fh in procs:
        fh.close()
        if p.returncode != 0 or not of.exists():
            fail.append((L, s, lg))
        else:
            rows.setdefault(L, {})[s] = json.loads(of.read_text())
    if fail:
        for L, s, lg in fail:
            print(f"\n!! FAILED L={L} seed={s}; tail of {lg}:")
            print("\n".join(Path(lg).read_text().splitlines()[-20:]))
        raise SystemExit(f"{len(fail)} cell(s) failed — fix before trusting the read.")
    _read(rows, out)


def _parse_seeds(s):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--L", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out-file")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/landscape_2026-05-30")
    ap.add_argument("--landscapes", default="64,256,512")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--stagger", type=float, default=1.0)
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--max-concurrent", type=int, default=16,
                    help="max worker subprocesses alive at once (cap CPU oversub).")
    a = ap.parse_args()
    if a.worker:
        run_worker(a.L, a.seed, a.device, a.out_file, a.smoke)
        return
    lands = [16, 24] if a.smoke else [int(x) for x in a.landscapes.split(",")]
    seeds = [0, 1] if a.smoke else _parse_seeds(a.seeds)
    orchestrate(lands, seeds, a.device, a.out, a.smoke,
                stagger=(0.2 if a.smoke else a.stagger),
                poll=(2.0 if a.smoke else a.poll),
                max_concurrent=(4 if a.smoke else a.max_concurrent))


if __name__ == "__main__":
    main()
