import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SEEDS = [17, 11, 23, 1, 2, 3, 5, 7, 13, 29]
GAINS = [0.0, 1.0, 2.0, 4.0]
N_CUES = 100
BETA = 10.0
GAMMA = 0.5
K_MAIN = 1
BINDING_NOISE_STD = 0.05
CONTENT_DISTORTION = 0.6
FORMULATION = "per_pattern"
CUE_SEED = 117
MAGNITUDE_FLOOR = 5.5e-3
RUN_TAG = "phase5_log_prior_n10_gain_sweep_20260524"
GIT_REF = "codex/phase5-log-prior-spike"

print("Active phase: 5")
print(
    "Headline metric per notes/emergent-codebook/phase-5-unified-design.md:"
    "269-292: DeltaE = E_content-prior - E_role-prior; mean over seeds/cues "
    "with 95% CI."
)
print(
    "C-first diagnostic note per notes/emergent-codebook/"
    "phase-5-unified-design.md:258-267: exploratory Phase 5' diagnostic, "
    "not a graduation retune."
)
print(
    "Required controls per notes/emergent-codebook/phase-5-unified-design.md:"
    "296-304: random-schema branches, K=1, no-prior, no-schema-store."
)
print(
    "This confirmation preserves random-prior and K=1 readouts; "
    "no-prior/no-schema-store are not run here, so no graduation claim "
    "follows from this notebook."
)
print(
    "Last verified result: Report 059 local smoke. Why now: confirm gain "
    "{0,1,2,4} at locked beta=10, gamma=0.5, K=1, "
    "content_distortion=0.6, binding_noise_std=0.05 across n=10 seeds."
)
print()


def run_cmd(cmd, *, cwd=None, env=None, check=True, max_tail=4000):
    cmd = [str(x) for x in cmd]
    print("$ " + " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd, cwd=cwd, env=env, text=True, capture_output=True
    )
    if result.stdout:
        out = (
            result.stdout
            if len(result.stdout) <= max_tail
            else result.stdout[-max_tail:]
        )
        print(out)
    if result.stderr:
        err = (
            result.stderr
            if len(result.stderr) <= max_tail
            else result.stderr[-max_tail:]
        )
        print(err)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {' '.join(cmd)}"
        )
    return result


content_root = Path("/content")
repo = content_root / "Neuro-AI"
if repo.exists() and (repo / ".git").exists():
    print(f"Using existing clone at {repo}")
else:
    if repo.exists():
        shutil.rmtree(repo)
    run_cmd(
        ["git", "clone", "https://github.com/Dypatterson/Neuro-AI.git", str(repo)],
        cwd=content_root,
    )
run_cmd(["git", "checkout", GIT_REF], cwd=repo)
commit = run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=repo).stdout.strip()
branch = run_cmd(["git", "branch", "--show-current"], cwd=repo).stdout.strip()
print(f"Checked out branch={branch!r} commit={commit}")

markers = [
    (
        "--log-prior-sweep",
        "scripts/phase5_frozen_snapshot_audit.py",
        "log-prior sweep CLI",
    ),
    (
        "log_prior_gain",
        "experiments/40_phase5_branching.py",
        "branch-local log-prior gain",
    ),
    (
        "test_run_branched_log_prior_gain_zero_matches_default",
        "tests/test_phase5_branching.py",
        "log-prior regression tests",
    ),
    (
        "Report 059",
        "reports/059_phase5_log_prior_spike_local_smoke.md",
        "local smoke report",
    ),
]
for marker, rel, label in markers:
    text = (repo / rel).read_text(errors="replace")
    if marker not in text:
        raise RuntimeError(f"missing marker {marker!r} in {rel}")
    print(f"[OK] {label}: {marker} in {rel}")

os.chdir(repo)

drive_root = Path("/content/drive/MyDrive")
if drive_root.exists():
    print("Google Drive already mounted at /content/drive")
else:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
drive_results = drive_root / "neuro-ai" / "results"
drive_results.mkdir(parents=True, exist_ok=True)
print("Drive results root:", drive_results)

gpu = run_cmd(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
    cwd=repo,
    check=False,
)
if gpu.returncode != 0:
    raise RuntimeError(
        "No CUDA GPU is visible. Set Runtime -> Change runtime type -> "
        "T4/L4/A100 GPU and rerun."
    )

snapshot_paths = {}
missing = []
for seed in SEEDS:
    path = (
        drive_results
        / f"phase5_headline_substrate_seed{seed}"
        / "snapshots"
        / "phase3_phase4_w4_step1800.pt"
    )
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        snapshot_paths[seed] = path
        print(f"seed {seed:>3}: snapshot OK ({size_mb:.1f} MB) {path}")
    else:
        missing.append(seed)
        print(f"seed {seed:>3}: MISSING {path}")
if missing:
    raise RuntimeError(
        f"Missing {len(missing)} snapshots: {missing}. "
        "Stop before interpreting Path C."
    )

local_root = repo / "reports" / RUN_TAG
local_log_root = local_root / "logs"
drive_root = drive_results / RUN_TAG
drive_log_root = drive_root / "logs"
for directory in (local_root, local_log_root, drive_root, drive_log_root):
    directory.mkdir(parents=True, exist_ok=True)

expected_config = {
    "beta": BETA,
    "k_main": K_MAIN,
    "gamma": GAMMA,
    "gains": GAINS,
    "n_cues": N_CUES,
    "binding_noise_std": BINDING_NOISE_STD,
    "content_distortion": CONTENT_DISTORTION,
    "formulation": FORMULATION,
    "cue_seed": CUE_SEED,
    "magnitude_floor": MAGNITUDE_FLOOR,
}


def close_float(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def validate_log_prior_json(path):
    path = Path(path)
    data = json.loads(path.read_text())
    sweep = data.get("log_prior_sweep")
    if not isinstance(sweep, dict):
        raise RuntimeError(f"{path} missing log_prior_sweep")
    cells = sweep.get("cells")
    if not isinstance(cells, list) or len(cells) != len(GAINS):
        got = 0 if cells is None else len(cells)
        raise RuntimeError(f"{path} has {got} gain cells; expected {len(GAINS)}")
    got_gains = [float(c.get("log_prior_gain")) for c in cells]
    if got_gains != GAINS:
        raise RuntimeError(f"{path} gains {got_gains}; expected {GAINS}")
    cfg = sweep.get("config", {})
    for key, expected in expected_config.items():
        got = cfg.get(key)
        if isinstance(expected, list):
            if [float(x) for x in got] != expected:
                raise RuntimeError(
                    f"{path} config {key}={got}; expected {expected}"
                )
        elif isinstance(expected, float):
            if not close_float(got, expected):
                raise RuntimeError(
                    f"{path} config {key}={got}; expected {expected}"
                )
        elif got != expected:
            raise RuntimeError(f"{path} config {key}={got}; expected {expected}")
    return data


def copy_complete(local_path, drive_path):
    local_path = Path(local_path)
    drive_path = Path(drive_path)
    drive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = drive_path.with_name(drive_path.name + ".partial")
    shutil.copy2(local_path, tmp)
    os.replace(tmp, drive_path)
    return drive_path


def tail(path, n=80):
    lines = Path(path).read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


env = {**os.environ, "PYTHONPATH": str(repo / "src")}
per_seed_jsons = {}
start = time.time()
for idx, seed in enumerate(SEEDS, start=1):
    drive_json = drive_root / f"seed{seed}.json"
    drive_log = drive_log_root / f"seed{seed}.log"
    if drive_json.exists():
        data = validate_log_prior_json(drive_json)
        per_seed_jsons[seed] = drive_json
        cells = data["log_prior_sweep"]["cells"]
        preview = "  ".join(
            f"g{int(c['log_prior_gain'])}: {c['mean_delta_e_raw']:+.5f}"
            for c in cells
        )
        print(f"[{idx}/{len(SEEDS)}] seed {seed}: resume valid - {preview}")
        continue

    local_json = local_root / f"seed{seed}.json"
    local_log = local_log_root / f"seed{seed}.log"
    cmd = [
        sys.executable,
        "scripts/phase5_frozen_snapshot_audit.py",
        "--snapshot",
        str(snapshot_paths[seed]),
        "--output",
        str(local_json),
        "--log-prior-sweep",
        "--log-prior-gains",
        ",".join(str(int(g)) for g in GAINS),
        "--log-prior-beta",
        str(BETA),
        "--headline-k-main",
        str(K_MAIN),
        "--headline-gamma",
        str(GAMMA),
        "--binding-noise-std",
        str(BINDING_NOISE_STD),
        "--content-distortion",
        str(CONTENT_DISTORTION),
        "--headline-formulation",
        FORMULATION,
        "--headline-cue-seed",
        str(CUE_SEED),
        "--headline-n-cues",
        str(N_CUES),
        "--device",
        "cuda",
    ]
    print(f"\n[{idx}/{len(SEEDS)}] running seed {seed} on {snapshot_paths[seed]}")
    t0 = time.time()
    with local_log.open("w") as logf:
        result = subprocess.run(
            cmd, cwd=repo, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True
        )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(tail(local_log))
        raise RuntimeError(f"seed {seed} failed rc={result.returncode}; see {local_log}")
    data = validate_log_prior_json(local_json)
    copy_complete(local_json, drive_json)
    copy_complete(local_log, drive_log)
    validate_log_prior_json(drive_json)
    per_seed_jsons[seed] = drive_json
    cells = data["log_prior_sweep"]["cells"]
    preview = "  ".join(
        f"g{int(c['log_prior_gain'])}: {c['mean_delta_e_raw']:+.5f}"
        for c in cells
    )
    print(f"seed {seed} done in {elapsed/60:.1f} min - {preview}")

print(
    f"\nPer-seed sweeps available for {len(per_seed_jsons)}/{len(SEEDS)} seeds "
    f"in {(time.time() - start)/60:.1f} min this cell."
)
if set(per_seed_jsons) != set(SEEDS):
    raise RuntimeError(f"Expected all seeds {SEEDS}; got {sorted(per_seed_jsons)}")


def t_critical_95(df):
    table = {
        1: 12.71,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df < 1:
        return float("nan")
    if df > 30:
        return 1.96
    for key in sorted(table):
        if key > df:
            return table[key]
    return 2.0


def ci_t(vals):
    n = len(vals)
    if n == 0:
        return [float("nan"), float("nan")]
    if n == 1:
        return [vals[0], vals[0]]
    avg = statistics.mean(vals)
    sd = statistics.stdev(vals)
    half = t_critical_95(n - 1) * sd / math.sqrt(n)
    return [avg - half, avg + half]


def bootstrap_ci(vals, seed=20260524, n_boot=20000):
    vals = list(vals)
    n = len(vals)
    if n == 0:
        return [float("nan"), float("nan")]
    if n == 1:
        return [vals[0], vals[0]]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return [boots[int(0.025 * (n_boot - 1))], boots[int(0.975 * (n_boot - 1))]]


def mean(vals):
    return statistics.mean(vals) if vals else float("nan")


per_seed = {}
by_seed_gain = {}
for seed in SEEDS:
    data = validate_log_prior_json(per_seed_jsons[seed])
    sweep = data["log_prior_sweep"]
    geom = data.get("geometry", {})
    cells_by_gain = {float(c["log_prior_gain"]): c for c in sweep["cells"]}
    by_seed_gain[seed] = cells_by_gain
    per_seed[str(seed)] = {
        "source_json": str(per_seed_jsons[seed]),
        "snapshot": sweep.get("snapshot"),
        "n_atoms": sweep.get("n_atoms"),
        "dim": sweep.get("dim"),
        "coverage_lambda": sweep.get("coverage_lambda"),
        "bias_cv": geom.get("bias_cv"),
        "gains": {},
    }
    for gain in GAINS:
        cell = cells_by_gain[gain]
        per_seed[str(seed)]["gains"][str(int(gain))] = {
            "n_pairs": cell["n_pairs"],
            "mean_delta_e_raw": cell["mean_delta_e_raw"],
            "mean_delta_e_step3": cell["mean_delta_e_step3"],
            "frac_positive_raw": cell["frac_positive_raw"],
            "frac_role_lt_content_lt_random": cell[
                "frac_role_lt_content_lt_random"
            ],
            "frac_role_lt_content": cell["frac_role_lt_content"],
            "frac_random_lowest": cell["frac_random_lowest"],
            "role_target_basin_hit_role": cell["per_condition_basin_hit_rate"][
                "role"
            ],
            "role_target_rank_role": cell[
                "per_condition_mean_role_target_rank"
            ]["role"],
        }

cross_seed = {}
for gain in GAINS:
    dE = [by_seed_gain[s][gain]["mean_delta_e_raw"] for s in SEEDS]
    dE_step3 = [by_seed_gain[s][gain]["mean_delta_e_step3"] for s in SEEDS]
    frac_pos = [by_seed_gain[s][gain]["frac_positive_raw"] for s in SEEDS]
    role_lt_content = [
        by_seed_gain[s][gain]["frac_role_lt_content"] for s in SEEDS
    ]
    role_lt_content_lt_random = [
        by_seed_gain[s][gain]["frac_role_lt_content_lt_random"] for s in SEEDS
    ]
    random_lowest = [by_seed_gain[s][gain]["frac_random_lowest"] for s in SEEDS]
    hit_role = [
        by_seed_gain[s][gain]["per_condition_basin_hit_rate"]["role"]
        for s in SEEDS
    ]
    hit_content = [
        by_seed_gain[s][gain]["per_condition_basin_hit_rate"]["content"]
        for s in SEEDS
    ]
    hit_random = [
        by_seed_gain[s][gain]["per_condition_basin_hit_rate"]["random"]
        for s in SEEDS
    ]
    rank_role = [
        by_seed_gain[s][gain]["per_condition_mean_role_target_rank"]["role"]
        for s in SEEDS
    ]
    rank_content = [
        by_seed_gain[s][gain]["per_condition_mean_role_target_rank"]["content"]
        for s in SEEDS
    ]
    rank_random = [
        by_seed_gain[s][gain]["per_condition_mean_role_target_rank"]["random"]
        for s in SEEDS
    ]
    base_random = [by_seed_gain[s][0.0]["frac_random_lowest"] for s in SEEDS]
    base_rank = [
        by_seed_gain[s][0.0]["per_condition_mean_role_target_rank"]["role"]
        for s in SEEDS
    ]
    random_delta_vs_gain0 = [
        random_lowest[i] - base_random[i] for i in range(len(SEEDS))
    ]
    rank_delta_vs_gain0 = [rank_role[i] - base_rank[i] for i in range(len(SEEDS))]
    boot = bootstrap_ci(dE, seed=20260524 + int(gain * 100))
    tci = ci_t(dE)
    avg = mean(dE)
    cross_seed[str(int(gain))] = {
        "n_seeds": len(SEEDS),
        "mean_delta_e_raw": avg,
        "std_delta_e_raw_across_seeds": statistics.stdev(dE),
        "bootstrap_ci95_delta_e_raw": boot,
        "t_ci95_delta_e_raw": tci,
        "mean_delta_e_step3": mean(dE_step3),
        "delta_e_over_floor": avg / MAGNITUDE_FLOOR,
        "seeds_positive_raw": sum(1 for x in dE if x > 0),
        "mean_frac_positive_raw": mean(frac_pos),
        "mean_frac_role_lt_content": mean(role_lt_content),
        "mean_frac_role_lt_content_lt_random": mean(role_lt_content_lt_random),
        "mean_frac_random_lowest": mean(random_lowest),
        "mean_frac_random_lowest_delta_vs_gain0": mean(random_delta_vs_gain0),
        "random_lowest_worse_than_gain0_seed_count": sum(
            1 for x in random_delta_vs_gain0 if x > 0
        ),
        "mean_basin_hit_role": mean(hit_role),
        "mean_basin_hit_content": mean(hit_content),
        "mean_basin_hit_random": mean(hit_random),
        "mean_rank_role": mean(rank_role),
        "mean_rank_content": mean(rank_content),
        "mean_rank_random": mean(rank_random),
        "mean_rank_role_delta_vs_gain0": mean(rank_delta_vs_gain0),
        "gate_like_magnitude_and_boot_ci": bool(
            avg >= MAGNITUDE_FLOOR and boot[0] > 0.0
        ),
        "per_seed_delta_e_raw": {
            str(s): by_seed_gain[s][gain]["mean_delta_e_raw"] for s in SEEDS
        },
        "per_seed_random_lowest": {
            str(s): by_seed_gain[s][gain]["frac_random_lowest"] for s in SEEDS
        },
        "per_seed_rank_role": {
            str(s): by_seed_gain[s][gain][
                "per_condition_mean_role_target_rank"
            ]["role"]
            for s in SEEDS
        },
    }

aggregate = {
    "run_tag": RUN_TAG,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "branch": branch,
    "commit": commit,
    "diagnostic_not_graduation": True,
    "spec_refs": {
        "c_first_diagnostic_note": (
            "notes/emergent-codebook/phase-5-unified-design.md:258-267"
        ),
        "headline_metric": (
            "notes/emergent-codebook/phase-5-unified-design.md:269-292"
        ),
        "required_controls": (
            "notes/emergent-codebook/phase-5-unified-design.md:296-304"
        ),
    },
    "config": {
        "seeds": SEEDS,
        "gains": GAINS,
        "n_cues_per_seed": N_CUES,
        "beta": BETA,
        "gamma": GAMMA,
        "k_main": K_MAIN,
        "binding_noise_std": BINDING_NOISE_STD,
        "content_distortion": CONTENT_DISTORTION,
        "formulation": FORMULATION,
        "cue_seed": CUE_SEED,
        "magnitude_floor": MAGNITUDE_FLOOR,
        "controls_preserved": ["random-schema branches", "K=1"],
        "controls_not_run": ["no-prior", "no-schema-store"],
    },
    "cross_seed": cross_seed,
    "per_seed": per_seed,
}


def render_markdown(agg):
    lines = []
    lines.append("# Phase 5 Log-Prior n=10 Confirmation")
    lines.append("")
    lines.append(
        "This is a C-first Phase 5' diagnostic confirmation, not a Phase 5 "
        "graduation result."
    )
    lines.append("")
    lines.append(f"- Branch/commit: `{agg['branch']}` / `{agg['commit']}`")
    lines.append(f"- Seeds: {agg['config']['seeds']}")
    lines.append(
        f"- Locked operating point: beta={BETA}, gamma={GAMMA}, K={K_MAIN}, "
        f"content_distortion={CONTENT_DISTORTION}, "
        f"binding_noise_std={BINDING_NOISE_STD}"
    )
    lines.append(
        f"- Gain grid: {GAINS}; n_cues_per_seed={N_CUES}; "
        f"magnitude_floor={MAGNITUDE_FLOOR}"
    )
    lines.append(
        "- Preserved readouts: random-prior and role-target basin hit/rank. "
        "No-prior and no-schema-store were not run here."
    )
    lines.append("")
    lines.append("## Cross-Seed Gain Table")
    lines.append("")
    lines.append(
        "| gain | mean DeltaE | bootstrap 95% CI | seeds+ | DeltaE/floor | "
        "random_lowest | d random vs g0 | role<content | role<c<r | "
        "hit_role | rank_role | gate-like? |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"
    )
    for gain_key in ["0", "1", "2", "4"]:
        cell = agg["cross_seed"][gain_key]
        ci = cell["bootstrap_ci95_delta_e_raw"]
        lines.append(
            f"| {gain_key} | {cell['mean_delta_e_raw']:+.6f} | "
            f"[{ci[0]:+.5f}, {ci[1]:+.5f}] | "
            f"{cell['seeds_positive_raw']}/{cell['n_seeds']} | "
            f"{cell['delta_e_over_floor']:+.3f} | "
            f"{cell['mean_frac_random_lowest']:.3f} | "
            f"{cell['mean_frac_random_lowest_delta_vs_gain0']:+.3f} | "
            f"{cell['mean_frac_role_lt_content']:.3f} | "
            f"{cell['mean_frac_role_lt_content_lt_random']:.3f} | "
            f"{cell['mean_basin_hit_role']:.3f} | "
            f"{cell['mean_rank_role']:.1f} | "
            f"{'yes' if cell['gate_like_magnitude_and_boot_ci'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        "`gate-like?` means mean DeltaE >= 5.5e-3 and bootstrap lower bound "
        "> 0, but this notebook is still not a graduation run because "
        "required controls are incomplete."
    )
    lines.append("")
    lines.append("## Per-Seed DeltaE")
    lines.append("")
    lines.append(
        "| seed | g0 | g1 | g2 | g4 | random_lowest g0/g1/g2/g4 | "
        "rank_role g0/g1/g2/g4 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---|---|")
    for seed in SEEDS:
        gains = agg["per_seed"][str(seed)]["gains"]
        d = [gains[str(g)]["mean_delta_e_raw"] for g in [0, 1, 2, 4]]
        r = [gains[str(g)]["frac_random_lowest"] for g in [0, 1, 2, 4]]
        rk = [gains[str(g)]["role_target_rank_role"] for g in [0, 1, 2, 4]]
        lines.append(
            f"| {seed} | {d[0]:+.5f} | {d[1]:+.5f} | {d[2]:+.5f} | "
            f"{d[3]:+.5f} | {r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f}/"
            f"{r[3]:.2f} | {rk[0]:.1f}/{rk[1]:.1f}/{rk[2]:.1f}/"
            f"{rk[3]:.1f} |"
        )
    lines.append("")
    lines.append("## Interpretation Guard")
    lines.append("")
    lines.append(
        "- A positive gain cell can keep Path C live, but it does not "
        "graduate Phase 5 here."
    )
    lines.append(
        "- Random-prior degradation is tracked as a control caveat, not "
        "averaged away."
    )
    lines.append(
        "- Role-target basin hit/rank remain separate from the energy-margin "
        "headline readout."
    )
    return "\n".join(lines) + "\n"


md = render_markdown(aggregate)
agg_json_path = drive_root / "cross_seed_aggregate.json"
agg_md_path = drive_root / "cross_seed_aggregate.md"
agg_json_path.write_text(json.dumps(aggregate, indent=2))
agg_md_path.write_text(md)

local_agg_json = local_root / "cross_seed_aggregate.json"
local_agg_md = local_root / "cross_seed_aggregate.md"
local_agg_json.write_text(json.dumps(aggregate, indent=2))
local_agg_md.write_text(md)

print("\n=== CROSS-SEED SUMMARY ===")
print(md)
print("Aggregate JSON:", agg_json_path)
print("Aggregate markdown:", agg_md_path)
print("Per-seed/log directory:", drive_root)
