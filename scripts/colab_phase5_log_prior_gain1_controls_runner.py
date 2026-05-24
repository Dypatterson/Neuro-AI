import hashlib
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

import torch

SEEDS = [17, 11, 23, 1, 2, 3, 5, 7, 13, 29]
N_CUES = 100
BETA = 10.0
K_MAIN = 1
BASELINE_LOG_PRIOR_GAIN = 1.0
BINDING_NOISE_STD = 0.05
CONTENT_DISTORTION = 0.6
FORMULATION = "per_pattern"
CUE_SEED = 117
MAGNITUDE_FLOOR = 5.5e-3
TEMPERATURE = 1.0
DELTA_ENERGY = 0.1
DELTA_STATE = 0.3
DELTA_REDUNDANT = 0.95
MAX_SETTLING_ITER = 12
RUN_TAG = "phase5_log_prior_required_controls_v2_20260524"
RUNNER_VERSION = "v2_true_no_prior_identity_guard_20260524"
GIT_REF = "codex/phase5-log-prior-spike"
RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

BASELINE_GAIN1 = {
    "source": "reports/060_phase5_log_prior_n10_colab_confirmation.md",
    "mean_delta_e_raw": 0.008883,
    "bootstrap_ci95_delta_e_raw": [0.00645, 0.01164],
    "seeds_positive_raw": 10,
    "mean_frac_random_lowest": 0.372,
    "mean_frac_role_lt_content": 0.486,
    "mean_frac_role_lt_content_lt_random": 0.083,
    "mean_basin_hit_role": 0.003,
    "mean_rank_role": 443.3,
}

CONTROL_SPECS = [
    {
        "key": "true_no_prior",
        "label": "true no-prior gamma=0 gain=0",
        "gamma": 0.0,
        "log_prior_gain": 0.0,
        "schema_source": "slow_store",
        "expected": "effect removed because both prior channels are disabled",
    },
    {
        "key": "gamma0_log_prior_ablation",
        "label": "gamma=0 log-prior-only diagnostic",
        "gamma": 0.0,
        "log_prior_gain": 1.0,
        "schema_source": "slow_store",
        "expected": "isolates whether the additive log-prior spike bypasses gamma",
    },
    {
        "key": "gain1_no_schema_store",
        "label": "gain 1 no-schema-store full-codebook prior source",
        "gamma": 0.5,
        "log_prior_gain": 1.0,
        "schema_source": "full_codebook",
        "expected": "effect removed or shrunk because slow-store schema filtering is disabled",
    },
]

print("Active phase: 5")
print(
    "Headline metric: DeltaE = E_content-prior - E_role-prior; "
    "positive means the role-prior branch lands at lower final-state energy."
)
print(
    "Required controls: true no-prior gamma=0/gain=0, gamma-only diagnostic, "
    "and gain-1 no-schema-store, with random-prior/K=1/basin readouts preserved."
)
print(
    "Locked operating point except declared control changes: beta=10, K=1, "
    "gamma=0.5, gain=1, content_distortion=0.6, binding_noise_std=0.05, "
    "same seeds and cue seed."
)
print(f"Runner version: {RUNNER_VERSION} sha256={RUNNER_SHA256}")
print()


def run_cmd(cmd, *, cwd=None, env=None, check=True, max_tail=4000):
    cmd = [str(x) for x in cmd]
    print("$ " + " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd, cwd=cwd, env=env, text=True, capture_output=True
    )
    if result.stdout:
        out = result.stdout if len(result.stdout) <= max_tail else result.stdout[-max_tail:]
        print(out)
    if result.stderr:
        err = result.stderr if len(result.stderr) <= max_tail else result.stderr[-max_tail:]
        print(err)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}: {' '.join(cmd)}")
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
run_cmd(["git", "fetch", "origin", GIT_REF], cwd=repo)
run_cmd(["git", "checkout", GIT_REF], cwd=repo)
commit = run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=repo).stdout.strip()
branch = run_cmd(["git", "branch", "--show-current"], cwd=repo).stdout.strip()
print(f"Checked out branch={branch!r} commit={commit}")

for marker, rel, label in [
    ("log_prior_gain", "experiments/40_phase5_branching.py", "branch-local log-prior gain"),
    ("_build_role_binding_cues", "experiments/40_phase5_branching.py", "locked cue builder"),
    ("Report 060", "reports/060_phase5_log_prior_n10_colab_confirmation.md", "baseline report"),
]:
    text = (repo / rel).read_text(errors="replace")
    if marker not in text:
        raise RuntimeError(f"missing marker {marker!r} in {rel}")
    print(f"[OK] {label}: {marker} in {rel}")

os.chdir(repo)
src_path = str(repo / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import importlib.util

exp40_path = repo / "experiments" / "40_phase5_branching.py"
exp40_spec = importlib.util.spec_from_file_location("experiments_40", str(exp40_path))
exp40 = importlib.util.module_from_spec(exp40_spec)
sys.modules["experiments_40"] = exp40
exp40_spec.loader.exec_module(exp40)

audit_path = repo / "scripts" / "phase5_frozen_snapshot_audit.py"
audit_spec = importlib.util.spec_from_file_location("phase5_audit", str(audit_path))
audit = importlib.util.module_from_spec(audit_spec)
sys.modules["phase5_audit"] = audit
audit_spec.loader.exec_module(audit)

drive_mount = Path("/content/drive")
drive_root = drive_mount / "MyDrive"
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
        "No CUDA GPU is visible. Set Runtime -> Change runtime type -> GPU and rerun."
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
        snapshot_paths[seed] = path
        print(f"seed {seed:>3}: snapshot OK {path}")
    else:
        missing.append(seed)
        print(f"seed {seed:>3}: MISSING {path}")
if missing:
    raise RuntimeError(f"Missing snapshots for seeds: {missing}")

local_root = repo / "reports" / RUN_TAG
local_log_root = local_root / "logs"
drive_run_root = drive_results / RUN_TAG
drive_log_root = drive_run_root / "logs"
for directory in (local_root, local_log_root, drive_run_root, drive_log_root):
    directory.mkdir(parents=True, exist_ok=True)


def basin_diagnostics_from_sims(*, sims, role_target_idx):
    argmax_idx = int(sims.argmax())
    hit = 1.0 if argmax_idx == role_target_idx else 0.0
    sim_role_target = float(sims[role_target_idx])
    rank = 1 + int((sims > sim_role_target).sum())
    return {
        "role_target_basin_hit": hit,
        "role_target_rank": float(rank),
        "top_similarity": float(sims.max()),
    }


def mean(vals):
    return statistics.mean(vals) if vals else float("nan")


def run_control_sweep(*, snapshot_path, control):
    mem, cons, patterns, positions, info = exp40._load_substrate_from_snapshot(
        path=str(snapshot_path), device="cuda"
    )
    if positions is None:
        raise RuntimeError(f"snapshot {snapshot_path} has no positions")

    patterns_matrix = mem._pattern_matrix()
    schema_source = control["schema_source"]
    log_prior_gain = float(control["log_prior_gain"])
    if schema_source == "slow_store":
        schema_store, atom_idx = exp40.get_schema_store(
            consolidation=cons,
            patterns=patterns_matrix,
            selection_rule="top_k_by_effective_strength",
            k=min(8, len(patterns)),
        )
    elif schema_source == "full_codebook":
        schema_store = patterns_matrix
        atom_idx = None
    else:
        raise RuntimeError(f"unknown schema_source: {schema_source}")

    schema_bindings = exp40.compute_schema_bindings(
        substrate=mem.substrate, schemas=schema_store, positions=positions
    )
    step3_bias = cons.retrieval_weight_bias() if cons.config.coverage_lambda > 0.0 else None
    cue_specs = exp40._build_role_binding_cues(
        substrate=mem.substrate,
        positions=positions,
        patterns=patterns,
        n_cues=N_CUES,
        binding_noise_std=BINDING_NOISE_STD,
        content_distortion=CONTENT_DISTORTION,
        seed=CUE_SEED,
    )

    condition_specs = [("role", "role"), ("content", "content"), ("random", "random")]
    per_condition = {
        name: {
            "e_min_raw": [],
            "e_min_step3": [],
            "role_target_basin_hit": [],
            "role_target_rank": [],
            "top_similarity": [],
            "selected_atom_index": [],
        }
        for name, _ in condition_specs
    }
    boltzmann_rng = torch.Generator().manual_seed(13)
    random_prior_rng = torch.Generator().manual_seed(29)

    for spec in cue_specs:
        for name, prior_type in condition_specs:
            res = exp40.run_branched_retrieval(
                cue=spec["cue"],
                cue_id=0,
                target_id=spec["role_target_idx"],
                memory=mem,
                codebook=patterns_matrix,
                positions=positions,
                decode_ids=[],
                decode_k=5,
                masked_pos=0,
                schema_store=schema_store,
                schema_atom_idx=atom_idx,
                consolidation=cons,
                prior_type=prior_type,
                k_main=K_MAIN,
                gamma=float(control["gamma"]),
                beta=BETA,
                temperature=TEMPERATURE,
                delta_energy=DELTA_ENERGY,
                delta_state=DELTA_STATE,
                delta_redundant=DELTA_REDUNDANT,
                formulation=FORMULATION,
                cue_bindings=spec["cue_bindings"],
                schema_bindings=schema_bindings,
                include_surprise_branch=False,
                max_settling_iter=MAX_SETTLING_ITER,
                boltzmann_rng=boltzmann_rng,
                random_prior_rng=random_prior_rng,
                score_bias=step3_bias,
                log_prior_gain=log_prior_gain,
                run_combiners=False,
            )
            if not res.branches:
                continue
            branches = res.branches
            e_raw = min(b.energy_unbiased for b in branches)
            e_step3 = min(b.energy_unbiased_step3 for b in branches)
            per_condition[name]["e_min_raw"].append(e_raw)
            per_condition[name]["e_min_step3"].append(e_step3)
            for branch_state in branches:
                if branch_state.schema_atom_index is not None:
                    per_condition[name]["selected_atom_index"].append(
                        float(branch_state.schema_atom_index)
                    )

            q_star = branches[0].q_settled
            sims = mem.substrate.similarity_matrix(
                q_star.to(mem.substrate.device), patterns_matrix
            )
            bd = basin_diagnostics_from_sims(
                sims=sims, role_target_idx=int(spec["role_target_idx"])
            )
            per_condition[name]["role_target_basin_hit"].append(
                bd["role_target_basin_hit"]
            )
            per_condition[name]["role_target_rank"].append(
                bd["role_target_rank"]
            )
            per_condition[name]["top_similarity"].append(bd["top_similarity"])

    content = per_condition["content"]["e_min_raw"]
    role = per_condition["role"]["e_min_raw"]
    n_pairs = min(len(content), len(role))
    per_cue_delta_raw = [content[i] - role[i] for i in range(n_pairs)]
    content_step3 = per_condition["content"]["e_min_step3"]
    role_step3 = per_condition["role"]["e_min_step3"]
    per_cue_delta_step3 = [
        content_step3[i] - role_step3[i] for i in range(n_pairs)
    ]

    role_lt_content_lt_random = 0
    role_lt_content = 0
    random_lowest = 0
    for i in range(n_pairs):
        er = per_condition["role"]["e_min_raw"][i]
        ec = per_condition["content"]["e_min_raw"][i]
        ed = (
            per_condition["random"]["e_min_raw"][i]
            if i < len(per_condition["random"]["e_min_raw"])
            else float("inf")
        )
        if er < ec < ed:
            role_lt_content_lt_random += 1
        if er < ec:
            role_lt_content += 1
        if ed < er and ed < ec:
            random_lowest += 1

    mean_delta = mean(per_cue_delta_raw)
    cell = {
        "control_key": control["key"],
        "control_label": control["label"],
        "log_prior_gain": log_prior_gain,
        "n_pairs": n_pairs,
        "mean_delta_e_raw": mean_delta,
        "mean_delta_e_step3": mean(per_cue_delta_step3),
        "frac_positive_raw": (
            sum(1 for x in per_cue_delta_raw if x > 0) / n_pairs
            if n_pairs
            else float("nan")
        ),
        "delta_e_over_floor": mean_delta / MAGNITUDE_FLOOR if n_pairs else float("nan"),
        "frac_role_lt_content_lt_random": (
            role_lt_content_lt_random / n_pairs if n_pairs else float("nan")
        ),
        "frac_role_lt_content": role_lt_content / n_pairs if n_pairs else float("nan"),
        "frac_random_lowest": random_lowest / n_pairs if n_pairs else float("nan"),
        "per_condition_basin_hit_rate": {
            name: mean(per_condition[name]["role_target_basin_hit"])
            for name, _ in condition_specs
        },
        "per_condition_mean_role_target_rank": {
            name: mean(per_condition[name]["role_target_rank"])
            for name, _ in condition_specs
        },
        "per_condition_mean_e_min_raw": {
            name: mean(per_condition[name]["e_min_raw"])
            for name, _ in condition_specs
        },
        "per_condition_mean_top_similarity": {
            name: mean(per_condition[name]["top_similarity"])
            for name, _ in condition_specs
        },
    }
    return {
        "snapshot": str(snapshot_path),
        "label": info.get("label"),
        "n_atoms": len(patterns),
        "dim": int(mem.substrate.dim),
        "coverage_lambda": float(cons.config.coverage_lambda),
        "step3_bias_active": step3_bias is not None,
        "config": {
            "beta": BETA,
            "k_main": K_MAIN,
            "gamma": float(control["gamma"]),
            "log_prior_gain": log_prior_gain,
            "n_cues": N_CUES,
            "binding_noise_std": BINDING_NOISE_STD,
            "content_distortion": CONTENT_DISTORTION,
            "formulation": FORMULATION,
            "cue_seed": CUE_SEED,
            "schema_source": schema_source,
            "schema_store_size": int(schema_store.shape[0]),
            "magnitude_floor": MAGNITUDE_FLOOR,
        },
        "cells": [cell],
    }


def copy_complete(local_path, drive_path):
    local_path = Path(local_path)
    drive_path = Path(drive_path)
    drive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = drive_path.with_name(drive_path.name + ".partial")
    shutil.copy2(local_path, tmp)
    os.replace(tmp, drive_path)
    return drive_path


def normalized_control_spec(control):
    return {
        "key": control["key"],
        "label": control["label"],
        "gamma": float(control["gamma"]),
        "log_prior_gain": float(control["log_prior_gain"]),
        "schema_source": control["schema_source"],
        "expected": control["expected"],
    }


def run_metadata(control):
    return {
        "run_tag": RUN_TAG,
        "runner_version": RUNNER_VERSION,
        "runner_sha256": RUNNER_SHA256,
        "branch": branch,
        "commit": commit,
        "control_key": control["key"],
        "control_spec": normalized_control_spec(control),
    }


def validate_control_json(path, control):
    path = Path(path)
    data = json.loads(path.read_text())
    expected_meta = run_metadata(control)
    meta = data.get("metadata")
    if meta != expected_meta:
        raise RuntimeError(
            f"{path} metadata mismatch; expected {expected_meta}, got {meta}"
        )
    sweep = data.get("log_prior_control")
    if not isinstance(sweep, dict):
        raise RuntimeError(f"{path} missing log_prior_control")
    cells = sweep.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        raise RuntimeError(f"{path} has invalid cells")
    cfg = sweep.get("config", {})
    checks = {
        "beta": BETA,
        "k_main": K_MAIN,
        "gamma": float(control["gamma"]),
        "log_prior_gain": float(control["log_prior_gain"]),
        "n_cues": N_CUES,
        "binding_noise_std": BINDING_NOISE_STD,
        "content_distortion": CONTENT_DISTORTION,
        "formulation": FORMULATION,
        "cue_seed": CUE_SEED,
        "schema_source": control["schema_source"],
    }
    for key, expected in checks.items():
        got = cfg.get(key)
        if isinstance(expected, float):
            if abs(float(got) - expected) > 1e-9:
                raise RuntimeError(f"{path} config {key}={got}; expected {expected}")
        elif got != expected:
            raise RuntimeError(f"{path} config {key}={got}; expected {expected}")
    return data


per_seed_jsons = {control["key"]: {} for control in CONTROL_SPECS}
env = {**os.environ, "PYTHONPATH": str(repo / "src")}
start = time.time()
for control in CONTROL_SPECS:
    key = control["key"]
    print(f"\n=== CONTROL: {control['label']} ===")
    for idx, seed in enumerate(SEEDS, start=1):
        drive_json = drive_run_root / key / f"seed{seed}.json"
        drive_log = drive_log_root / key / f"seed{seed}.log"
        if drive_json.exists():
            try:
                data = validate_control_json(drive_json, control)
                per_seed_jsons[key][seed] = drive_json
                cell = data["log_prior_control"]["cells"][0]
                print(
                    f"[{idx}/{len(SEEDS)}] seed {seed}: resume valid "
                    f"DeltaE={cell['mean_delta_e_raw']:+.5f} "
                    f"random_lowest={cell['frac_random_lowest']:.3f}"
                )
                continue
            except RuntimeError as exc:
                print(
                    f"[{idx}/{len(SEEDS)}] seed {seed}: existing JSON rejected; "
                    f"regenerating ({exc})"
                )

        local_json = local_root / key / f"seed{seed}.json"
        local_log = local_log_root / key / f"seed{seed}.log"
        local_json.parent.mkdir(parents=True, exist_ok=True)
        local_log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(SEEDS)}] running seed {seed} {key}", flush=True)
        t0 = time.time()
        try:
            sweep = run_control_sweep(snapshot_path=snapshot_paths[seed], control=control)
            geometry = audit.audit_snapshot(
                snapshot_path=snapshot_paths[seed],
                device="cuda",
                beta_preflight=False,
                betas=[1.0, 3.0, 5.0, 10.0, 30.0],
                n_cue_probes=4,
                preflight_seed=17,
            )
            out_doc = {
                "metadata": run_metadata(control),
                "geometry": geometry,
                "log_prior_control": sweep,
            }
            local_json.write_text(json.dumps(out_doc, indent=2))
            copy_complete(local_json, drive_json)
            validate_control_json(drive_json, control)
            per_seed_jsons[key][seed] = drive_json
            cell = sweep["cells"][0]
            elapsed = time.time() - t0
            print(
                f"seed {seed} done in {elapsed/60:.1f} min - "
                f"DeltaE={cell['mean_delta_e_raw']:+.5f} "
                f"random_lowest={cell['frac_random_lowest']:.3f} "
                f"role<content={cell['frac_role_lt_content']:.3f} "
                f"rank_role={cell['per_condition_mean_role_target_rank']['role']:.1f}",
                flush=True,
            )
            local_log.write_text("completed\n" + json.dumps(cell, indent=2) + "\n")
            copy_complete(local_log, drive_log)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if any(set(per_seed_jsons[c["key"]]) != set(SEEDS) for c in CONTROL_SPECS):
    raise RuntimeError(f"Expected all seeds for all controls; got {per_seed_jsons}")


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


per_seed = {}
by_control_seed = {}
for control in CONTROL_SPECS:
    key = control["key"]
    per_seed[key] = {}
    by_control_seed[key] = {}
    for seed in SEEDS:
        data = validate_control_json(per_seed_jsons[key][seed], control)
        sweep = data["log_prior_control"]
        geom = data.get("geometry", {})
        cell = sweep["cells"][0]
        by_control_seed[key][seed] = cell
        per_seed[key][str(seed)] = {
            "source_json": str(per_seed_jsons[key][seed]),
            "snapshot": sweep.get("snapshot"),
            "n_atoms": sweep.get("n_atoms"),
            "dim": sweep.get("dim"),
            "coverage_lambda": sweep.get("coverage_lambda"),
            "bias_cv": geom.get("bias_cv"),
            "n_pairs": cell["n_pairs"],
            "mean_delta_e_raw": cell["mean_delta_e_raw"],
            "mean_delta_e_step3": cell["mean_delta_e_step3"],
            "frac_positive_raw": cell["frac_positive_raw"],
            "frac_role_lt_content_lt_random": cell["frac_role_lt_content_lt_random"],
            "frac_role_lt_content": cell["frac_role_lt_content"],
            "frac_random_lowest": cell["frac_random_lowest"],
            "role_target_basin_hit_role": cell["per_condition_basin_hit_rate"]["role"],
            "role_target_rank_role": cell["per_condition_mean_role_target_rank"]["role"],
        }

cross_seed = {}
for control in CONTROL_SPECS:
    key = control["key"]
    cells = by_control_seed[key]
    dE = [cells[s]["mean_delta_e_raw"] for s in SEEDS]
    dE_step3 = [cells[s]["mean_delta_e_step3"] for s in SEEDS]
    frac_pos = [cells[s]["frac_positive_raw"] for s in SEEDS]
    role_lt_content = [cells[s]["frac_role_lt_content"] for s in SEEDS]
    role_lt_content_lt_random = [
        cells[s]["frac_role_lt_content_lt_random"] for s in SEEDS
    ]
    random_lowest = [cells[s]["frac_random_lowest"] for s in SEEDS]
    hit_role = [cells[s]["per_condition_basin_hit_rate"]["role"] for s in SEEDS]
    hit_content = [
        cells[s]["per_condition_basin_hit_rate"]["content"] for s in SEEDS
    ]
    hit_random = [
        cells[s]["per_condition_basin_hit_rate"]["random"] for s in SEEDS
    ]
    rank_role = [
        cells[s]["per_condition_mean_role_target_rank"]["role"] for s in SEEDS
    ]
    rank_content = [
        cells[s]["per_condition_mean_role_target_rank"]["content"] for s in SEEDS
    ]
    rank_random = [
        cells[s]["per_condition_mean_role_target_rank"]["random"] for s in SEEDS
    ]
    boot = bootstrap_ci(dE, seed=20260524 + CONTROL_SPECS.index(control))
    tci = ci_t(dE)
    avg = mean(dE)
    cross_seed[key] = {
        "label": control["label"],
        "expected": control["expected"],
        "n_seeds": len(SEEDS),
        "mean_delta_e_raw": avg,
        "std_delta_e_raw_across_seeds": statistics.stdev(dE),
        "bootstrap_ci95_delta_e_raw": boot,
        "t_ci95_delta_e_raw": tci,
        "mean_delta_e_step3": mean(dE_step3),
        "delta_e_over_floor": avg / MAGNITUDE_FLOOR,
        "delta_e_vs_report060_gain1": avg - BASELINE_GAIN1["mean_delta_e_raw"],
        "ratio_vs_report060_gain1": avg / BASELINE_GAIN1["mean_delta_e_raw"],
        "seeds_positive_raw": sum(1 for x in dE if x > 0),
        "mean_frac_positive_raw": mean(frac_pos),
        "mean_frac_role_lt_content": mean(role_lt_content),
        "mean_frac_role_lt_content_lt_random": mean(role_lt_content_lt_random),
        "mean_frac_random_lowest": mean(random_lowest),
        "random_lowest_vs_report060_gain1": (
            mean(random_lowest) - BASELINE_GAIN1["mean_frac_random_lowest"]
        ),
        "mean_basin_hit_role": mean(hit_role),
        "mean_basin_hit_content": mean(hit_content),
        "mean_basin_hit_random": mean(hit_random),
        "mean_rank_role": mean(rank_role),
        "mean_rank_content": mean(rank_content),
        "mean_rank_random": mean(rank_random),
        "mean_rank_role_vs_report060_gain1": (
            mean(rank_role) - BASELINE_GAIN1["mean_rank_role"]
        ),
        "magnitude_and_boot_ci_gate": bool(
            avg >= MAGNITUDE_FLOOR and boot[0] > 0.0
        ),
        "per_seed_delta_e_raw": {str(s): cells[s]["mean_delta_e_raw"] for s in SEEDS},
        "per_seed_random_lowest": {str(s): cells[s]["frac_random_lowest"] for s in SEEDS},
        "per_seed_rank_role": {
            str(s): cells[s]["per_condition_mean_role_target_rank"]["role"]
            for s in SEEDS
        },
    }

aggregate = {
    "run_tag": RUN_TAG,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "branch": branch,
    "commit": commit,
    "diagnostic_not_graduation": True,
    "baseline_report060_gain1": BASELINE_GAIN1,
    "runner_version": RUNNER_VERSION,
    "runner_sha256": RUNNER_SHA256,
    "spec_refs": {
        "headline_metric": "notes/emergent-codebook/phase-5-unified-design.md:269-292",
        "required_controls": "notes/emergent-codebook/phase-5-unified-design.md:296-304",
        "checklist_controls": "notes/emergent-codebook/phase-5-checklist.md:B3-B4",
    },
    "config": {
        "seeds": SEEDS,
        "n_cues_per_seed": N_CUES,
        "beta": BETA,
        "k_main": K_MAIN,
        "baseline_log_prior_gain": BASELINE_LOG_PRIOR_GAIN,
        "binding_noise_std": BINDING_NOISE_STD,
        "content_distortion": CONTENT_DISTORTION,
        "formulation": FORMULATION,
        "cue_seed": CUE_SEED,
        "magnitude_floor": MAGNITUDE_FLOOR,
        "controls": CONTROL_SPECS,
    },
    "cross_seed": cross_seed,
    "per_seed": per_seed,
}


def render_markdown(agg):
    lines = []
    lines.append("# Phase 5 Log-Prior Gain-1 Required Controls")
    lines.append("")
    lines.append(
        "This is a C-first Phase 5 diagnostic control run, not a Phase 5 "
        "graduation result."
    )
    lines.append("")
    lines.append(f"- Branch/commit: `{agg['branch']}` / `{agg['commit']}`")
    lines.append(f"- Seeds: {agg['config']['seeds']}")
    lines.append(
        f"- Locked operating point: beta={BETA}, K={K_MAIN}, "
        f"baseline gain={BASELINE_LOG_PRIOR_GAIN}, "
        f"content_distortion={CONTENT_DISTORTION}, "
        f"binding_noise_std={BINDING_NOISE_STD}"
    )
    lines.append(
        "- Control changes only: true no-prior uses gamma=0/gain=0; "
        "gamma-only diagnostic uses gamma=0/gain=1; no-schema-store uses "
        "the full codebook as the prior source with gamma=0.5/gain=1."
    )
    lines.append("")
    lines.append("## Cross-Seed Control Table")
    lines.append("")
    lines.append(
        "| control | mean DeltaE | bootstrap 95% CI | seeds+ | DeltaE/floor | "
        "vs report060 g1 | random_lowest | vs report060 g1 | role<content | "
        "role<c<r | hit_role | rank_role | magnitude gate? |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"
    )
    for control in CONTROL_SPECS:
        key = control["key"]
        cell = agg["cross_seed"][key]
        ci = cell["bootstrap_ci95_delta_e_raw"]
        lines.append(
            f"| {key} | {cell['mean_delta_e_raw']:+.6f} | "
            f"[{ci[0]:+.5f}, {ci[1]:+.5f}] | "
            f"{cell['seeds_positive_raw']}/{cell['n_seeds']} | "
            f"{cell['delta_e_over_floor']:+.3f} | "
            f"{cell['delta_e_vs_report060_gain1']:+.6f} | "
            f"{cell['mean_frac_random_lowest']:.3f} | "
            f"{cell['random_lowest_vs_report060_gain1']:+.3f} | "
            f"{cell['mean_frac_role_lt_content']:.3f} | "
            f"{cell['mean_frac_role_lt_content_lt_random']:.3f} | "
            f"{cell['mean_basin_hit_role']:.3f} | "
            f"{cell['mean_rank_role']:.1f} | "
            f"{'yes' if cell['magnitude_and_boot_ci_gate'] else 'no'} |"
        )
    lines.append("")
    lines.append("Baseline from report 060 gain 1:")
    base_ci = agg["baseline_report060_gain1"]["bootstrap_ci95_delta_e_raw"]
    lines.append(
        f"- DeltaE={BASELINE_GAIN1['mean_delta_e_raw']:+.6f}, "
        f"CI=[{base_ci[0]:+.5f}, {base_ci[1]:+.5f}], "
        f"random_lowest={BASELINE_GAIN1['mean_frac_random_lowest']:.3f}, "
        f"role<content={BASELINE_GAIN1['mean_frac_role_lt_content']:.3f}, "
        f"role<c<r={BASELINE_GAIN1['mean_frac_role_lt_content_lt_random']:.3f}, "
        f"hit_role={BASELINE_GAIN1['mean_basin_hit_role']:.3f}, "
        f"rank_role={BASELINE_GAIN1['mean_rank_role']:.1f}"
    )
    lines.append("")
    lines.append("## Per-Seed DeltaE")
    lines.append("")
    lines.append("| seed | true no-prior | gamma0 log-prior | no-schema-store |")
    lines.append("|---:|---:|---:|---:|")
    for seed in SEEDS:
        true_no_prior = agg["per_seed"]["true_no_prior"][str(seed)][
            "mean_delta_e_raw"
        ]
        gamma0_log_prior = agg["per_seed"]["gamma0_log_prior_ablation"][str(seed)][
            "mean_delta_e_raw"
        ]
        no_schema = agg["per_seed"]["gain1_no_schema_store"][str(seed)][
            "mean_delta_e_raw"
        ]
        lines.append(
            f"| {seed} | {true_no_prior:+.5f} | {gamma0_log_prior:+.5f} | "
            f"{no_schema:+.5f} |"
        )
    lines.append("")
    lines.append("## Per-Seed Random Lowest")
    lines.append("")
    lines.append("| seed | true no-prior | gamma0 log-prior | no-schema-store |")
    lines.append("|---:|---:|---:|---:|")
    for seed in SEEDS:
        true_no_prior = agg["per_seed"]["true_no_prior"][str(seed)][
            "frac_random_lowest"
        ]
        gamma0_log_prior = agg["per_seed"]["gamma0_log_prior_ablation"][str(seed)][
            "frac_random_lowest"
        ]
        no_schema = agg["per_seed"]["gain1_no_schema_store"][str(seed)][
            "frac_random_lowest"
        ]
        lines.append(
            f"| {seed} | {true_no_prior:.2f} | {gamma0_log_prior:.2f} | "
            f"{no_schema:.2f} |"
        )
    lines.append("")
    lines.append("## Interpretation Guard")
    lines.append("")
    lines.append(
        "- Passing these controls can keep Path C live only as a controlled "
        "energy-margin result; basin hit/rank remain separate and must not "
        "be laundered into a retrieval claim."
    )
    lines.append(
        "- If either control preserves the gain-1 effect near the report 060 "
        "scale, the log-prior spike is useful but insufficient evidence for "
        "Phase 5 structural retrieval."
    )
    return "\n".join(lines) + "\n"


md = render_markdown(aggregate)
agg_json_path = drive_run_root / "cross_seed_aggregate.json"
agg_md_path = drive_run_root / "cross_seed_aggregate.md"
agg_json_path.write_text(json.dumps(aggregate, indent=2))
agg_md_path.write_text(md)

local_agg_json = local_root / "cross_seed_aggregate.json"
local_agg_md = local_root / "cross_seed_aggregate.md"
local_agg_json.write_text(json.dumps(aggregate, indent=2))
local_agg_md.write_text(md)

print(f"\nPer-seed controls finished in {(time.time() - start)/60:.1f} min")
print("\n=== CROSS-SEED SUMMARY ===")
print(md)
print("Aggregate JSON:", agg_json_path)
print("Aggregate markdown:", agg_md_path)
print("Per-seed/log directory:", drive_run_root)
