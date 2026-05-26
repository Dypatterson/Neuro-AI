"""Precommit the next Phase 5' range-shaped replay downstream lane.

This script freezes the lane after Report 109 closes the current bridge path.
It performs no replay, retrieval, or metric-producing run; it only validates the
local input artifact and writes a static plan for the downstream comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import List, Sequence


REPORT_ID = 110
LANE_ID = "range_shaped_replay_downstream_f9_v1"
BRIDGE_CLOSURE_COMMIT = "6c05685"
BRIDGE_CLOSURE_REPORT = "reports/109_phase5_prime_strict_discriminator_viability.md"
DEFAULT_CODEBOOK_PATH = (
    "reports/phase5_prime_phase3c_repo_sample_codebook/"
    "phase3c_codebook_reconstruction.pt"
)
DEFAULT_OUTPUT = "reports/phase5_prime_range_replay_downstream/precommit.json"
DEFAULT_SMOKE_OUTPUT = (
    "reports/phase5_prime_range_replay_downstream/smoke_seed17.json"
)
DEFAULT_RESULTS_OUTPUT = (
    "reports/phase5_prime_range_replay_downstream/results_n10.json"
)
DEFAULT_ANALYSIS_OUTPUT = (
    "reports/phase5_prime_range_replay_downstream/analysis_n10.json"
)
STANDARD_CONDITION = "standard"
RANGE_POSTSETTLE_CONDITION = "range_postsettle"
RANGE_PRESETTLE_CONDITION = "range_presettle"
DEFAULT_CONDITIONS = [
    STANDARD_CONDITION,
    RANGE_POSTSETTLE_CONDITION,
    RANGE_PRESETTLE_CONDITION,
]
DEFAULT_SEEDS = [17, 11, 23, 1, 2, 3, 5, 7, 13, 29]
DEFAULT_SMOKE_SEEDS = [17]
DEFAULT_SCALES = [2, 3, 4]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.strip()


def _experiment_command(
    *,
    seeds: Sequence[int],
    conditions: Sequence[str],
    codebook_path: str,
    out_path: str,
    n_cues: int,
    test_samples: int,
    landscape_size: int,
    dim: int,
    max_vocab: int,
    replay_every: int,
    replay_batch_size: int,
    smoothing_alpha: float,
) -> List[str]:
    return [
        "PYTHONPATH=src:.",
        ".venv/bin/python",
        "experiments/47_phase34_presettle_novelty.py",
        "--seeds",
        *[str(seed) for seed in seeds],
        "--conditions",
        *conditions,
        "--codebook-path",
        codebook_path,
        "--corpus-source",
        "repo_sample",
        "--max-vocab",
        str(max_vocab),
        "--dim",
        str(dim),
        "--scales",
        *[str(scale) for scale in DEFAULT_SCALES],
        "--landscape-size",
        str(landscape_size),
        "--n-cues",
        str(n_cues),
        "--test-samples",
        str(test_samples),
        "--replay-every",
        str(replay_every),
        "--replay-batch-size",
        str(replay_batch_size),
        "--smoothing-alpha",
        str(smoothing_alpha),
        "--out",
        out_path,
    ]


def build_precommit_payload(
    *,
    repo_root: Path,
    codebook_path: str = DEFAULT_CODEBOOK_PATH,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    smoke_seeds: Sequence[int] = DEFAULT_SMOKE_SEEDS,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    dim: int = 128,
    max_vocab: int = 128,
    n_cues: int = 120,
    smoke_n_cues: int = 20,
    test_samples: int = 40,
    smoke_test_samples: int = 8,
    landscape_size: int = 16,
    smoke_landscape_size: int = 8,
    replay_every: int = 5,
    replay_batch_size: int = 4,
    smoothing_alpha: float = 0.0,
    smoke_output: str = DEFAULT_SMOKE_OUTPUT,
    results_output: str = DEFAULT_RESULTS_OUTPUT,
    analysis_output: str = DEFAULT_ANALYSIS_OUTPUT,
) -> dict:
    unknown = sorted(set(conditions) - set(DEFAULT_CONDITIONS))
    if unknown:
        raise ValueError(f"unknown downstream conditions: {unknown}")
    if STANDARD_CONDITION not in conditions:
        raise ValueError("standard condition is required")
    if len(set(conditions)) != len(list(conditions)):
        raise ValueError("conditions must be unique")
    if len(set(seeds)) != len(list(seeds)):
        raise ValueError("seeds must be unique")
    if len(seeds) < 10:
        raise ValueError("scale plan must keep n_seeds >= 10")

    codebook = repo_root / codebook_path
    if not codebook.exists():
        raise FileNotFoundError(codebook)

    fixed_parameters = {
        "dim": int(dim),
        "device": "cpu",
        "corpus_source": "repo_sample",
        "max_vocab": int(max_vocab),
        "codebook_path": codebook_path,
        "scales": list(DEFAULT_SCALES),
        "landscape_size": int(landscape_size),
        "eval_window_size": 4,
        "mask_position": "center",
        "n_cues": int(n_cues),
        "test_samples": int(test_samples),
        "decode_k": 10,
        "replay_every": int(replay_every),
        "replay_batch_size": int(replay_batch_size),
        "store_capacity": 500,
        "resolve_threshold": 0.2,
        "smoothing_alpha": float(smoothing_alpha),
        "beta": 10.0,
        "near_duplicate_threshold": 0.95,
    }
    smoke_parameters = {
        **fixed_parameters,
        "n_cues": int(smoke_n_cues),
        "test_samples": int(smoke_test_samples),
        "landscape_size": int(smoke_landscape_size),
    }
    smoke_command = _experiment_command(
        seeds=smoke_seeds,
        conditions=conditions,
        codebook_path=codebook_path,
        out_path=smoke_output,
        n_cues=smoke_n_cues,
        test_samples=smoke_test_samples,
        landscape_size=smoke_landscape_size,
        dim=dim,
        max_vocab=max_vocab,
        replay_every=replay_every,
        replay_batch_size=replay_batch_size,
        smoothing_alpha=smoothing_alpha,
    )
    scale_command = _experiment_command(
        seeds=seeds,
        conditions=conditions,
        codebook_path=codebook_path,
        out_path=results_output,
        n_cues=n_cues,
        test_samples=test_samples,
        landscape_size=landscape_size,
        dim=dim,
        max_vocab=max_vocab,
        replay_every=replay_every,
        replay_batch_size=replay_batch_size,
        smoothing_alpha=smoothing_alpha,
    )
    analysis_command = [
        "PYTHONPATH=src:.",
        ".venv/bin/python",
        "scripts/phase5_prime_range_replay_downstream_analysis.py",
        "--precommit",
        DEFAULT_OUTPUT,
        "--results",
        results_output,
        "--out",
        analysis_output,
    ]
    payload = {
        "report_id": REPORT_ID,
        "lane_id": LANE_ID,
        "status": "precommitted_no_retrieval_run",
        "active_phase": "Phase 5' precommit",
        "current_commit": git_commit(repo_root),
        "bridge_path_freeze": {
            "commit": BRIDGE_CLOSURE_COMMIT,
            "report": BRIDGE_CLOSURE_REPORT,
            "decision": "not_viable_current_bridge",
            "boundary": (
                "Do not widen the current bundle-first Delta E bridge/readout "
                "path to n=3, n=10, gate, full matrix, M2, headline, or "
                "graduation scale."
            ),
        },
        "why_this_lane_now": (
            "Checklist F9 remains open for a downstream Phase 4 consolidation "
            "comparison after range-shaped replay support was wired and after "
            "Report 109 closed the current bridge lane."
        ),
        "source_inputs": {
            "corpus_source": "repo_sample",
            "codebook_path": codebook_path,
            "codebook_sha256": sha256_file(codebook),
            "codebook_lineage": (
                "Report 072 regenerated repo-sample Phase 3C reconstruction "
                "tensor used by Reports 073-074."
            ),
            "checkpoint_inputs": [
                "reports/phase5_prime_phase3c_repo_sample_codebook/phase3c_vocab.json",
                "reports/phase5_prime_phase3c_repo_sample_codebook/04_phase3c_reconstruction.md",
            ],
        },
        "conditions": {
            STANDARD_CONDITION: {
                "sampler": "standard",
                "fallback": "whole_trace",
                "rebind_mode": None,
                "candidate_insertion_source": "post_settle_final_state",
                "role": "matched baseline",
            },
            RANGE_POSTSETTLE_CONDITION: {
                "sampler": "range_shaped",
                "fallback": "rebind",
                "rebind_mode": "window_preserving",
                "candidate_insertion_source": "post_settle_final_state",
                "role": "current production-compatible range-shaped path",
            },
            RANGE_PRESETTLE_CONDITION: {
                "sampler": "range_shaped",
                "fallback": "rebind",
                "rebind_mode": "window_preserving",
                "candidate_insertion_source": "pre_settle_query",
                "role": (
                    "fixed diagnostic positive control only; not production "
                    "mechanism and not a graduation route"
                ),
            },
        },
        "seeds": {
            "smoke": list(smoke_seeds),
            "scale": list(seeds),
            "seed_discipline": (
                "Seed 17 is wiring/smoke only. The n=10 set is required for "
                "a bounded lane decision, not Phase 5 graduation."
            ),
        },
        "fixed_parameters": fixed_parameters,
        "smoke_parameters": smoke_parameters,
        "metrics": [
            "heldout_top1",
            "heldout_topk",
            "heldout_cap_t05",
            "d_eff_final",
            "near_duplicate_rate before settle",
            "near_duplicate_rate after settle",
            "stored_near_duplicate_rate",
            "candidate_count",
            "provenance_cells",
            "provenance_rectangularity_kl",
            "query/final/stored winner diversity",
            "bootstrap 95% CI over seed-paired deltas in analysis artifact",
        ],
        "stop_criteria": {
            "current_range_path_viable_only_if": [
                (
                    "range_postsettle increases candidate/provenance support "
                    "over standard on the same seed/scale pairs"
                ),
                "stored_near_duplicate_rate_mean <= 0.25 for W=3 or W=4",
                "d_eff_final seed-paired mean delta is non-negative",
                (
                    "held-out top1 improves by >=0.02 absolute OR held-out "
                    "topk improves by >=0.05 absolute OR cap_t05 improves "
                    "by >=0.02 absolute, with bootstrap CI lower bound >=0"
                ),
            ],
            "stop_current_range_path_if": [
                (
                    "range_postsettle only increases candidate/provenance "
                    "count while stored_near_duplicate_rate remains high"
                ),
                "held-out retrieval/cap metrics are flat or worse",
                (
                    "range_presettle preserves novelty/d_eff but still lacks "
                    "held-out movement, because that would mean novelty alone "
                    "is not a useful downstream consequence in this lane"
                ),
            ],
            "phase5_escalation_allowed": False,
            "phase5_escalation_rule": (
                "Do not run Phase 5 Delta E, n=10 headline, M2, matrix, or "
                "graduation path from this lane unless a fresh precommit is "
                "written after downstream movement passes."
            ),
        },
        "commands": {
            "local_smoke": smoke_command,
            "local_scale": scale_command,
            "analysis": analysis_command,
            "colab_safari_fallback": {
                "when": (
                    "Only if local smoke passes but the n=10 run is too slow "
                    "or unavailable locally."
                ),
                "plan": [
                    "Open Safari to Colab manually or with Computer Use.",
                    "Upload/paste the same local_scale command into a notebook cell.",
                    "Use GPU only as a runtime accelerator; do not alter seeds, conditions, or stop criteria.",
                    "Download the JSON to the exact results_output path before analysis.",
                ],
            },
        },
        "artifacts": {
            "precommit": DEFAULT_OUTPUT,
            "smoke": smoke_output,
            "results": results_output,
            "analysis": analysis_output,
        },
        "anti_homunculus_check": (
            "Pass. Conditions are static before execution; diagnostics are "
            "read after the run and never switch sampler, insertion source, "
            "or replay parameters inside a run."
        ),
        "claims_not_allowed": [
            "Phase 5 graduation",
            "headline replacement",
            "bridge-path widening",
            "M2 escalation",
            "adaptive or metric-triggered routing",
            "candidate-count-only success",
        ],
        "precommit_passes": True,
    }
    payload["precommit_payload_sha256"] = stable_sha256(payload)
    return payload


def write_payload(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook-path", default=DEFAULT_CODEBOOK_PATH)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--n-cues", type=int, default=120)
    parser.add_argument("--test-samples", type=int, default=40)
    parser.add_argument("--smoke-n-cues", type=int, default=20)
    parser.add_argument("--smoke-test-samples", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=128)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    payload = build_precommit_payload(
        repo_root=repo_root,
        codebook_path=args.codebook_path,
        n_cues=args.n_cues,
        test_samples=args.test_samples,
        smoke_n_cues=args.smoke_n_cues,
        smoke_test_samples=args.smoke_test_samples,
        dim=args.dim,
        max_vocab=args.max_vocab,
    )
    out = Path(args.out)
    digest = write_payload(out, payload)
    print(f"wrote {out}")
    print(f"sha256 {digest}")
    print(f"payload_sha256 {payload['precommit_payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
