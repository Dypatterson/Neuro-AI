"""Preflight for a cleaned natural-source/control protocol after Report 091.

This script does not run candidate/control retrieval. It inspects the committed
Report 089 source rows and selected observed-role plans to determine whether a
stricter query/control protocol has enough support before any new gate is
allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from energy_memory.phase2.corpus import build_vocabulary, load_corpus_splits
from energy_memory.phase5.natural_source_protocol import (
    DEFAULT_FREQUENCY_CAPS as FREQ_CAPS,
    SPECIAL_ATOMS,
    cap_label as _cap_label,
    protocol_for_frequency_cap as _protocol_for_cap,
    select_recommended_protocol as _select_recommended_protocol,
)
from scripts import phase5_prime_nonsynthetic_native_gate as gate


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_vocab(repo_root: Path, corpus_source: str, c_codebook: int) -> dict:
    splits = load_corpus_splits(corpus_source, repo_root)
    vocab = build_vocabulary(splits["train"], max_vocab=max(2, c_codebook - 2))
    return {
        "id_to_token": list(vocab.id_to_token),
        "counts": dict(vocab.counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="reports/phase5_prime_nonsynthetic_native_context_source.json",
    )
    parser.add_argument(
        "--gate",
        default="reports/phase5_prime_nonsynthetic_native_gate.json",
    )
    parser.add_argument(
        "--residual",
        default="reports/phase5_prime_nonsynthetic_native_residual_analysis.json",
    )
    parser.add_argument(
        "--out",
        default="reports/phase5_prime_natural_source_control_cleanup_preflight.json",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--n_queries", type=int, default=512)
    args = parser.parse_args()

    source_path = Path(args.source)
    gate_path = Path(args.gate)
    residual_path = Path(args.residual)
    source = json.loads(source_path.read_text())
    gate_payload = json.loads(gate_path.read_text())
    residual = json.loads(residual_path.read_text())

    if source.get("framing", {}).get("source_name") != gate.SOURCE_NAME:
        raise ValueError("unexpected source artifact")
    source_sha = _sha256(source_path)
    gate_sha = _sha256(gate_path)
    residual_sha = _sha256(residual_path)
    if gate_payload.get("source_manifest", {}).get("source_artifact_sha256") != source_sha:
        raise ValueError("gate source SHA does not match source artifact")
    if residual.get("source_manifest", {}).get("gate_artifact_sha256") != gate_sha:
        raise ValueError("residual analysis SHA does not match gate artifact")

    vocab = _load_vocab(
        Path(args.repo_root),
        source.get("config", {}).get("corpus_source", "repo_sample"),
        int(source["config"]["C_codebook"]),
    )
    protocols = [
        _protocol_for_cap(
            cap=cap,
            source=source,
            vocab=vocab,
            n_queries=args.n_queries,
        )
        for cap in FREQ_CAPS
    ]
    recommended = _select_recommended_protocol(protocols)
    payload = {
        "framing": {
            "phase": "5-prime natural source/control cleanup preflight",
            "source_name": gate.SOURCE_NAME,
            "preflight_only": True,
            "analysis_only": True,
            "not_graduation": True,
            "no_candidate_control_retrieval": True,
            "description": (
                "Support and exact-opportunity preflight for a stricter natural "
                "source/control protocol after Report 091. No top1 gate is run."
            ),
        },
        "source_manifest": {
            "source_artifact_path": str(source_path),
            "source_artifact_sha256": source_sha,
            "gate_artifact_path": str(gate_path),
            "gate_artifact_sha256": gate_sha,
            "residual_artifact_path": str(residual_path),
            "residual_artifact_sha256": residual_sha,
        },
        "config": {
            "n_queries": args.n_queries,
            "frequency_caps": [_cap_label(cap) for cap in FREQ_CAPS],
            "special_atoms": sorted(SPECIAL_ATOMS),
            "source_config": source["config"],
        },
        "protocols": protocols,
        "recommended_protocol": recommended,
        "decision_read": (
            "Preflight only. A passing protocol may be used to write a separate "
            "gate precommit, but does not authorize a candidate/control top1 run."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out_path),
        "recommended_protocol": recommended,
        "protocols": [
            {
                "protocol_name": protocol["protocol_name"],
                "passes_all_criteria": protocol["passes_all_criteria"],
                "eligible_triples_min": protocol["aggregate"]["eligible_triples"]["min"],
                "target_frequency_mean": protocol["aggregate"]["target_frequency_mean"]["mean"],
                "legacy_shuffle_fixed_point_rate_mean": protocol["aggregate"][
                    "legacy_shuffle_fixed_point_rate"
                ]["mean"],
            }
            for protocol in protocols
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
