"""Run provenance, shared CLI, and shard/merge for Bet-B experiments.

**Why this exists.** Before 2026-07-25 nothing about a run was recorded and
nothing about it was enforced:

- `find reports -type f ! -name '*.md'` returned **14 files** (all PNG/stderr)
  across 196 reports — `.gitignore` excluded every `reports/**/*.json`, so not one
  number behind a published claim was on disk. Report 139's statement that a
  verifier "re-merged all 60 shards and reproduced every headline at atol < 1e-9"
  cannot be checked, because the shards are gone.
- Five separate `--merge` implementations (exps 81-85) recorded no seed ids and
  stamped no git SHA, and exp82 stored `per_seed_raw` as a list where the others
  used a dict — so the shard formats were never mutually compatible.
- `CONTEXT-B.md` §8 marks five controls "all mandatory," but Report 139 ran two.
  **`frozen-model-in-context` was never implemented in any experiment**, yet
  `reports/134/report.md:15` lists it among the controls used. A stated-but-unrun
  control is worse than a dropped one: it is invisible.

So the preamble's checkable half moves here. Declaring a control is *configuring*
it, and a run that declares one it did not execute raises — attestation replaced
by enforcement. Everything needed to re-run a result travels in the envelope.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

REPO = pathlib.Path(__file__).resolve().parents[3]


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except Exception:
        return "unknown"


class MissingControlError(RuntimeError):
    """A declared control was not executed. Fails the run rather than the reader."""


@dataclass
class Provenance:
    """Everything needed to re-run and to judge a result's scope."""

    experiment: str
    git_sha: str
    argv: List[str]
    seeds: List[int]
    config: Dict[str, Any]
    #: controls this run promises to execute (CONTEXT-B §8 names five as mandatory)
    declared_controls: List[str] = field(default_factory=list)
    executed_controls: List[str] = field(default_factory=list)
    #: van de Ven scenario. Reports 134-139 were Task-IL and mostly did not say so.
    scenario: str = "unspecified"
    task_family: str = "unspecified"
    python: str = field(default_factory=lambda: sys.version.split()[0])
    torch_version: str = "unknown"
    platform: str = field(default_factory=platform.platform)

    def verify(self) -> None:
        missing = [c for c in self.declared_controls if c not in self.executed_controls]
        if missing:
            raise MissingControlError(
                f"{self.experiment}: declared controls not executed: {missing}. "
                f"Executed: {self.executed_controls}. "
                "Either run them or stop declaring them — a stated-but-unrun control "
                "is how reports/134 came to list frozen-model-in-context."
            )


#: CONTEXT-B.md §8 "Controls (all mandatory)"
MANDATORY_CONTROLS = [
    "scratch_denominator",
    "joint_train_ceiling",
    "factorial_2x2",
    "scrambled_control",
    "frozen_model_in_context",
]


def base_parser(experiment: str) -> argparse.ArgumentParser:
    """The shared CLI. Eleven argparse lines used to appear verbatim in six files."""
    ap = argparse.ArgumentParser(description=experiment)
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--crit", type=float, default=0.95)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--replay-frac", type=float, default=0.5)
    ap.add_argument("--device", default="auto",
                    help="auto|cpu|cuda|mps — 'auto' picks the best available")
    ap.add_argument("--out", default="", help="output JSON path")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--merge", nargs="*", default=None,
                    help="merge these shard JSONs into one headline artifact")
    ap.add_argument("--tiny", action="store_true",
                    help="fast smoke: tiny p/K/steps/seeds. Never publish --tiny numbers.")
    return ap


def apply_tiny(args) -> None:
    """A CI-sized run. Kept in one place so 'does it still run?' is cheap."""
    if getattr(args, "tiny", False):
        args.p, args.K, args.seeds, args.max_steps, args.eval_every = 5, 4, 2, 120, 20


def resolve_device(spec: str) -> str:
    import torch

    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_result(
    path: str,
    payload: Dict[str, Any],
    prov: Provenance,
    *,
    verify: bool = True,
) -> Optional[pathlib.Path]:
    """Write payload + provenance. Raises if a declared control never ran."""
    if verify:
        prov.verify()
    try:
        import torch

        prov.torch_version = torch.__version__
    except Exception:
        pass
    doc = {"provenance": asdict(prov), **payload}
    if not path:
        print(json.dumps(doc, indent=2, sort_keys=True, default=str))
        return None
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
    return p


def merge_shards(paths: Sequence[str]) -> Dict[str, Any]:
    """Merge shard JSONs into one headline artifact.

    Refuses to merge shards whose configs disagree. The old per-experiment
    mergers did not check, so a merged headline could silently mix runs with
    different `p`, `K`, or `max_steps`.
    """
    docs = [json.loads(pathlib.Path(p).read_text()) for p in paths]
    if not docs:
        raise ValueError("no shards to merge")

    ref = docs[0]["provenance"]["config"]
    for pth, d in zip(paths, docs):
        cfg = d["provenance"]["config"]
        diff = {k: (ref.get(k), cfg.get(k)) for k in set(ref) | set(cfg)
                if ref.get(k) != cfg.get(k) and k not in ("shard", "n_shards", "out", "seed_start")}
        if diff:
            raise ValueError(f"shard {pth} config disagrees with {paths[0]}: {diff}")

    per_seed: Dict[str, Any] = {}
    seeds: List[int] = []
    for d in docs:
        seeds.extend(d["provenance"]["seeds"])
        raw = d.get("per_seed_raw", {})
        if isinstance(raw, list):  # exp82's format; normalize rather than reject
            raw = {str(i): v for i, v in enumerate(raw)}
        for k, v in raw.items():
            per_seed[str(k)] = v

    merged = dict(docs[0])
    merged["per_seed_raw"] = per_seed
    merged["provenance"] = dict(docs[0]["provenance"])
    merged["provenance"]["seeds"] = sorted(seeds)
    merged["provenance"]["merged_from"] = list(paths)
    merged["provenance"]["n_shards_merged"] = len(docs)
    return merged
