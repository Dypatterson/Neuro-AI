"""Bet-B continual-learning harness.

The live line (`CONTEXT-B.md`). One harness, injected task families, swappable
consolidation recipes, and a provenance envelope on every run.

Extracted 2026-07-25 from `experiments/83_betb_two_timescale.py`, which
experiments 84 and 85 had been loading by filename via
`importlib.spec_from_file_location`.
"""

from .consolidators import (
    BennaFusi,
    Consolidator,
    EWCAnchor,
    SubspaceRestructure,
    make_consolidator,
)
from .continual import (
    ARMS_2X2,
    ArmResult,
    ContinualNet,
    boot_ci,
    evaluate,
    log_ftsr,
    offline_replay,
    retention_matrix_metrics,
    run_arm,
    to_tensors,
    train_task,
)
from .runner import (
    MANDATORY_CONTROLS,
    MissingControlError,
    Provenance,
    apply_tiny,
    base_parser,
    git_sha,
    merge_shards,
    resolve_device,
    write_result,
)
from .tasks import (
    CompositionalAffineFamily,
    ModularArithmeticFamily,
    Stream,
    Task,
    build_family,
)

__all__ = [
    "ARMS_2X2", "ArmResult", "BennaFusi", "CompositionalAffineFamily",
    "Consolidator", "ContinualNet", "EWCAnchor", "MANDATORY_CONTROLS",
    "MissingControlError", "ModularArithmeticFamily", "Provenance", "Stream",
    "SubspaceRestructure", "Task", "apply_tiny", "base_parser", "boot_ci",
    "build_family", "evaluate", "git_sha", "log_ftsr", "make_consolidator",
    "merge_shards", "offline_replay", "resolve_device", "retention_matrix_metrics",
    "run_arm", "to_tensors", "train_task", "write_result",
]
