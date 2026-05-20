"""Save/load Phase 4 substrate snapshots.

A snapshot captures `(TorchHopfieldMemory._patterns, ConsolidationState)`
so Phase 5 experiments can run against a frozen post-death, pre-death,
or step-N substrate without re-running Phase 4 each session.

Indices in the saved patterns align 1:1 with the saved consolidation
rows — required by Phase 5's `get_schema_store`, which assumes
`patterns[i]` corresponds to `consolidation.u[i]`. The substrate's
removal logic (`memory.remove_pattern(idx)` + `consolidation.remove_pattern(idx)`)
keeps these aligned during a Phase 4 run; the snapshot preserves
that alignment.

The snapshot is a single .pt file via `torch.save`:

    {
      "version": 1,
      "label": str,                          # optional tag (e.g. "post_death")
      "patterns": Tensor [N, D] complex,     # stacked stored patterns
      "pattern_labels": list,                # parallel to patterns; may contain Nones
      "consolidation": {
        "u": Tensor [N, m] float32,
        "below_threshold_steps": Tensor [N] int32,
        "A": Tensor [N] float32,
        "retrieval_count": Tensor [N] int32,
        "step_count": int,
        "config": dict,                      # ConsolidationConfig as kwargs
      },
      "metadata": dict,                      # caller-supplied (seed, n_cues_seen, etc.)
    }

Save and load round-trip exactly; downstream code (get_schema_store,
settle_branch_with_prior) sees the same indices and effective_strength
ordering it would see at the moment of save.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("phase4.snapshot requires torch") from exc

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase4.consolidation import ConsolidationConfig, ConsolidationState
from energy_memory.substrate.torch_fhrr import TorchFHRR


SNAPSHOT_VERSION = 1


def save_substrate_snapshot(
    *,
    memory: TorchHopfieldMemory,
    consolidation: ConsolidationState,
    path: str | Path,
    label: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    positions: Optional[Any] = None,
) -> Path:
    """Save (memory, consolidation) to `path`.

    Refuses to save if memory and consolidation have different row counts
    — that means the caller's bookkeeping is out of sync, which would
    corrupt the snapshot's index alignment.

    `positions` (optional): a list of position vectors (or a stacked
    [W, D] tensor) used by the encoding the patterns were built from.
    Saved alongside the patterns so Phase 5's role-binding cue generator
    can use the same positions the schemas were originally encoded with.
    Without this, role-binding decomposition has no reference frame.

    Returns the path written.
    """
    if memory.stored_count != consolidation.n_patterns:
        raise ValueError(
            f"memory has {memory.stored_count} patterns but consolidation has "
            f"{consolidation.n_patterns}; substrate is out of sync, refusing "
            "to save a snapshot that would scramble Phase 5's schema store"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if memory.stored_count == 0:
        patterns_tensor = torch.empty(
            0, memory.substrate.dim, dtype=torch.complex64,
        )
    else:
        patterns_tensor = torch.stack(memory._patterns, dim=0).detach().cpu()

    cfg = consolidation.config
    cfg_dict = dataclasses.asdict(cfg)
    # strength_weights may be None or a sequence; tuple-ize for stable storage.
    if cfg_dict["strength_weights"] is not None:
        cfg_dict["strength_weights"] = tuple(cfg_dict["strength_weights"])

    positions_tensor: Optional[torch.Tensor] = None
    if positions is not None:
        if isinstance(positions, torch.Tensor):
            positions_tensor = positions.detach().cpu()
        else:
            positions_tensor = torch.stack(
                [p.detach().cpu() for p in positions], dim=0,
            )

    state = {
        "version": SNAPSHOT_VERSION,
        "label": label,
        "patterns": patterns_tensor,
        "pattern_labels": list(memory.labels),
        "positions": positions_tensor,
        "consolidation": {
            "u": consolidation.u.detach().cpu(),
            "below_threshold_steps": consolidation.below_threshold_steps.detach().cpu(),
            "A": consolidation.A.detach().cpu(),
            "retrieval_count": consolidation.retrieval_count.detach().cpu(),
            "step_count": int(consolidation._step_count),
            "config": cfg_dict,
        },
        "metadata": dict(metadata or {}),
    }
    torch.save(state, path)
    return path


def load_substrate_snapshot(
    *,
    path: str | Path,
    substrate: TorchFHRR,
    device: Optional[str] = None,
) -> Tuple[TorchHopfieldMemory, ConsolidationState, Dict[str, Any]]:
    """Restore (memory, consolidation, metadata) from a .pt snapshot.

    The caller supplies the substrate (since it carries device + dim and
    must match the original to be meaningful). Substrate dim is checked
    against the saved patterns' last dimension and raises on mismatch.

    Returns (memory, consolidation, info), where `info` contains
    `{"label": str|None, "metadata": dict, "version": int}`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"substrate snapshot not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("version") != SNAPSHOT_VERSION:
        raise ValueError(
            f"snapshot version {state.get('version')!r} does not match expected "
            f"{SNAPSHOT_VERSION!r}; cannot load"
        )
    patterns: torch.Tensor = state["patterns"]
    if patterns.numel() > 0 and patterns.shape[-1] != substrate.dim:
        raise ValueError(
            f"snapshot patterns have dim {patterns.shape[-1]} but substrate "
            f"is dim {substrate.dim}; cannot load"
        )

    mem = TorchHopfieldMemory(substrate)
    target_device = device or str(substrate.device)
    labels = state.get("pattern_labels") or [None] * patterns.shape[0]
    for i in range(patterns.shape[0]):
        mem.store(patterns[i].to(target_device), label=labels[i] if i < len(labels) else None)

    cfg_dict = dict(state["consolidation"]["config"])
    sw = cfg_dict.get("strength_weights")
    if sw is not None:
        cfg_dict["strength_weights"] = tuple(sw)
    cfg = ConsolidationConfig(**cfg_dict)
    cons = ConsolidationState(cfg, device=target_device)
    cons.u = state["consolidation"]["u"].to(target_device)
    cons.below_threshold_steps = state["consolidation"]["below_threshold_steps"].to(target_device)
    cons.A = state["consolidation"]["A"].to(target_device)
    cons.retrieval_count = state["consolidation"]["retrieval_count"].to(target_device)
    cons._step_count = int(state["consolidation"]["step_count"])

    positions_saved = state.get("positions")
    positions = None
    if positions_saved is not None:
        positions = positions_saved.to(target_device)

    info = {
        "label": state.get("label"),
        "metadata": dict(state.get("metadata") or {}),
        "version": state["version"],
        "positions": positions,
    }
    return mem, cons, info
