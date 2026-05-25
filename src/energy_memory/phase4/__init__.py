"""Phase 4: Replay and trajectory consolidation."""

from energy_memory.phase4.range_shaped_replay import (
    RangeShapedReplaySampler,
    synthesize_single_binding_trace,
    synthesize_window_preserving_trace,
)

__all__ = [
    "RangeShapedReplaySampler",
    "synthesize_single_binding_trace",
    "synthesize_window_preserving_trace",
]
