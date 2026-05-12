"""Phase 2 retrieval-baseline utilities."""

from .corpus import Vocabulary, build_vocabulary, load_repo_sample_splits, load_wikitext_splits, make_windows
from .metrics import RetrievalAggregate, summarize_binary_outcomes
try:  # pragma: no cover - exercised only when torch is available
    from .persistence import load_codebook, load_vocabulary, save_codebook, save_vocabulary
except ModuleNotFoundError:  # pragma: no cover - exercised when torch missing
    pass

try:  # pragma: no cover - exercised only when torch is available
    from .encoding import build_position_vectors, decode_position, encode_window, mask_positions, masked_window
except ModuleNotFoundError:  # pragma: no cover - exercised when torch missing
    pass

__all__ = [
    "RetrievalAggregate",
    "Vocabulary",
    "build_vocabulary",
    "load_repo_sample_splits",
    "load_wikitext_splits",
    "make_windows",
    "summarize_binary_outcomes",
]

for name in [
    "build_position_vectors",
    "decode_position",
    "encode_window",
    "load_codebook",
    "load_vocabulary",
    "mask_positions",
    "masked_window",
    "save_codebook",
    "save_vocabulary",
]:
    if name in globals():
        __all__.append(name)
