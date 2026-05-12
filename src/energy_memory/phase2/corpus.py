"""Text and window preparation for the Phase 2 retrieval baseline."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Dict, Iterable, List, Sequence

try:
    from datasets import load_dataset
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dataset = None

TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Vocabulary:
    id_to_token: List[str]
    token_to_id: Dict[str, int]
    counts: Dict[str, int]
    unk_token: str = "<UNK>"
    mask_token: str = "<MASK>"

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[self.mask_token]

    def encode_token(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def decode_token(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def encode_tokens(self, tokens: Sequence[str]) -> List[int]:
        return [self.encode_token(token) for token in tokens]


def iter_normalized_tokens(text: str) -> Iterable[str]:
    yield from TOKEN_RE.findall(text.lower())


def build_vocabulary(
    texts: Sequence[str],
    max_vocab: int = 5000,
    unk_token: str = "<UNK>",
    mask_token: str = "<MASK>",
) -> Vocabulary:
    if max_vocab <= 0:
        raise ValueError("max_vocab must be positive")
    counts = Counter()
    for text in texts:
        counts.update(iter_normalized_tokens(text))
    most_common = [token for token, _ in counts.most_common(max_vocab)]
    id_to_token = [unk_token, mask_token] + [token for token in most_common if token not in {unk_token, mask_token}]
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return Vocabulary(id_to_token=id_to_token, token_to_id=token_to_id, counts=dict(counts))


def encode_texts(texts: Sequence[str], vocab: Vocabulary) -> List[int]:
    tokens: List[int] = []
    for text in texts:
        tokens.extend(vocab.encode_tokens(list(iter_normalized_tokens(text))))
    return tokens


def make_windows(token_ids: Sequence[int], window_size: int) -> List[tuple[int, ...]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    limit = len(token_ids) - (len(token_ids) % window_size)
    return [tuple(token_ids[index : index + window_size]) for index in range(0, limit, window_size)]


def sample_windows(windows: Sequence[tuple[int, ...]], count: int, seed: int) -> List[tuple[int, ...]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(windows) <= count:
        return list(windows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(windows)), count))
    return [windows[index] for index in indices]


def load_repo_sample_splits(repo_root: Path) -> Dict[str, List[str]]:
    """Load a deterministic local fallback corpus from project docs.

    This is a small validation fallback for environments where WikiText-2 is not
    available yet. It keeps the Phase 2 pipeline executable without network or
    extra dependencies.
    """
    root = Path(repo_root)
    paths = [root / "README.md"]
    paths.extend(sorted((root / "notes").rglob("*.md")))
    texts = []
    for path in paths:
        if path.exists():
            texts.append(path.read_text(encoding="utf-8"))
    if len(texts) < 3:
        raise ValueError("repo sample corpus needs at least three text documents")
    split_index = max(1, int(len(texts) * 0.8))
    train = texts[:split_index]
    validation = texts[split_index:]
    return {"train": train, "validation": validation, "test": list(validation)}


def load_wikitext_splits(name: str = "wikitext-2-raw-v1") -> Dict[str, List[str]]:
    if load_dataset is None:  # pragma: no cover - exercised only when dependency missing
        raise ModuleNotFoundError("datasets is required to load WikiText-2")
    dataset = load_dataset("wikitext", name)
    return {
        "train": [row["text"] for row in dataset["train"]],
        "validation": [row["text"] for row in dataset["validation"]],
        "test": [row["text"] for row in dataset["test"]],
    }


def load_corpus_splits(source: str, repo_root: Path, wikitext_name: str = "wikitext-2-raw-v1") -> Dict[str, List[str]]:
    if source == "repo_sample":
        return load_repo_sample_splits(repo_root)
    if source == "wikitext":
        return load_wikitext_splits(name=wikitext_name)
    raise ValueError(f"unknown corpus source: {source}")


@dataclass(frozen=True)
class NgramBaseline:
    unigram_best: int
    forward_counts: Dict[int, Dict[int, int]]
    backward_counts: Dict[int, Dict[int, int]]

    def predict_next(self, prefix_last: int) -> int:
        counts = self.forward_counts.get(prefix_last)
        return _best_from_counts(counts, self.unigram_best)

    def predict_masked(self, window: Sequence[int], masked_positions: Sequence[int], unk_id: int) -> List[int]:
        masked = set(masked_positions)
        predictions = []
        for position in masked_positions:
            left = _nearest_unmasked(window, masked, position, step=-1)
            right = _nearest_unmasked(window, masked, position, step=1)
            combined = Counter()
            if left is not None and window[left] != unk_id:
                combined.update(self.forward_counts.get(window[left], {}))
            if right is not None and window[right] != unk_id:
                combined.update(self.backward_counts.get(window[right], {}))
            predictions.append(_best_from_counts(dict(combined), self.unigram_best))
        return predictions


def build_ngram_baseline(token_ids: Sequence[int], unk_id: int) -> NgramBaseline:
    unigram_counts = Counter(token_ids)
    unigram_counts.pop(unk_id, None)
    if not unigram_counts:
        raise ValueError("need at least one non-UNK token to build baselines")

    forward_counts: Dict[int, Counter[int]] = defaultdict(Counter)
    backward_counts: Dict[int, Counter[int]] = defaultdict(Counter)
    for left, right in zip(token_ids, token_ids[1:]):
        if right != unk_id:
            forward_counts[left][right] += 1
        if left != unk_id:
            backward_counts[right][left] += 1

    return NgramBaseline(
        unigram_best=unigram_counts.most_common(1)[0][0],
        forward_counts={token: dict(counts) for token, counts in forward_counts.items() if counts},
        backward_counts={token: dict(counts) for token, counts in backward_counts.items() if counts},
    )


def _nearest_unmasked(window: Sequence[int], masked: set[int], start: int, step: int) -> int | None:
    index = start + step
    while 0 <= index < len(window):
        if index not in masked:
            return index
        index += step
    return None


def _best_from_counts(counts: Dict[int, int] | None, fallback: int) -> int:
    if not counts:
        return fallback
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]
