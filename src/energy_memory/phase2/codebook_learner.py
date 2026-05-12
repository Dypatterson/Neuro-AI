"""Hebbian co-occurrence codebook learning for FHRR substrates.

Learns distributional codebook vectors by interpolating each token's
vector toward the unit-modulus-normalized distributional centroid of
its co-occurring context tokens.  The ``lr`` parameter controls the
interpolation strength (0 = keep random, 1 = fully distributional).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterator, Sequence

import torch

if TYPE_CHECKING:
    from energy_memory.phase2.corpus import Vocabulary
    from energy_memory.substrate.torch_fhrr import TorchFHRR


class CodebookLearner:

    def __init__(
        self,
        substrate: TorchFHRR,
        codebook: torch.Tensor,
        vocab: Vocabulary,
        lr: float = 0.01,
        lr_decay: float = 0.85,
        repulsion_threshold: float = 0.7,
        repulsion_strength: float = 0.05,
    ):
        self.substrate = substrate
        self.codebook = codebook.clone()
        self.vocab = vocab
        self.lr = lr
        self.lr_decay = lr_decay
        self.repulsion_threshold = repulsion_threshold
        self.repulsion_strength = repulsion_strength
        self._inv_freq = self._build_inv_freq()

    def _build_inv_freq(self) -> torch.Tensor:
        weights = torch.ones(len(self.vocab.id_to_token), dtype=torch.float32)
        for i, token in enumerate(self.vocab.id_to_token):
            count = self.vocab.counts.get(token, 0)
            if count > 0:
                weights[i] = 1.0 / math.sqrt(count)
        weights[self.vocab.unk_id] = 0.0
        weights[self.vocab.mask_id] = 0.0
        return weights

    def build_cooccurrence(self, windows: Sequence[tuple[int, ...]]) -> torch.Tensor:
        """Build row-normalized, inv-freq-weighted co-occurrence matrix."""
        V = len(self.vocab.id_to_token)
        window_tensor = torch.tensor(windows, dtype=torch.long)
        _, W = window_tensor.shape
        inv_freq = self._inv_freq

        C_flat = torch.zeros(V * V, dtype=torch.float32)
        for i in range(W):
            for j in range(W):
                if i == j:
                    continue
                src = window_tensor[:, i]
                tgt = window_tensor[:, j]
                flat_idx = src * V + tgt
                w = inv_freq[tgt].float()
                C_flat += torch.bincount(flat_idx, weights=w, minlength=V * V)

        C = C_flat.reshape(V, V)

        for sid in (self.vocab.unk_id, self.vocab.mask_id):
            C[sid, :] = 0
            C[:, sid] = 0

        row_sums = C.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return C / row_sums

    def train(
        self, windows: Sequence[tuple[int, ...]], epochs: int = 10
    ) -> Iterator[dict]:
        """Yield per-epoch diagnostics while training the codebook.

        The co-occurrence matrix is computed once.  Each epoch:
        1. Computes the distributional centroid for each token (C @ codebook).
        2. Normalizes the centroid to unit modulus per component so that
           the interpolation strength is independent of vector dimension
           and co-occurrence fan-out.
        3. Interpolates: codebook = normalize(alpha * centroid + (1-alpha) * codebook).
        4. Applies pairwise repulsion if any atoms are too close.
        """
        C = self.build_cooccurrence(windows)
        has_data = C.sum(dim=1) > 1e-8  # tokens with co-occurrence info
        has_data_mask = has_data.unsqueeze(1).to(self.codebook.device)

        alpha = self.lr
        for epoch in range(epochs):
            old_codebook = self.codebook.clone()

            # Complex matmul on CPU for MPS compatibility
            cb_cpu = self.codebook.cpu()
            C_cpu = C.to(dtype=cb_cpu.dtype)
            centroid_cpu = torch.mm(C_cpu, cb_cpu)
            centroid = centroid_cpu.to(self.codebook.device)

            # Per-component unit-modulus normalization of centroids
            centroid_norm = self.substrate.normalize(centroid)

            # Interpolate toward distributional centroid
            blended = self.substrate.normalize(
                alpha * centroid_norm + (1.0 - alpha) * self.codebook
            )

            # Only update tokens that have co-occurrence data
            self.codebook = torch.where(has_data_mask, blended, self.codebook)

            mean_drift = float(
                (self.codebook - old_codebook).abs().mean().cpu().item()
            )
            repulsion_count = self._apply_repulsion()
            max_sim = self._max_pairwise_similarity()

            yield {
                "epoch": epoch,
                "lr": alpha,
                "mean_drift": mean_drift,
                "max_sim": max_sim,
                "repulsion_count": repulsion_count,
            }

            alpha *= self.lr_decay

    # ------------------------------------------------------------------
    # Collapse diagnostics
    # ------------------------------------------------------------------

    def _max_pairwise_similarity(self) -> float:
        cb = self.codebook.detach().cpu()
        D = cb.shape[1]
        sim = (cb.conj() @ cb.T).real / D
        sim.fill_diagonal_(-float("inf"))
        return float(sim.max().item())

    def _apply_repulsion(self) -> int:
        cb = self.codebook.detach().cpu()
        D = cb.shape[1]
        sim = (cb.conj() @ cb.T).real / D
        sim.fill_diagonal_(0)

        mask = sim > self.repulsion_threshold
        if not mask.any():
            return 0

        count = 0
        rows, cols = mask.nonzero(as_tuple=True)
        processed: set[tuple[int, int]] = set()
        for r, c in zip(rows.tolist(), cols.tolist()):
            pair = (min(r, c), max(r, c))
            if pair in processed:
                continue
            processed.add(pair)
            diff = self.codebook[r] - self.codebook[c]
            if diff.abs().max().item() < 1e-10:
                diff = self.substrate.random_vector().to(self.codebook.device)
            self.codebook[r] = self.substrate.normalize(
                self.codebook[r] + self.repulsion_strength * diff
            )
            self.codebook[c] = self.substrate.normalize(
                self.codebook[c] - self.repulsion_strength * diff
            )
            count += 1
        return count
