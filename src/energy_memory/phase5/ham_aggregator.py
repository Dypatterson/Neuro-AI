"""HAM-style coupled multi-scale aggregator.

Replaces summed-score aggregation across multi-scale Hopfield memories
with iteratively-coupled settling. At each iteration:

  1. Each scale does one Hopfield update (bottom-up: cues activate
     stored patterns).
  2. Each scale decodes its masked position to a probability
     distribution over candidate tokens.
  3. A consensus distribution is computed across scales (geometric
     mean of per-scale distributions).
  4. Each scale's state is biased toward the consensus's expected
     codebook vector at its masked position (top-down: cross-scale
     consensus shapes scale-local dynamics).

Per Krotov 2021 HAM, this captures the bidirectional energy
convergence: bottom-up flow activates patterns, top-down flow
constrains those activations through a global "consensus" signal.

We don't enforce strict global energy decrease (that requires fully
symmetric coupling, which would constrain the architecture). What we
do enforce is convergence to a fixed point via small top-down bias
strength α and bounded iterations.

Anti-homunculus check: the consensus is a geometric aggregation of
per-scale decoded distributions. No supervisor decides what to bias
toward; the consensus is what the system geometrically agrees on at
each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.encoding import encode_window
from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass(frozen=True)
class HAMConfig:
    beta: float = 30.0
    max_iter: int = 12
    alpha: float = 0.3
    convergence_tol: float = 1e-5
    consensus_mode: str = "geometric_mean"


@dataclass
class HAMScaleInput:
    """One scale's contribution to a HAM retrieval."""

    memory: TorchHopfieldMemory
    positions: List
    sub_window: List[int]
    local_masked_pos: int


@dataclass(frozen=True)
class HAMResult:
    final_states: Dict[int, "torch.Tensor"]
    consensus: "torch.Tensor"
    top_indices_per_scale: Dict[int, int]
    top_scores_per_scale: Dict[int, float]
    iterations: int
    converged: bool


class HAMAggregator:
    """Coupled multi-scale Hopfield retrieval with bidirectional dynamics."""

    def __init__(
        self,
        substrate: TorchFHRR,
        config: HAMConfig = HAMConfig(),
    ):
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError("HAMAggregator requires torch") from _IMPORT_ERROR
        self.substrate = substrate
        self.config = config

    def retrieve(
        self,
        scale_inputs: Dict[int, HAMScaleInput],
        codebook: "torch.Tensor",
        mask_id: int,
        decode_ids: Sequence[int],
    ) -> HAMResult:
        """Run coupled multi-scale settling and return joint result."""
        if not scale_inputs:
            raise ValueError("must provide at least one scale_input")

        config = self.config

        states: Dict[int, "torch.Tensor"] = {}
        pattern_matrices: Dict[int, "torch.Tensor"] = {}
        for scale, inp in scale_inputs.items():
            cue_w = list(inp.sub_window)
            cue_w[inp.local_masked_pos] = mask_id
            states[scale] = encode_window(
                self.substrate, inp.positions, codebook, cue_w,
            )
            pattern_matrices[scale] = inp.memory._pattern_matrix()

        candidate_ids = torch.tensor(list(decode_ids), device=codebook.device)
        cand_matrix = codebook[candidate_ids]

        last_consensus = None
        converged = False
        iteration = 0

        for iteration in range(1, config.max_iter + 1):
            new_states: Dict[int, "torch.Tensor"] = {}
            per_scale_dists: List["torch.Tensor"] = []

            for scale, inp in scale_inputs.items():
                patterns = pattern_matrices[scale]
                scores = self.substrate.similarity_matrix(states[scale], patterns)
                weights = torch.softmax(config.beta * scores, dim=0)
                next_state = self.substrate.normalize(
                    (patterns * weights[:, None]).sum(dim=0),
                )
                new_states[scale] = next_state

                slot_query = self.substrate.unbind(
                    next_state, inp.positions[inp.local_masked_pos],
                )
                sims = self.substrate.similarity_matrix(slot_query, cand_matrix)
                dist = torch.softmax(config.beta * sims, dim=0)
                per_scale_dists.append(dist)

            consensus = self._consensus(per_scale_dists)

            expected_codebook_vec = (cand_matrix * consensus[:, None]).sum(dim=0)

            for scale, inp in scale_inputs.items():
                bias = self.substrate.bind(
                    inp.positions[inp.local_masked_pos],
                    expected_codebook_vec,
                )
                new_states[scale] = self.substrate.normalize(
                    new_states[scale] + config.alpha * bias,
                )

            if last_consensus is not None:
                delta = float((consensus - last_consensus).abs().max().detach().cpu())
                if delta < config.convergence_tol:
                    converged = True
                    states = new_states
                    last_consensus = consensus
                    break

            states = new_states
            last_consensus = consensus

        top_indices_per_scale: Dict[int, int] = {}
        top_scores_per_scale: Dict[int, float] = {}
        for scale, inp in scale_inputs.items():
            scores = self.substrate.similarity_matrix(
                states[scale], pattern_matrices[scale],
            )
            top_idx = int(scores.argmax().detach().cpu())
            top_indices_per_scale[scale] = top_idx
            top_scores_per_scale[scale] = float(scores[top_idx].detach().cpu())

        return HAMResult(
            final_states=states,
            consensus=last_consensus if last_consensus is not None else torch.zeros(len(decode_ids)),
            top_indices_per_scale=top_indices_per_scale,
            top_scores_per_scale=top_scores_per_scale,
            iterations=iteration,
            converged=converged,
        )

    def _consensus(self, dists: List["torch.Tensor"]) -> "torch.Tensor":
        if not dists:
            raise ValueError("at least one distribution required")
        if len(dists) == 1:
            return dists[0]

        if self.config.consensus_mode == "geometric_mean":
            log_dists = torch.stack(dists).clamp(min=1e-12).log()
            mean_log = log_dists.mean(dim=0)
            consensus = torch.softmax(mean_log, dim=0)
        elif self.config.consensus_mode == "arithmetic_mean":
            consensus = torch.stack(dists).mean(dim=0)
            consensus = consensus / consensus.sum().clamp(min=1e-12)
        else:
            raise ValueError(
                f"unknown consensus_mode: {self.config.consensus_mode}"
            )
        return consensus


def predict_top_k(
    consensus: "torch.Tensor",
    decode_ids: Sequence[int],
    k: int,
) -> List[Tuple[int, float]]:
    """Pick top-K candidates from consensus distribution."""
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("predict_top_k requires torch")
    k = min(k, consensus.shape[0])
    values, indices = torch.topk(consensus, k)
    return [
        (int(decode_ids[i]), float(v))
        for i, v in zip(
            indices.detach().cpu().tolist(),
            values.detach().cpu().tolist(),
        )
    ]
