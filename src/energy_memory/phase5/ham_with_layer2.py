"""HAM aggregator extended with layer-2 attractors.

Per the Phase 5 design, Phase 4 discoveries become layer-2 attractors
(not additional layer-1 patterns). Layer 2 lives in the consensus
distribution space: each attractor is a distribution over candidate
tokens that captures a "concept" the system has discovered.

The dynamics:
  - Bottom-up: per-scale Hopfield retrieval → per-scale decoded
    distributions → raw consensus (same as base HAM)
  - Layer-2 activation: cosine similarity between raw consensus and
    each layer-2 attractor's profile → softmax over attractors
  - Top-down: weighted sum of layer-2 profiles → modified consensus
    that biases the next iteration's scale-level dynamics

The key architectural property: discovered layer-2 attractors don't
compete with layer-1 patterns. They bias layer-1 dynamics via the
consensus pathway. This avoids the winner-take-all problem we saw
when discoveries were added as layer-1 patterns.

Anti-homunculus check:
  - Layer-2 activations are softmax(cos_sim(consensus, profile)) — a
    geometric similarity, not a routing rule
  - Modified consensus is a weighted blend governed by lambda_l2, a
    fixed architectural constant
  - Layer-2 attractor age decays naturally; no supervisor decides which
    attractor to remove
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from energy_memory.memory._torch_math import torch_normalized_entropy
from energy_memory.phase2.encoding import encode_window
from energy_memory.phase5.ham_aggregator import (
    HAMConfig,
    HAMResult,
    HAMScaleInput,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass(frozen=True)
class Layer2Config:
    lambda_l2: float = 0.3
    beta_l2: float = 10.0
    capacity: int = 200
    initial_strength: float = 1.0
    strength_decay: float = 0.995
    reinforcement_gain: float = 0.05
    min_strength: float = 0.05


@dataclass
class Layer2Attractor:
    profile: "torch.Tensor"
    strength: float
    age: int = 0
    activations: int = 0
    source_query: Optional[Tuple] = None


class Layer2State:
    """Holds all layer-2 attractors and their decay/reinforcement dynamics."""

    def __init__(self, config: Layer2Config = Layer2Config()):
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError("Layer2State requires torch") from _IMPORT_ERROR
        self.config = config
        self.attractors: List[Layer2Attractor] = []

    def __len__(self) -> int:
        return len(self.attractors)

    def add(
        self,
        profile: "torch.Tensor",
        source_query: Optional[Tuple] = None,
        novelty_strength: Optional[float] = None,
    ) -> int:
        """Add a new attractor; return its index. Evicts weakest if at capacity."""
        strength = (
            self.config.initial_strength
            if novelty_strength is None
            else float(novelty_strength)
        )
        if len(self.attractors) >= self.config.capacity:
            weakest_idx = min(
                range(len(self.attractors)),
                key=lambda i: self.attractors[i].strength,
            )
            self.attractors.pop(weakest_idx)
        self.attractors.append(Layer2Attractor(
            profile=profile.detach().clone(),
            strength=strength,
            source_query=source_query,
        ))
        return len(self.attractors) - 1

    def reinforce(self, idx: int, activation_weight: float) -> None:
        """Strengthen an attractor in proportion to how much it was activated."""
        if not 0 <= idx < len(self.attractors):
            return
        a = self.attractors[idx]
        a.strength = min(
            self.config.initial_strength * 2.0,
            a.strength + self.config.reinforcement_gain * activation_weight,
        )
        a.activations += 1
        a.age = 0

    def decay_all(self) -> None:
        """Apply uniform decay to all attractor strengths and increment age."""
        cfg = self.config
        for a in self.attractors:
            a.strength *= cfg.strength_decay
            a.age += 1

    def prune_weak(self) -> List[int]:
        """Remove attractors with strength below min_strength. Return removed indices."""
        removed: List[int] = []
        survivors: List[Layer2Attractor] = []
        for i, a in enumerate(self.attractors):
            if a.strength < self.config.min_strength:
                removed.append(i)
            else:
                survivors.append(a)
        self.attractors = survivors
        return removed

    def stats(self) -> dict:
        if not self.attractors:
            return {
                "n_attractors": 0, "mean_strength": 0.0, "max_strength": 0.0,
                "mean_age": 0.0, "total_activations": 0,
            }
        return {
            "n_attractors": len(self.attractors),
            "mean_strength": sum(a.strength for a in self.attractors) / len(self.attractors),
            "max_strength": max(a.strength for a in self.attractors),
            "mean_age": sum(a.age for a in self.attractors) / len(self.attractors),
            "total_activations": sum(a.activations for a in self.attractors),
        }


@dataclass(frozen=True)
class HAML2Result:
    final_states: Dict[int, "torch.Tensor"]
    consensus: "torch.Tensor"
    layer2_activations: "torch.Tensor"
    iterations: int
    converged: bool
    engagement: float
    resolution: float
    consensus_history: List["torch.Tensor"] = field(default_factory=list)


class HAMWithLayer2:
    """HAM aggregator that incorporates layer-2 attractors in its dynamics."""

    def __init__(
        self,
        substrate: TorchFHRR,
        ham_config: HAMConfig = HAMConfig(),
        layer2_config: Layer2Config = Layer2Config(),
    ):
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError("HAMWithLayer2 requires torch") from _IMPORT_ERROR
        self.substrate = substrate
        self.ham_config = ham_config
        self.layer2_config = layer2_config
        self.layer2 = Layer2State(layer2_config)

    def retrieve(
        self,
        scale_inputs: Dict[int, HAMScaleInput],
        codebook: "torch.Tensor",
        mask_id: int,
        decode_ids: Sequence[int],
        record_trace: bool = False,
    ) -> HAML2Result:
        if not scale_inputs:
            raise ValueError("must provide at least one scale_input")

        ham_cfg = self.ham_config
        l2_cfg = self.layer2_config

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

        consensus_history: List["torch.Tensor"] = []
        per_iter_entropies: List[float] = []
        last_consensus: Optional["torch.Tensor"] = None
        layer2_activations: "torch.Tensor" = torch.zeros(
            max(len(self.layer2), 1), device=codebook.device,
        )
        converged = False
        iteration = 0

        for iteration in range(1, ham_cfg.max_iter + 1):
            new_states: Dict[int, "torch.Tensor"] = {}
            per_scale_dists: List["torch.Tensor"] = []

            for scale, inp in scale_inputs.items():
                patterns = pattern_matrices[scale]
                scores = self.substrate.similarity_matrix(states[scale], patterns)
                weights = torch.softmax(ham_cfg.beta * scores, dim=0)
                next_state = self.substrate.normalize(
                    (patterns * weights[:, None]).sum(dim=0),
                )
                new_states[scale] = next_state
                slot_query = self.substrate.unbind(
                    next_state, inp.positions[inp.local_masked_pos],
                )
                sims = self.substrate.similarity_matrix(slot_query, cand_matrix)
                dist = torch.softmax(ham_cfg.beta * sims, dim=0)
                per_scale_dists.append(dist)

            raw_consensus = self._raw_consensus(per_scale_dists)

            if len(self.layer2) > 0:
                modified_consensus, layer2_activations = self._apply_layer2(
                    raw_consensus,
                )
            else:
                modified_consensus = raw_consensus
                layer2_activations = torch.zeros(0, device=codebook.device)

            if record_trace:
                consensus_history.append(modified_consensus.detach().clone())
            per_iter_entropies.append(
                torch_normalized_entropy(modified_consensus)
            )

            expected_codebook_vec = (cand_matrix * modified_consensus[:, None]).sum(dim=0)
            for scale, inp in scale_inputs.items():
                bias = self.substrate.bind(
                    inp.positions[inp.local_masked_pos],
                    expected_codebook_vec,
                )
                new_states[scale] = self.substrate.normalize(
                    new_states[scale] + ham_cfg.alpha * bias,
                )

            if last_consensus is not None:
                delta = float((modified_consensus - last_consensus).abs().max().detach().cpu())
                if delta < ham_cfg.convergence_tol:
                    converged = True
                    states = new_states
                    last_consensus = modified_consensus
                    break

            states = new_states
            last_consensus = modified_consensus

        if last_consensus is None:
            last_consensus = torch.zeros(len(decode_ids), device=codebook.device)
            last_consensus[0] = 1.0

        engagement = (
            sum(per_iter_entropies) / len(per_iter_entropies)
            if per_iter_entropies else 0.0
        )
        # Resolution = 1 - normalized_entropy of the final consensus.
        # max-based resolution doesn't work for V-large probability
        # distributions because max(consensus) is bounded near 1/V even
        # when the system is moderately committed.
        resolution = 1.0 - torch_normalized_entropy(last_consensus)

        if len(self.layer2) > 0 and layer2_activations.shape[0] > 0:
            top_l2 = int(layer2_activations.argmax().detach().cpu())
            top_weight = float(layer2_activations[top_l2].detach().cpu())
            if top_weight > 1.0 / max(len(self.layer2), 1):
                self.layer2.reinforce(top_l2, top_weight)
            self.layer2.decay_all()

        return HAML2Result(
            final_states=states,
            consensus=last_consensus,
            layer2_activations=layer2_activations,
            iterations=iteration,
            converged=converged,
            engagement=engagement,
            resolution=resolution,
            consensus_history=consensus_history,
        )

    def _raw_consensus(self, dists: List["torch.Tensor"]) -> "torch.Tensor":
        if len(dists) == 1:
            return dists[0]
        mode = self.ham_config.consensus_mode
        if mode == "geometric_mean":
            log_dists = torch.stack(dists).clamp(min=1e-12).log()
            return torch.softmax(log_dists.mean(dim=0), dim=0)
        elif mode == "arithmetic_mean":
            c = torch.stack(dists).mean(dim=0)
            return c / c.sum().clamp(min=1e-12)
        else:
            raise ValueError(f"unknown consensus_mode: {mode}")

    def _apply_layer2(
        self,
        raw_consensus: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        cfg = self.layer2_config
        profiles = torch.stack([a.profile for a in self.layer2.attractors])
        strengths = torch.tensor(
            [a.strength for a in self.layer2.attractors],
            device=raw_consensus.device,
        )

        a_norm = raw_consensus / raw_consensus.norm().clamp(min=1e-9)
        p_norm = profiles / profiles.norm(dim=1, keepdim=True).clamp(min=1e-9)
        sims = (p_norm * a_norm[None, :]).sum(dim=1)

        weighted_logits = cfg.beta_l2 * sims + strengths.log().clamp(min=-10)
        activations = torch.softmax(weighted_logits, dim=0)
        layer2_signal = (profiles * activations[:, None]).sum(dim=0)
        layer2_signal = layer2_signal / layer2_signal.sum().clamp(min=1e-9)

        modified = (1.0 - cfg.lambda_l2) * raw_consensus + cfg.lambda_l2 * layer2_signal
        modified = modified / modified.sum().clamp(min=1e-9)
        return modified, activations

    def add_discovery(
        self,
        profile: "torch.Tensor",
        source_query: Optional[Tuple] = None,
    ) -> int:
        return self.layer2.add(profile, source_query=source_query)

    def prune_dead(self) -> int:
        return len(self.layer2.prune_weak())


