"""Benna-Fusi multi-timescale consolidation state.

Each stored pattern has m hidden variables u_1, ..., u_m representing
synaptic state at exponentially-spaced timescales. Per Benna & Fusi
(2016) "Computational principles of synaptic memory consolidation":

  u_i(t+1) = u_i(t) + α * (-2*u_i + u_{i-1} + u_{i+1})    (i ≥ 2, Eq. 10)
  u_1(t+1) = u_1(t) + I(t) + α * (-2*u_1 + u_2)           (Eq. 11)
  u_{m+1} = 0                                              (boundary)

Where:
  - u_1 is the fast variable (encodes recent input I)
  - u_m is the slowest variable (long-term consolidation)
  - α controls the global timescale (≈ 1/4 in Benna-Fusi simulations)
  - Time constants grow exponentially with k: τ_k ~ 2^k

The bidirectional coupling between u_k and u_{k+1} is the key innovation
(p. 959). It yields linear-N memory lifetime (vs. √N for one-way coupling).

Per the paper (p. 1026), the inter-variable coupling is predicted to be
mediated by replay activity. In our Phase 4 architecture, calling
`step_dynamics()` during a replay cycle is what drives the consolidation
chain forward.

Effective pattern strength used during retrieval is a weighted sum across
the u_k chain: strong fast component (just-input) and strong slow component
(durable) both contribute; mid-chain decay is the "trajectory through
consolidation" of an item.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class ConsolidationConfig:
    """Per-pattern multi-timescale consolidation parameters."""

    m: int = 6
    alpha: float = 0.25
    novelty_strength: float = 1.0
    retrieval_gain: float = 0.1
    death_threshold: float = 0.01
    death_window: int = 100
    strength_weights: Optional[Sequence[float]] = None
    # Saighi & Rozenberg (2025) per-pattern self-inhibition (A_k).
    # When inhibition_gain > 0, each successful retrieval of pattern k
    # increments A_k; A_k is subtracted from beta*score for k during
    # subsequent retrievals, locally narrowing the basin.
    # See notes/notes/2026-05-15-saighi-hrr-replay-synthesis.md.
    inhibition_gain: float = 0.0
    inhibition_decay: float = 0.0
    # Retrieval-frequency-weighted α (brainstorm idea 5).
    # α_eff(k) = alpha × (1 + alpha_freq_lambda × count_k / max_count).
    # At lambda=0 the cascade is identical to fixed-α Benna-Fusi. At
    # lambda>0 frequently-retrieved patterns transfer faster through
    # u_1 → ... → u_m, producing an under-capacity slow store that
    # filters for retrieval frequency. See plan at
    # notes/notes/2026-05-16-freq-weighted-alpha-experiment-plan.md.
    alpha_freq_lambda: float = 0.0


class ConsolidationState:
    """Holds u_1...u_m for every stored pattern.

    Layout is a [N, m] tensor on the substrate device. Updates are
    sparse-per-pattern (per SQ-HN's columnar update principle): adding
    a pattern initializes a new row, retrieval reinforcement updates a
    single row's u_1.

    The bidirectional dynamics step (`step_dynamics`) applies the Eq. 10/11
    coupling globally — every pattern's u-chain advances one tick. This is
    the "replay drives coupling" mechanism from Benna-Fusi p. 1026.
    """

    def __init__(
        self,
        config: ConsolidationConfig = ConsolidationConfig(),
        device: Optional[str] = None,
    ):
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError("ConsolidationState requires torch") from _IMPORT_ERROR
        if config.m < 2:
            raise ValueError("m must be >= 2 (need at least u_1 and u_2)")
        self.config = config
        self.device = torch.device(device or "cpu")
        self.u = torch.zeros((0, config.m), dtype=torch.float32, device=self.device)
        self.below_threshold_steps = torch.zeros(0, dtype=torch.int32, device=self.device)
        # Per-pattern inhibition accumulator (Saighi & Rozenberg 2025). Grows
        # on each successful retrieval; subtracted from score during settling.
        self.A = torch.zeros(0, dtype=torch.float32, device=self.device)
        # Per-pattern retrieval count (brainstorm idea 5). Increments on
        # reinforce(); used by step_dynamics() to scale α_eff per row when
        # config.alpha_freq_lambda > 0. Kept as int32; no decay.
        self.retrieval_count = torch.zeros(0, dtype=torch.int32, device=self.device)
        self._step_count = 0

        if config.strength_weights is not None:
            w = torch.tensor(
                list(config.strength_weights),
                dtype=torch.float32, device=self.device,
            )
            if w.shape[0] != config.m:
                raise ValueError("strength_weights length must equal m")
            self._strength_weights = w
        else:
            ks = torch.arange(1, config.m + 1, dtype=torch.float32, device=self.device)
            self._strength_weights = 2.0 ** (-ks + 1)

    @property
    def n_patterns(self) -> int:
        return int(self.u.shape[0])

    def add_pattern(self, novelty_strength: Optional[float] = None) -> int:
        """Append a new pattern's u-chain; return its index.

        New patterns enter at u_1 with `novelty_strength`. All other u_k
        start at zero — they'll fill up only if replay sustains the pattern.
        """
        s = (
            self.config.novelty_strength
            if novelty_strength is None
            else float(novelty_strength)
        )
        new_row = torch.zeros((1, self.config.m), dtype=torch.float32, device=self.device)
        new_row[0, 0] = s
        self.u = torch.cat([self.u, new_row], dim=0)
        self.below_threshold_steps = torch.cat([
            self.below_threshold_steps,
            torch.zeros(1, dtype=torch.int32, device=self.device),
        ])
        self.A = torch.cat([
            self.A,
            torch.zeros(1, dtype=torch.float32, device=self.device),
        ])
        self.retrieval_count = torch.cat([
            self.retrieval_count,
            torch.zeros(1, dtype=torch.int32, device=self.device),
        ])
        return self.n_patterns - 1

    def initialize_existing(self, idx: int, novelty_strength: Optional[float] = None) -> None:
        """Set u_1 to novelty_strength for an existing pattern index.

        Used when the underlying Hopfield memory already has the pattern
        stored (e.g., from initial landscape population) and we're just
        attaching consolidation state to it.
        """
        if not 0 <= idx < self.n_patterns:
            raise IndexError(f"pattern index {idx} out of range")
        s = (
            self.config.novelty_strength
            if novelty_strength is None
            else float(novelty_strength)
        )
        self.u[idx, 0] = s
        self.below_threshold_steps[idx] = 0

    def reinforce(self, idx: int, magnitude: Optional[float] = None) -> None:
        """Add to u_1 of a single pattern (retrieval reinforcement).

        Also increments retrieval_count[idx]; this counter is consumed
        by step_dynamics() when config.alpha_freq_lambda > 0 to scale
        the per-pattern coupling coefficient.
        """
        if not 0 <= idx < self.n_patterns:
            raise IndexError(f"pattern index {idx} out of range")
        m = (
            self.config.retrieval_gain
            if magnitude is None
            else float(magnitude)
        )
        self.u[idx, 0] += m
        self.retrieval_count[idx] += 1

    def accumulate_inhibition(self, idx: int, magnitude: Optional[float] = None) -> None:
        """Increment per-pattern self-inhibition A_k (Saighi & Rozenberg 2025).

        Called after a successful retrieval converges to attractor idx.
        A_k is later subtracted from beta*score for k during settling,
        locally narrowing the basin proportional to use.
        """
        if not 0 <= idx < self.n_patterns:
            raise IndexError(f"pattern index {idx} out of range")
        m = (
            self.config.inhibition_gain
            if magnitude is None
            else float(magnitude)
        )
        if m == 0.0:
            return
        self.A[idx] += m

    def inhibition_bias(self) -> "torch.Tensor":
        """Return the per-pattern inhibition vector (alias for self.A).

        Callers may pass this as `score_bias` to retrieve() to enable the
        Saighi-style basin-narrowing dynamic. When all entries are zero
        (e.g., inhibition_gain=0.0 was used), the retrieval behaves
        identically to the no-inhibition baseline.
        """
        return self.A

    def step_dynamics(self, input_vector: Optional["torch.Tensor"] = None) -> None:
        """Advance all patterns one Benna-Fusi tick.

        Eq. 10/11 applied across all rows:
            Δu_i = α * (-2*u_i + u_{i-1} + u_{i+1})    for i in [1, m-1]
            Δu_m = α * (-2*u_m + u_{m-1})              (u_{m+1} = 0 boundary)
            u_1 += I if input_vector provided

        Boundary at u_0 is implicit: there's no u_0 below u_1, so the
        symmetric Laplacian truncates and Eq. 11 has only -2*u_1 + u_2
        as Benna-Fusi specifies.
        """
        if self.n_patterns == 0:
            return

        alpha = self.config.alpha
        u = self.u
        m = self.config.m

        u_left = torch.zeros_like(u)
        u_left[:, 1:] = u[:, :-1]
        u_right = torch.zeros_like(u)
        u_right[:, :-1] = u[:, 1:]

        laplacian = -2.0 * u + u_left + u_right
        laplacian[:, 0] = -2.0 * u[:, 0] + u[:, 1]
        laplacian[:, m - 1] = -2.0 * u[:, m - 1] + u[:, m - 2]

        # Per-pattern α scaling (brainstorm idea 5). At lambda=0 alpha_eff
        # collapses to the scalar alpha and the math is identical to the
        # original Eq.10/11 implementation.
        lam = self.config.alpha_freq_lambda
        if lam > 0.0:
            max_count = self.retrieval_count.max()
            if max_count > 0:
                norm_count = self.retrieval_count.to(torch.float32) / max_count.to(torch.float32)
            else:
                norm_count = torch.zeros_like(self.retrieval_count, dtype=torch.float32)
            alpha_eff = alpha * (1.0 + lam * norm_count)
            new_u = u + alpha_eff.unsqueeze(1) * laplacian
        else:
            new_u = u + alpha * laplacian

        if input_vector is not None:
            if input_vector.shape != (self.n_patterns,):
                raise ValueError(
                    f"input_vector must have shape ({self.n_patterns},), "
                    f"got {input_vector.shape}"
                )
            new_u[:, 0] += input_vector.to(self.device)

        self.u = new_u
        # Optional Saighi inhibition decay: A_k *= (1 - decay) per step.
        # Default decay=0.0 keeps monotonic growth (Saighi's basic form).
        if self.config.inhibition_decay > 0.0 and self.n_patterns > 0:
            self.A *= (1.0 - self.config.inhibition_decay)
        self._step_count += 1
        self._update_death_counter()

    def effective_strength(self) -> "torch.Tensor":
        """Per-pattern retrieval strength: weighted sum across u-chain.

        Default weights: 2^(1-k) — fast variables dominate, slow
        variables contribute durability.
        """
        return (self.u * self._strength_weights[None, :]).sum(dim=1)

    def _update_death_counter(self) -> None:
        strength = self.effective_strength().abs()
        below = (strength < self.config.death_threshold).to(torch.int32)
        self.below_threshold_steps = (self.below_threshold_steps + 1) * below

    def dead_indices(self) -> List[int]:
        """Patterns whose strength has been below threshold for death_window steps."""
        mask = self.below_threshold_steps >= self.config.death_window
        return mask.nonzero(as_tuple=True)[0].detach().cpu().tolist()

    def remove_pattern(self, idx: int) -> None:
        """Remove a single pattern's consolidation state.

        Caller is responsible for keeping the underlying Hopfield
        memory in sync (calling its own remove logic).
        """
        if not 0 <= idx < self.n_patterns:
            raise IndexError(f"pattern index {idx} out of range")
        keep = torch.ones(self.n_patterns, dtype=torch.bool, device=self.device)
        keep[idx] = False
        self.u = self.u[keep]
        self.below_threshold_steps = self.below_threshold_steps[keep]
        self.A = self.A[keep]
        self.retrieval_count = self.retrieval_count[keep]

    def stats(self) -> dict:
        if self.n_patterns == 0:
            return {
                "n_patterns": 0,
                "mean_strength": 0.0,
                "max_strength": 0.0,
                "mean_u": [0.0] * self.config.m,
                "patterns_below_threshold": 0,
                "patterns_dead": 0,
                "inhibition_mean": 0.0,
                "inhibition_max": 0.0,
                "inhibition_nonzero": 0,
                "retrieval_count_max": 0,
                "retrieval_count_mean": 0.0,
                "retrieval_count_nonzero": 0,
            }
        strength = self.effective_strength().abs()
        rc = self.retrieval_count
        return {
            "n_patterns": self.n_patterns,
            "mean_strength": float(strength.mean().detach().cpu()),
            "max_strength": float(strength.max().detach().cpu()),
            "mean_u": self.u.mean(dim=0).detach().cpu().tolist(),
            "patterns_below_threshold": int(
                (strength < self.config.death_threshold).sum().detach().cpu()
            ),
            "patterns_dead": len(self.dead_indices()),
            "step_count": self._step_count,
            "inhibition_mean": float(self.A.mean().detach().cpu()),
            "inhibition_max": float(self.A.max().detach().cpu()),
            "inhibition_nonzero": int((self.A > 0).sum().detach().cpu()),
            "retrieval_count_max": int(rc.max().detach().cpu()),
            "retrieval_count_mean": float(rc.to(torch.float32).mean().detach().cpu()),
            "retrieval_count_nonzero": int((rc > 0).sum().detach().cpu()),
        }
