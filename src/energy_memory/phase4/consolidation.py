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

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# Maximum alpha_eff that keeps the explicit Euler step CFL-stable.
# The 1D discrete Laplacian has eigenvalues in [-4, 0]. The amplification
# factor for mode λ is (1 + α_eff·λ); stability requires |1 + α_eff·λ| ≤ 1
# for all λ, which gives α_eff ≤ 0.5 (saturated by the highest-frequency
# mode at λ=-4). Above 0.5 the explicit Euler scheme diverges; at exactly
# 0.5 the highest mode is marginally stable. We clamp at the boundary so
# any frequency-weighted configuration that would push α_eff > 0.5 is
# silently held at the boundary instead of producing NaN.
_CFL_MAX_ALPHA_EFF = 0.5


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
    # Candidate A (continuous coverage-weighted reinforcement rate).
    # When coverage_lambda > 0, each atom's reinforcement on retrieval is
    # multiplied by (1 - coverage_lambda * r_ema_i), where r_ema_i is a
    # per-atom continuous EMA of the atom's coverage redundancy against
    # the rest of the substrate. Highly redundant atoms gain almost
    # nothing per retrieval; novel atoms gain at full rate. "Death" is
    # the asymptotic limit of an atom whose strength decays to zero
    # under repeated non-reinforcement — no discrete delete event.
    # See notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md
    # §"Candidate A". coverage_ema_rate is a slow EMA (matching the
    # death_window=100 baseline timescale): r_ema halflife ≈ 100 steps.
    # Both default off; values pre-committed for Phase 5 retrain.
    coverage_lambda: float = 0.0
    coverage_ema_rate: float = 0.01
    # Step 3 of A+B: E_i-weighted retrieval contribution.
    # Each atom's retrieval-softmax weight is multiplied by
    # w_i = σ((E_i - retrieval_weight_epsilon) / retrieval_weight_tau),
    # implemented as score_bias = softplus((ε - E_i) / τ) added to the
    # log-domain score. Atoms whose E_i decays toward zero contribute
    # infinitesimally to retrieval by construction (no membership flag,
    # no threshold). Per the anti-homunculus precondition (design note
    # §"Combining candidates"): ε and τ are FIXED substrate parameters,
    # NOT adapted from observation. Active only when coverage_lambda > 0
    # (the same gate as A's reinforcement modulation). Pre-committed for
    # the next pilot: ε=0.05 (matches legacy death_threshold so the
    # sigmoid centers on the same scale), τ=0.02 (gives a 2.5σ
    # transition width around ε — smooth enough to remain continuous,
    # sharp enough to separate dead-strength atoms from alive ones by
    # ~12× in softmax weight at the population's median).
    retrieval_weight_epsilon: float = 0.05
    retrieval_weight_tau: float = 0.02
    # Pair #4 (metastability ~ replay-prioritization).
    # Per-atom metastability EMA m_i over c_i = w_i · (1 − max_j w_j),
    # where w is the softmax weight vector from each retrieve() call.
    # metastability_obs_rate (μ_obs): EMA blending coefficient applied
    # on every retrieval observation. m_i ← (1 − μ_obs)·m_i + μ_obs·c_i.
    # See notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md.
    # Default 0.0 leaves m_i at zero so the priority composition is bit-
    # identical to baseline (the κ=0 control depends on this).
    metastability_obs_rate: float = 0.0
    # C.2.1 anti-collapse pressure (NC1 as substrate dynamic).
    # See notes/notes/2026-05-26-c21-nc1-anti-collapse-precommit.md.
    # Adds a per-basin anti-collapse term to the consolidation energy
    # landscape: E_anti_collapse(k) = -λ_ac · log(tr(Σ_k) + ε_ac), whose
    # gradient w.r.t. atom k is the continuous repulsive force
    # -λ_ac · 2(μ_k − atom_k) / (tr(Σ_k) + ε_ac), applied at the
    # consolidation update site. Both λ_ac and ε_ac are FIXED substrate
    # constants set at construction — never adaptive on any observable
    # (binding watch-edge from the anti-homunculus reviewer; H8 in the
    # precommit). Defaults preserve the κ=0 baseline byte-identically.
    lambda_ac: float = 0.0
    epsilon_ac: float = 1e-6
    # C.2.1 substrate-side basin trace buffer. Larger than C.1.1's N=5
    # because the actuator benefits from more samples per basin (see
    # C.1.1 finite-sample finding in the Path C precommit). This buffer
    # is the substrate primitive both the actuator and a future C.1.1
    # refactor will consume; per H6/H9, the actuator must NOT read
    # BasinDiagnostics — it reads this buffer directly.
    basin_trace_buffer_size: int = 64
    # C.2.2 splitting-tension EMA over per-basin λ_2/λ_1 (spatial bimodality).
    # See notes/notes/2026-05-26-c22-splitting-tension-precommit.md. Per-atom
    # T_k accumulates the eigenvalue ratio of Σ_k (the same substrate
    # primitive C.2.1 reads — H11 binds the actuator to never read
    # BimodalityDiagnostics or ContextBagHistory). The modulation
    # 1 / (1 + T_k / τ_T) multiplicatively attenuates the per-atom
    # consolidation update (H12: multiplicative, not additive). All four
    # constants are FIXED at construction; H10 binds non-adaptivity.
    mu_T: float = 0.0
    tau_T: float = 0.5
    epsilon_T: float = 1e-6
    min_basin_for_signal: int = 4
    # C.2.3 cap-coverage error gradient as a substrate dynamic.
    # See notes/notes/2026-05-26-c23-cap-coverage-gradient-precommit.md.
    # Per-atom force: λ_cc · mean_i w_cc(q*_i) · (q*_i − atom_k), with
    # w_cc(q*_i) = σ((θ_cc − sim(q*_i, atom_k)) / τ_cc), a continuous
    # sigmoidal "uncovered weight." The actuator reads basin members from
    # the C.2.1 substrate primitive _basin_covariance; per H14 it does
    # NOT consume the Phase 2 cap_coverage_error function or any field of
    # src/energy_memory/phase2/metrics.py. Stateless at the per-atom level
    # (no slow per-atom buffer): atom_k(t) is the slow state, force_cc is
    # the instantaneous gradient.
    lambda_cc: float = 0.0
    # θ_cc substrate-side justification (binding watch-edge H14): at β=10
    # Hopfield retrieval the softmax landscape has a natural similarity-cap
    # structure around sim ≈ 0.5; per the C.1.4 empirical θ′(β) calibration
    # at notes/emergent-codebook/theta_prime_calibration.json the
    # recoverable similarity boundary at β=10 falls within the cap-friendly
    # range. The Phase 2 ``cap_t05`` operating point happens to coincide at
    # 0.5, but the load-bearing justification is the substrate's retrieval
    # geometry — NOT alignment with the Phase 2 metric.
    theta_cc: float = 0.5
    # τ_cc sigmoid sharpness. Default 0.1 means the sigmoid transitions
    # smoothly across sim values in [θ - 0.3, θ + 0.3]; sharp enough that
    # well-covered members get w_cc ≈ 0, soft enough that it does not
    # collapse to a hard threshold (H15). Smaller τ_cc approaches a step
    # — caught by A6's IQR check on the substrate's natural similarity-gap
    # distribution.
    tau_cc: float = 0.1
    # C.2.5 drift -> replay-tension energy.
    # See notes/notes/2026-05-26-c25-drift-replay-tension-precommit.md.
    # drift_ema_rate (μ_drift): EMA blending coefficient on the per-atom
    # finite-difference drift signal ||atom_k(t) - atom_k(t-1)|| at each
    # consolidation event. drift_replay_gain (κ_drift): multiplicative
    # gain on the per-trace drift factor in the replay-store priority
    # composition: priority *= (1 + κ_drift · Ψ[primary]). Both are
    # FIXED substrate constants — never adaptive on any observable
    # (binding H20). Defaults of 0.0 preserve the κ=0 baseline byte-
    # identically: Ψ stays at 0 and the multiplier is exactly 1.0. Per
    # H23 there is NO clamp on Ψ — if A8 stability fails, defaults are
    # revised, never bounded post-hoc.
    drift_ema_rate: float = 0.0
    drift_replay_gain: float = 0.0


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
        # Per-pattern coverage redundancy EMA (Candidate A).
        # Updated each step_dynamics() call when pattern_matrix is supplied
        # and config.coverage_lambda > 0. Stays at zero (and reinforce()
        # multiplies by 1.0) when the mechanism is off.
        self.r_ema = torch.zeros(0, dtype=torch.float32, device=self.device)
        # Per-atom metastability EMA m_i (pair #4). Updated on every
        # retrieve() call via update_metastability(weights). Stays at zero
        # when metastability_obs_rate == 0 (the κ=0 control baseline).
        self.metastability_ema = torch.zeros(0, dtype=torch.float32, device=self.device)
        # C.2.1 substrate-side basin trace buffer. Holds
        # (settled_state, top1_atom) tuples bounded by basin_trace_buffer_size.
        # Both C.2.1 (the actuator) and any future C.1.1 refactor read this
        # same primitive; per H6/H9 the actuator must never read the
        # BasinDiagnostics dataclass. The buffer is independent of C.1.1's
        # BasinTraceBuffer to keep the diagnostic module untouched.
        self._basin_buffer: Deque[Tuple["torch.Tensor", int]] = deque(
            maxlen=int(config.basin_trace_buffer_size)
        )
        # C.2.2 per-atom splitting tension T_k ∈ [0, 1]. Grows with sustained
        # spatial bimodality of basin geometry; modulates the per-atom
        # consolidation update via 1/(1 + T_k / τ_T). Stays at zero when
        # mu_T == 0 (κ=0 baseline; modulation is exactly 1.0).
        self.splitting_tension = torch.zeros(0, dtype=torch.float32, device=self.device)
        # C.2.5 per-atom drift tension Ψ_k ≥ 0 (it's a magnitude). EMA of
        # ||atom_k(t) − atom_k(t−1)||; modulates replay priority via
        # (1 + κ_drift · Ψ_k). Stays at zero when drift_ema_rate == 0
        # (κ=0 baseline; multiplier exactly 1.0). _previous_codebook is
        # snapshotted at the start of each consolidation event by the
        # orchestrator and consumed at the end via update_drift_tension.
        self.drift_tension = torch.zeros(0, dtype=torch.float32, device=self.device)
        self._previous_codebook: Optional["torch.Tensor"] = None
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

    def add_pattern(
        self,
        novelty_strength: Optional[float] = None,
        r_ema_init: Optional[float] = None,
    ) -> int:
        """Append a new pattern's u-chain; return its index.

        New patterns enter at u_1 with `novelty_strength`. All other u_k
        start at zero — they'll fill up only if replay sustains the pattern.

        A1 (notes/notes/2026-05-20-discovery-channel-r-ema-init-dynamic-form.md):
        when ``r_ema_init`` is supplied, the new atom's r_ema starts at the
        geometric equilibrium implied by the current substrate (caller
        computes r_inst for the new row via
        ``_coverage_redundancy_instantaneous`` on the augmented pattern
        matrix). When omitted, r_ema starts at 0 — the default-off
        behavior preserved for ``coverage_lambda=0`` runs and for tests
        that don't have a pattern matrix at add-time.
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
        r_ema_value = 0.0 if r_ema_init is None else float(r_ema_init)
        self.r_ema = torch.cat([
            self.r_ema,
            torch.full((1,), r_ema_value, dtype=torch.float32, device=self.device),
        ])
        # Pair #4: new atoms enter with zero metastability accumulation.
        # The audit binds this — no "prior" derived from population stats.
        self.metastability_ema = torch.cat([
            self.metastability_ema,
            torch.zeros(1, dtype=torch.float32, device=self.device),
        ])
        # C.2.2: new atoms enter with zero splitting tension (no prior basin).
        self.splitting_tension = torch.cat([
            self.splitting_tension,
            torch.zeros(1, dtype=torch.float32, device=self.device),
        ])
        # C.2.5: new atoms enter with zero drift tension; the previous-
        # codebook snapshot is invalidated to drop any stale shape — the
        # next snapshot_previous_codebook() call will re-capture.
        self.drift_tension = torch.cat([
            self.drift_tension,
            torch.zeros(1, dtype=torch.float32, device=self.device),
        ])
        self._previous_codebook = None
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

        When config.coverage_lambda > 0 (Candidate A), the input magnitude
        is multiplied by (1 - coverage_lambda * r_ema[idx]). With
        coverage_lambda=0 the multiplier is exactly 1 and behavior is
        bit-identical to baseline.
        """
        if not 0 <= idx < self.n_patterns:
            raise IndexError(f"pattern index {idx} out of range")
        m = (
            self.config.retrieval_gain
            if magnitude is None
            else float(magnitude)
        )
        cov = self.config.coverage_lambda
        if cov > 0.0:
            m = m * float((1.0 - cov * self.r_ema[idx]).clamp(min=0.0))
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

    def retrieval_weight_bias(self) -> "torch.Tensor":
        """Per-atom score bias for step 3: E_i-weighted retrieval contribution.

        Returns ``softplus((ε − |E_i|) / τ)`` per atom, equivalent to
        ``-log(σ((|E_i| − ε) / τ))``. Subtracted from ``beta · scores``
        before softmax (the existing ``score_bias`` mechanism), this
        produces:

            weight_i ∝ σ((|E_i| − ε) / τ) · exp(β · score_i)

        At ``|E_i| >> ε``: bias → 0, w_i → 1 (full contribution).
        At ``|E_i| << ε``: bias → (ε − |E_i|) / τ, w_i → 0
        (atoms with near-zero effective strength contribute
        infinitesimally to retrieval — the design's asymptotic-death
        property at the retrieval surface, not just in consolidation).

        Per the design note's anti-homunculus precondition (§"Combining
        candidates"): ``ε`` and ``τ`` are read from ``self.config`` and
        are fixed-at-construction substrate parameters; they are NOT
        adapted from observation. This method is a measurement — the
        bias values depend only on the substrate's current state, not
        on any controller decision.
        """
        e = self.effective_strength().abs()
        eps = self.config.retrieval_weight_epsilon
        tau = self.config.retrieval_weight_tau
        return torch.nn.functional.softplus((eps - e) / tau)

    def update_metastability(self, contribution: "torch.Tensor") -> None:
        """Pair #4 (Path 3): update per-atom metastability EMA from a pre-computed contribution.

        For a retrieval with trajectory-based contribution
        ``c_i^(traj) ∈ [0, 1]``:

            m_i ← (1 − μ_obs) · m_i + μ_obs · c_i^(traj)

        ``c_i^(traj)`` is the per-atom "lost-out atom" signal defined in
        notes/notes/2026-05-20-metastability-replay-prioritization-dynamic-form.md
        §ADDENDUM:

            c_i^(traj) = max_{t < T} w_i^(t) − w_i^(final)

        computed inside the existing settling loop's running max and
        surfaced on ``TorchRetrievalResult.metastability_contribution``.
        This reformulation replaces the fixed-point operationalization
        ``c_i = w_i · (1 − max_w)`` which collapses to zero on substrates
        with sharp self-retrieving basins (HEN / Kashyap 2024 finding;
        smoke-falsified on the A+B+A1' substrate, 2026-05-21).

        Atoms with sharp final basins (winners) and atoms that never
        participated contribute c_i ≈ 0; atoms that competed mid-settling
        but lost contribute c_i > 0.

        When ``config.metastability_obs_rate == 0`` this method is a no-op
        (the κ=0 control baseline; m_i stays at zero so the priority
        composition is bit-identical to the pre-pivot replay store).

        Args:
            contribution: per-atom ``c_i^(traj)`` tensor. Shape
                ``(n_patterns,)``. Sourced from
                ``TorchRetrievalResult.metastability_contribution`` —
                pre-computed inside retrieve() (audit constraint #1: no
                re-evaluation pass).
        """
        if self.config.metastability_obs_rate <= 0.0:
            return
        if self.n_patterns == 0:
            return
        if contribution.shape[0] != self.n_patterns:
            raise ValueError(
                f"contribution length ({contribution.shape[0]}) must match "
                f"n_patterns ({self.n_patterns})"
            )
        c = contribution.to(self.metastability_ema.dtype).to(self.device)
        mu = self.config.metastability_obs_rate
        self.metastability_ema = (1.0 - mu) * self.metastability_ema + mu * c

    def metastability_payback(self, idx: int, factor: float) -> None:
        """Pair #4: pay down m_i for atom idx when its trace is replayed.

        Called from ``ReplayStore.sample()`` after the multinomial draw
        for each sampled trace's primary atom. ``factor`` is
        ``(1 − μ_rep)`` ∈ [0, 1]; values outside that range are clamped.

        See audit constraint #3 (notes/notes/2026-05-20-metastability-...
        ): pay-down lives inside ``sample()`` after multinomial, not in
        a separate maintenance call.
        """
        if not 0 <= idx < self.n_patterns:
            raise IndexError(f"pattern index {idx} out of range")
        f = max(0.0, min(1.0, float(factor)))
        self.metastability_ema[idx] = self.metastability_ema[idx] * f

    def record_retrieval(
        self,
        settled_state: "torch.Tensor",
        top1_atom: int,
    ) -> None:
        """Append a settled-state / top1-atom pair to the basin trace buffer.

        This is the substrate-side primitive for C.2.1. Called by the
        orchestrator after each retrieval at the consolidation timescale.

        Short-circuits when no substrate consumer of the buffer is active
        (lambda_ac == 0 AND mu_T == 0 AND lambda_cc == 0): C.1.1's
        BasinDiagnostics owns its own BasinTraceBuffer, so recording is a
        pure cost when all three substrate dynamics (C.2.1, C.2.2, C.2.3)
        are off.
        """
        if (
            self.config.lambda_ac == 0.0
            and self.config.mu_T == 0.0
            and self.config.lambda_cc == 0.0
        ):
            return
        self._basin_buffer.append((settled_state.detach().clone(), int(top1_atom)))

    def basin_buffer_size(self) -> int:
        return len(self._basin_buffer)

    def _basin_covariance(
        self,
        atom_idx: int,
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"], float]:
        """Return (centroid, members, tr(Σ)) for atom_idx's basin.

        Σ is the centered Gram of members; tr(Σ) is the total within-basin
        variance, the substrate-level quantity C.2.1 bounds from below.
        For complex (FHRR) members the per-row centered inner product is
        Hermitian and tr(Σ) is real-positive.
        """
        members_list = [s for s, k in self._basin_buffer if k == atom_idx]
        if not members_list:
            return None, None, 0.0
        members = torch.stack(members_list, dim=0)
        centroid = members.mean(dim=0)
        diffs = members - centroid.unsqueeze(0)
        # tr(Σ_k) = (1/N) Σ_i <diff_i, diff_i> = (1/N) Σ_i ||diff_i||² (Hermitian).
        sq = (diffs.conj() * diffs).real.sum(dim=-1)
        tr_sigma = float(sq.mean().detach().cpu())
        return centroid, members, tr_sigma

    def anti_collapse_force(
        self,
        atom_idx: int,
        atom_state: "torch.Tensor",
    ) -> "torch.Tensor":
        """C.2.1 per-atom anti-collapse force.

        Returns -λ_ac · 2 · (μ_k − atom_k) / (tr(Σ_k) + ε_ac), the pragmatic
        operationalization of the formal gradient of
        E_anti_collapse(k) = -λ_ac · log(tr(Σ_k) + ε_ac) w.r.t. atom_k.
        See notes/notes/2026-05-26-c21-nc1-anti-collapse-precommit.md.

        Force is repulsive: a basin with zero variance (centroid == atom)
        yields zero force; once atom drifts toward μ, the −(μ − atom)
        direction pushes atom away from μ with magnitude amplified by
        1/(tr(Σ) + ε). Reads Σ_k from the substrate-side buffer, never
        from BasinDiagnostics (binding watch-edge / H6 / H9).
        """
        if self.config.lambda_ac == 0.0:
            return torch.zeros_like(atom_state)
        centroid, _, tr_sigma = self._basin_covariance(atom_idx)
        if centroid is None:
            return torch.zeros_like(atom_state)
        centroid = centroid.to(atom_state.device).to(atom_state.dtype)
        denom = tr_sigma + self.config.epsilon_ac
        return -self.config.lambda_ac * 2.0 * (centroid - atom_state) / denom

    def cap_coverage_force(
        self,
        atom_idx: int,
        atom_state: "torch.Tensor",
    ) -> "torch.Tensor":
        """C.2.3 per-atom cap-coverage error force.

        Returns λ_cc · mean_i w_cc(q*_i) · (q*_i − atom_k), where
        w_cc(q*_i) = σ((θ_cc − sim(q*_i, atom_k)) / τ_cc) is a continuous
        sigmoid "uncovered weight" — q*_i contributes a full pull when
        outside the cap and ≈ 0 when well-covered. Reads basin members
        from the C.2.1 substrate primitive (_basin_covariance); per H14
        the actuator NEVER reads ``src/energy_memory/phase2/metrics.py``.
        """
        if self.config.lambda_cc == 0.0:
            return torch.zeros_like(atom_state)
        _, members, _ = self._basin_covariance(atom_idx)
        if members is None or members.shape[0] < self.config.min_basin_for_signal:
            return torch.zeros_like(atom_state)
        m = members.to(atom_state.device).to(atom_state.dtype)
        # FHRR cosine similarity: real part of Hermitian inner product
        # divided by norms. Real-valued by construction even for complex
        # tensors.
        inner = (m.conj() * atom_state.unsqueeze(0)).sum(dim=-1)
        sim = inner.real if torch.is_complex(inner) else inner
        m_norm = (m.conj() * m).real.sum(dim=-1).clamp_min(1e-12).sqrt()
        a_norm = (
            (atom_state.conj() * atom_state).real.sum().clamp_min(1e-12).sqrt()
        )
        sim = sim / (m_norm * a_norm)
        # Sigmoid is computed in real space (sim is real).
        w = torch.sigmoid((self.config.theta_cc - sim) / self.config.tau_cc)
        diffs = m - atom_state.unsqueeze(0)
        # weighted mean of diffs: (Σ_i w_i · diff_i) / N (mean over basin).
        # For complex (FHRR) diffs, casting w to the complex dtype
        # produces a real-valued imaginary part by definition.
        w_b = w.to(diffs.dtype)
        weighted = (w_b.unsqueeze(-1) * diffs).mean(dim=0)
        return self.config.lambda_cc * weighted

    def _spatial_bimodality_signal(self, atom_idx: int) -> "torch.Tensor":
        """C.2.2 substrate signal: λ_2 / (λ_1 + ε_T) from per-basin Σ_k.

        Returns a 0-dim float32 tensor on self.device. Stays on-device
        through the eigendecomposition; the caller is responsible for any
        sync. Returns 0 when basin has < min_basin_for_signal members
        (per the C.1.1 finite-sample finding).
        """
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        members_list = [s for s, k in self._basin_buffer if k == atom_idx]
        if len(members_list) < self.config.min_basin_for_signal:
            return zero
        members = torch.stack(members_list, dim=0)
        centroid = members.mean(dim=0)
        diffs = members - centroid.unsqueeze(0)
        n = diffs.shape[0]
        # Hermitian Gram of centered basin members. For complex (FHRR)
        # tensors, diffs.conj().T @ diffs is Hermitian → real eigenvalues
        # via torch.linalg.eigh.
        # Compute the eigenvalues of σ = diffs.conj().T @ diffs / n via the
        # n×n Gram matrix gram = diffs @ diffs.conj().T / n instead of the
        # D×D scatter matrix. The two matrices share exactly the same set
        # of non-zero eigenvalues (standard "kernel trick" identity); the
        # D×D form additionally carries (D - n) trivial zero eigenvalues
        # because rank(σ) ≤ n_members ≤ basin_trace_buffer_size (64) ≪ D
        # (4096 by default in this project). That (D - n) zero subspace
        # makes σ numerically ill-conditioned at the precision available
        # to torch.linalg.eigvalsh — observed on Colab CUDA at 2026-05-27
        # as LinAlgError 4095 and even on CPU LAPACK as LinAlgError 5/12.
        # The n×n Gram path is full-rank for non-degenerate samples and
        # an order of magnitude smaller (4 KB vs 16 MB at D=4096, n=8).
        # Mathematically byte-identical at the λ_1 / λ_2 layer used below;
        # the C.2.2 dynamic's behavior is unchanged.
        gram = (diffs @ diffs.conj().transpose(-1, -2)) / float(n)
        # Eigh returns ascending eigenvalues. Take top two: λ_1 (last),
        # λ_2 (second-to-last). All ops stay on-device.
        try:
            eigvals = torch.linalg.eigvalsh(gram)
        except torch._C._LinAlgError:
            # Defensive: keep the CPU fallback in case some pathological
            # input still trips cuSOLVER (e.g. identical basin members).
            eigvals = torch.linalg.eigvalsh(gram.cpu()).to(gram.device)
        lam_1 = eigvals[-1]
        lam_2 = eigvals[-2] if eigvals.shape[0] >= 2 else torch.zeros_like(lam_1)
        # Clamp at 0 — eigh may return tiny negatives for near-singular Σ.
        lam_1 = lam_1.clamp(min=0.0)
        lam_2 = lam_2.clamp(min=0.0)
        ratio = lam_2 / (lam_1 + self.config.epsilon_T)
        return ratio.to(self.device).to(torch.float32)

    def update_splitting_tension(self) -> None:
        """EMA-update T_k from substrate-side basin covariance.

        Early-exit at mu_T == 0 (κ=0 byte-identical baseline). Per H11
        the actuator reads Σ_k from self._basin_buffer (the C.2.1
        substrate primitive); it never reads BimodalityDiagnostics or
        ContextBagHistory.
        """
        mu = self.config.mu_T
        if mu == 0.0:
            return
        if self.n_patterns == 0:
            return
        signals = torch.stack([
            self._spatial_bimodality_signal(k)
            for k in range(self.n_patterns)
        ])
        self.splitting_tension = (
            (1.0 - mu) * self.splitting_tension + mu * signals
        )

    def splitting_tension_modulation(self, atom_idx: int) -> float:
        """Per-atom attenuation factor 1 / (1 + T_k / τ_T).

        Returns 1.0 exactly when mu_T == 0 (κ=0 baseline byte-identity).
        The per-atom sync via float() is necessary for the orchestrator's
        per-atom modulation application; the precommit's perf budget
        accepts one sync per atom per consolidation event.
        """
        if self.config.mu_T == 0.0:
            return 1.0
        if not 0 <= atom_idx < self.n_patterns:
            raise IndexError(f"atom index {atom_idx} out of range")
        t_k = float(self.splitting_tension[atom_idx].detach().cpu())
        return 1.0 / (1.0 + t_k / self.config.tau_T)

    def snapshot_previous_codebook(self, codebook: "torch.Tensor") -> None:
        # C.2.5: capture current codebook so update_drift_tension can compute
        # ||current − previous|| per atom at the end of the consolidation event.
        if self.config.drift_ema_rate == 0.0:
            return
        self._previous_codebook = codebook.detach().clone()

    def update_drift_tension(self, codebook: "torch.Tensor") -> None:
        # C.2.5: EMA-update Ψ_k from ||codebook[k] − _previous_codebook[k]||.
        # Skip on shape mismatch (post-add/prune transient) — next event recovers.
        mu = self.config.drift_ema_rate
        if mu == 0.0:
            return
        prev = self._previous_codebook
        if prev is None:
            return
        if prev.shape != codebook.shape:
            return
        if codebook.shape[0] != self.n_patterns:
            return
        diff = codebook - prev
        # FHRR codebooks are complex; .abs() yields per-coord magnitude and
        # the L2 norm of |z| equals the Hermitian norm of z. Stay on-device.
        per_coord_mag_sq = (diff.conj() * diff).real
        drift_signal = per_coord_mag_sq.sum(dim=-1).clamp_min(0.0).sqrt().to(
            self.drift_tension.dtype
        )
        self.drift_tension = (1.0 - mu) * self.drift_tension + mu * drift_signal

    def step_dynamics(
        self,
        input_vector: Optional["torch.Tensor"] = None,
        pattern_matrix: Optional["torch.Tensor"] = None,
    ) -> None:
        """Advance all patterns one Benna-Fusi tick.

        Eq. 10/11 applied across all rows:
            Δu_i = α * (-2*u_i + u_{i-1} + u_{i+1})    for i in [1, m-1]
            Δu_m = α * (-2*u_m + u_{m-1})              (u_{m+1} = 0 boundary)
            u_1 += I if input_vector provided

        Boundary at u_0 is implicit: there's no u_0 below u_1, so the
        symmetric Laplacian truncates and Eq. 11 has only -2*u_1 + u_2
        as Benna-Fusi specifies.

        Candidate A (continuous redundancy EMA): when ``pattern_matrix``
        is supplied and ``config.coverage_lambda > 0``, each atom's
        r_ema is updated with a fresh instantaneous coverage estimate
        from the row-wise off-diagonal Gram. The natural "decay" term
        from the design note's formula ``dE_i/dt = (...)·(1-r_i) - λ·E_i``
        is already provided by the Benna-Fusi Laplacian outflow with
        the ``u_{m+1}=0`` boundary — no explicit -λE_i term is added.
        See notes/notes/2026-05-20-diagnostic-actuator-death-dynamic-form.md.
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
        #
        # CFL stability: see _CFL_MAX_ALPHA_EFF docstring. We clamp at the
        # 0.5 boundary so configurations that would push α_eff > 0.5
        # (e.g., alpha=0.25 with lambda > 1.0 at saturated retrieval count)
        # are held at the marginal boundary instead of diverging into NaN.
        # The clamp is a safety floor against numerical blow-up, not a
        # tuning knob — choose lambda so the unclamped value stays below
        # the bound.
        lam = self.config.alpha_freq_lambda
        if lam > 0.0:
            max_count = self.retrieval_count.max()
            if max_count > 0:
                norm_count = self.retrieval_count.to(torch.float32) / max_count.to(torch.float32)
            else:
                norm_count = torch.zeros_like(self.retrieval_count, dtype=torch.float32)
            alpha_eff = (alpha * (1.0 + lam * norm_count)).clamp(max=_CFL_MAX_ALPHA_EFF)
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
        # Candidate A: continuous redundancy EMA update.
        if (
            self.config.coverage_lambda > 0.0
            and pattern_matrix is not None
            and self.n_patterns >= 2
        ):
            r_inst = _coverage_redundancy_instantaneous(pattern_matrix)
            if r_inst.shape[0] != self.n_patterns:
                raise ValueError(
                    f"pattern_matrix rows ({r_inst.shape[0]}) must match "
                    f"n_patterns ({self.n_patterns})"
                )
            eta = self.config.coverage_ema_rate
            self.r_ema = (1.0 - eta) * self.r_ema + eta * r_inst.to(self.r_ema.dtype)
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
        self.r_ema = self.r_ema[keep]
        self.metastability_ema = self.metastability_ema[keep]
        self.splitting_tension = self.splitting_tension[keep]
        self.drift_tension = self.drift_tension[keep]
        if self._previous_codebook is not None:
            self._previous_codebook = self._previous_codebook[keep]

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
                "coverage_r_ema_mean": 0.0,
                "coverage_r_ema_max": 0.0,
                "metastability_ema_mean": 0.0,
                "metastability_ema_max": 0.0,
                "splitting_tension_mean": 0.0,
                "splitting_tension_max": 0.0,
                "drift_tension_mean": 0.0,
                "drift_tension_max": 0.0,
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
            "coverage_r_ema_mean": float(self.r_ema.mean().detach().cpu()),
            "coverage_r_ema_max": float(self.r_ema.max().detach().cpu()),
            "metastability_ema_mean": float(self.metastability_ema.mean().detach().cpu()),
            "metastability_ema_max": float(self.metastability_ema.max().detach().cpu()),
            "splitting_tension_mean": float(self.splitting_tension.mean().detach().cpu()),
            "splitting_tension_max": float(self.splitting_tension.max().detach().cpu()),
            "drift_tension_mean": float(self.drift_tension.mean().detach().cpu()),
            "drift_tension_max": float(self.drift_tension.max().detach().cpu()),
        }


def _coverage_redundancy_instantaneous(patterns: "torch.Tensor") -> "torch.Tensor":
    """Per-atom instantaneous coverage redundancy r_i ∈ [0, 1].

    Operationalization of Candidate A's r_i. The formal definition is
    ``r_i = ||proj_{P_¬i}(p_i)|| / ||p_i||``, the projection magnitude
    of atom i onto the column-span of the rest of the substrate.

    We use a per-atom Gram-row max-reduction proxy that is local per-i
    (no global SVD, no scheduled population sweep — the proxy is
    computable from atom i's similarities to its own neighbors, which is
    the local geometry available to it):

        G_ij = (1/D) * <p_i, p_j>            (complex; |G_ii| = 1)
        r_i  = max_{j≠i} |G_ij|              (∈ [0, 1])

    For unit-magnitude FHRR patterns this proxy saturates at 1 when atom
    i is identical to *any* other atom (the duplicate's |G| = 1
    dominates the max), and goes to 0 when atom i is orthogonal to all
    others.

    A1' (notes/notes/2026-05-20-r-inst-measure-dynamic-form.md):
    earlier the reduction was ``sqrt(mean_{j≠i} |G_ij|²)`` (RMS off-
    diagonal). That mean-RMS form under-measures *sparse* duplicates:
    an atom that's a perfect duplicate of one neighbor but orthogonal
    to N-2 others got r_i ≈ sqrt(1/(N-1)) ≈ 0.03 — diluted by the
    averaging across N atoms — even though the formal projection-
    magnitude reading is 1.0. Report 048 traced the A1 mechanism-
    validity failure (criterion #1) to this dilution; A1' (max-over-
    others) restores the correct saturation behavior for sparse
    duplicates. The EMA in ``ConsolidationState.step_dynamics`` smooths
    this snapshot into the slow-timescale running estimate the design
    note prescribes.
    """
    n, d = patterns.shape
    if n < 2:
        return torch.zeros(n, dtype=torch.float32, device=patterns.device)
    gram = (patterns @ patterns.conj().T) / d  # [N, N] complex
    gram_abs = gram.abs()  # [N, N] real, in [0, 1]
    mask = ~torch.eye(n, dtype=torch.bool, device=patterns.device)
    # Zero out the diagonal so it doesn't dominate the max
    gram_abs_off = gram_abs.masked_fill(~mask, 0.0)
    return gram_abs_off.max(dim=1).values.clamp(min=0.0, max=1.0).to(torch.float32)
