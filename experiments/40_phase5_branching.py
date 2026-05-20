"""Phase 5 — HAM × Energy-Guided Structural Branching.

Headline experiment for the Phase 5 unified design
(see notes/emergent-codebook/phase-5-unified-design.md).

Status: **partial implementation**. Schema-source plugin and prior
selection are wired against the post-death substrate, per the schema
robustness conditions in phase-5-checklist.md §C. Settling, diagnostics,
and the full driver remain to land.

What this script does (when fully implemented):

  For each test cue:
    1. Generate K = K_main + 1 branches:
       - K_main seeded by schemas from the schema store (one of: content
         similarity ranking, or — when the structural-retrieval test is
         enabled — role-binding similarity).
       - 1 surprise branch seeded by the highest-novelty pattern
         (log(u_1+ε) − log(u_m+ε)).
       - Diversity filter applied across schema priors so the K_main are
         not near-duplicates.
    2. Settle each branch on the substrate with prior-biased energy
         E_k(q) = -logsumexp(β · X q*) - γ · Re(⟨q, p_k⟩)
    3. Score each branch with the unbiased energy E_k^unbiased(q_k*) and
       log per-branch diagnostics (energy_drop, prior_alignment,
       entropy_collapse, recall_support, cap_coverage_t05,
       final_state_divergence, structural_match, meta_stable, converged,
       energy_biased).
    4. Combine: energy-weighted bundle + unbiased re-settle (preferred);
       also run greedy-argmin and Boltzmann-sample as comparison
       conditions.
    5. Compute the atom-splitting joint diagnostic: how many branches
       are within δ_energy of the best, and how diverged are their
       settled states?

  For the headline experiment:
    Per cue, compute the final-state unbiased energy under content-prior
    vs role-prior branching. Aggregate ΔE across n_seeds × n_cues with
    95% CI.

Anti-homunculus discipline:
  Selection is energy-only. The diagnostics listed below are logged
  per branch but do NOT feed back into the selection mechanism.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.phase2.encoding import (
    build_position_vectors,
    decode_position,
    encode_window,
    masked_window,
)
from energy_memory.phase2.metrics import meta_stable_rate
from energy_memory.phase2.persistence import load_codebook
from energy_memory.phase4.consolidation import ConsolidationConfig, ConsolidationState
from energy_memory.substrate.torch_fhrr import TorchFHRR


# =============================================================================
# Branch state + diagnostics
# =============================================================================

@dataclass
class BranchState:
    """Per-branch record: seed, settled state, full diagnostic panel."""
    branch_id: int                  # 0..K_main for schema-prior branches; K for surprise
    prior_source: str               # 'schema' | 'surprise' | 'content' | 'role' | 'random' | 'stochastic' | 'topk'
    prior: torch.Tensor             # FHRR vector used to seed this branch
    q_initial: torch.Tensor         # state at the start of settling
    q_settled: torch.Tensor         # state after retrieval converges
    # ---- diagnostics (logged, NOT used for selection) ----
    energy_unbiased: float = 0.0    # E_k^unbiased(q_settled) — the selection score
    energy_biased: float = 0.0      # E_k(q_settled) - γ * Re(<q_settled, prior>)
    energy_drop: float = 0.0        # energy_unbiased(q_initial) - energy_unbiased(q_settled)
    prior_alignment: float = 0.0    # cos(q_settled, prior)
    score_entropy_initial: float = 0.0
    score_entropy_final: float = 0.0
    entropy_collapse: float = 0.0   # initial - final
    final_state_divergence: float = 0.0  # mean cos-distance to other branches; set in post-pass
    recall_support: bool = False
    cap_coverage_t05: float = 0.0
    meta_stable: bool = False       # top_score < 0.95
    structural_match: float = 0.0   # role-binding decomposition similarity to cue
    converged: bool = False


@dataclass
class BranchedRetrievalResult:
    """One cue, K branches, three combination outcomes."""
    cue_id: int
    cue: torch.Tensor
    target_id: Optional[int]                # for readout diagnostics

    branches: List[BranchState] = field(default_factory=list)

    # Branch-combination outputs (all three run for comparison)
    q_bundle: Optional[torch.Tensor] = None     # energy-weighted bundle re-settled (PREFERRED)
    q_greedy: Optional[torch.Tensor] = None     # argmin_k energy_unbiased (BASELINE)
    q_boltzmann: Optional[torch.Tensor] = None  # sampled from softmax(-E/τ) (COMPARISON)

    # Atom-splitting joint diagnostic
    split_eligible: bool = False
    n_in_low_energy_set: int = 0
    max_state_distance_in_low_energy_set: float = 0.0

    # Aggregate-level fields filled by analysis pass:
    softmax_weights: List[float] = field(default_factory=list)
    softmax_entropy: float = 0.0


# =============================================================================
# Schema-source plugin (decision #1 resolved by report 040: post-death substrate)
# =============================================================================

def get_schema_store(
    *,
    consolidation: ConsolidationState,
    patterns: torch.Tensor,
    selection_rule: str = "top_k_by_effective_strength",
    k: int,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select k schemas from the consolidation's living atoms.

    The schema-source robustness ablation in phase-5-checklist.md §C is
    realized by calling this function multiple times against different
    consolidation snapshots (post-death, pre-death, step-1500); this
    function itself is snapshot-agnostic. Anti-homunculus discipline:
    the selection rule is fixed by the caller per condition. There is no
    runtime metric-reading that picks one rule over another.

    Parameters
    ----------
    consolidation : ConsolidationState
        Phase 4 consolidation snapshot. Indices 0..n_patterns-1 align
        with `patterns` rows (the Hopfield substrate keeps them in sync
        — see phase4.replay_loop._purge_dead).
    patterns : torch.Tensor, shape [n_patterns, D] complex
        Stored Hopfield pattern matrix (e.g. memory._pattern_matrix()).
    selection_rule : str
        'top_k_by_effective_strength' — top-k atoms by
            consolidation.effective_strength().abs(). Phase-5 design
            default (post-death survivors ranked by strength).
        'random_k' — random k-subset of living atom indices. Ablation
            comparator for §C of the checklist; tests whether smaller
            schema set alone is load-bearing, independent of strength
            filtering.
    k : int
        Target schema count. If the consolidation has fewer than k
        living atoms, returns however many exist (no padding).
    rng : torch.Generator, optional
        For 'random_k'. If None, uses the default torch RNG (caller
        controls determinism upstream).

    Returns
    -------
    schemas : torch.Tensor, shape [k', D] complex
    atom_idx : torch.Tensor, shape [k'] int64
        Indices into `patterns` of the selected atoms.
    """
    n = consolidation.n_patterns
    if n == 0:
        raise ValueError(
            "consolidation has no patterns; schema store cannot be built"
        )
    if patterns.shape[0] != n:
        raise ValueError(
            f"patterns has {patterns.shape[0]} rows but consolidation has "
            f"{n} living atoms; substrate and consolidation are out of sync"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k_eff = min(k, n)

    if selection_rule == "top_k_by_effective_strength":
        strength = consolidation.effective_strength().abs()
        atom_idx = torch.topk(strength, k_eff, largest=True).indices
    elif selection_rule == "random_k":
        perm = torch.randperm(n, generator=rng)
        atom_idx = perm[:k_eff]
    else:
        raise ValueError(
            f"unknown selection_rule {selection_rule!r}; expected "
            "'top_k_by_effective_strength' or 'random_k'"
        )

    schemas = patterns.index_select(0, atom_idx.to(patterns.device))
    return schemas, atom_idx


# =============================================================================
# Branch seeding
# =============================================================================

def _fhrr_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Re(<a, b>) / (||a|| ||b||) for complex FHRR vectors.

    Accepts a: [D] or [N, D], b: [D] or [N, D]; broadcasts as torch dot.
    Returns a scalar or vector of real cosines.
    """
    if a.dim() == 1 and b.dim() == 1:
        num = torch.dot(a.conj(), b).real
        den = a.norm() * b.norm()
        return num / den.clamp(min=1e-12)
    if a.dim() == 1 and b.dim() == 2:
        num = (b.conj() * a).sum(dim=-1).real
        den = b.norm(dim=-1) * a.norm()
        return num / den.clamp(min=1e-12)
    if a.dim() == 2 and b.dim() == 1:
        num = (a.conj() * b).sum(dim=-1).real
        den = a.norm(dim=-1) * b.norm()
        return num / den.clamp(min=1e-12)
    raise ValueError(f"unsupported shapes a={tuple(a.shape)} b={tuple(b.shape)}")


def select_schema_priors(
    *,
    cue: torch.Tensor,
    schema_store: torch.Tensor,
    k_main: int,
    delta_redundant: float = 0.95,
    prior_type: str = "content",
    cue_bindings: Optional[torch.Tensor] = None,
    schema_bindings: Optional[torch.Tensor] = None,
    rng: Optional[torch.Generator] = None,
) -> List[Tuple[int, torch.Tensor]]:
    """Pick K_main schemas with a diversity filter.

    Parameters
    ----------
    cue : [D] complex
    schema_store : [N, D] complex
    k_main : int
    delta_redundant : float
        Greedy diversity walk: any candidate whose max cosine to an
        already-selected schema exceeds this is skipped. delta_redundant=1.0
        disables the filter; 0.0 forces orthogonality.
    prior_type : {'content', 'role', 'random'}
        'content' : rank by FHRR cosine cue vs schema.
        'role'    : rank by mean-of-best-match role-binding similarity.
                    Requires cue_bindings [n_roles_cue, D] and
                    schema_bindings [N, n_roles_schema, D]. For each
                    cue binding b, find max_r cos(schema_bindings[i, r], b);
                    score = mean over cue bindings of that max.
        'random'  : score = independent uniform draws per schema.
                    Cue/bindings ignored. Anti-homunculus note: this is the
                    falsifier for "schemas matter" — if random priors
                    yield the same headline ΔE, no structural retrieval.
    cue_bindings : [n_roles_cue, D] complex, optional
        Role-decomposed cue (unbound fillers per role position).
    schema_bindings : [N, n_roles_schema, D] complex, optional
        Same decomposition for each schema.
    rng : torch.Generator, optional
        For prior_type='random'; otherwise unused.

    Returns
    -------
    list of (schema_index, prior_vector) of length <= k_main.
    Fewer than k_main returned when the diversity filter cannot find
    enough non-redundant candidates.
    """
    n_schemas = schema_store.shape[0]
    if n_schemas == 0:
        return []
    if k_main <= 0:
        return []

    if prior_type == "content":
        scores = _fhrr_cosine(cue, schema_store)
    elif prior_type == "role":
        if cue_bindings is None or schema_bindings is None:
            raise ValueError(
                "prior_type='role' requires cue_bindings and schema_bindings"
            )
        if schema_bindings.shape[0] != n_schemas:
            raise ValueError(
                f"schema_bindings has {schema_bindings.shape[0]} rows but "
                f"schema_store has {n_schemas}"
            )
        # Per cue binding, find best matching schema binding (max over schema
        # roles); average across cue bindings.
        cue_norm = cue_bindings.norm(dim=-1).clamp(min=1e-12)  # [n_roles_cue]
        sch_norm = schema_bindings.norm(dim=-1).clamp(min=1e-12)  # [N, n_roles_schema]
        # cue_bindings: [Rc, D]; schema_bindings: [N, Rs, D]
        # inner: [N, Rs, Rc] = Re(<sch[i, r, :], cue[b, :]>)
        inner = torch.einsum("nrd,bd->nrb", schema_bindings.conj(), cue_bindings).real
        cos = inner / (sch_norm.unsqueeze(-1) * cue_norm.unsqueeze(0).unsqueeze(0))
        # For each (schema i, cue binding b), max over schema roles r:
        best_per_cue_binding = cos.max(dim=1).values  # [N, Rc]
        scores = best_per_cue_binding.mean(dim=-1)  # [N]
    elif prior_type == "random":
        scores = torch.rand(n_schemas, generator=rng, device=schema_store.device)
    else:
        raise ValueError(
            f"unknown prior_type {prior_type!r}; expected "
            "'content' | 'role' | 'random'"
        )

    ranked = torch.argsort(scores, descending=True)

    selected: List[Tuple[int, torch.Tensor]] = []
    selected_vecs: List[torch.Tensor] = []
    for idx_t in ranked:
        idx = int(idx_t)
        candidate = schema_store[idx]
        redundant = False
        for prev in selected_vecs:
            sim = _fhrr_cosine(candidate, prev)
            if float(sim) > delta_redundant:
                redundant = True
                break
        if redundant:
            continue
        selected.append((idx, candidate))
        selected_vecs.append(candidate)
        if len(selected) >= k_main:
            break

    return selected


def surprise_prior(
    *,
    consolidation: ConsolidationState,
    codebook: torch.Tensor,
    eps: float = 1e-6,
) -> Optional[Tuple[int, torch.Tensor]]:
    """Return (pattern_idx, prior_vector) for the max-novelty pattern.

    novelty_score(k) = log(u_1[k] + eps) - log(u_m[k] + eps)

    Returns None if consolidation has no patterns (no surprise branch
    possible).
    """
    if consolidation.n_patterns == 0:
        return None
    u = consolidation.u  # [N, m]
    log_u1 = torch.log(u[:, 0].clamp(min=0) + eps)
    log_um = torch.log(u[:, -1].clamp(min=0) + eps)
    novelty = log_u1 - log_um
    idx = int(novelty.argmax().item())
    # The "prior vector" for the surprise branch is the codebook row for
    # the chosen pattern, or — once schemas are explicit — the schema
    # whose underlying atom is `idx`. For the skeleton, return codebook row.
    prior = codebook[idx].clone()
    return idx, prior


# =============================================================================
# Per-branch settling with γ-biased energy
# =============================================================================

def _softmax_entropy(weights: torch.Tensor) -> float:
    """Shannon entropy of a non-negative probability vector (nats)."""
    safe = weights.clamp(min=1e-12)
    return float(-(safe * safe.log()).sum().detach().cpu())


FORMULATIONS = ("per_pattern", "global_pull")


def settle_branch_with_prior(
    *,
    memory: TorchHopfieldMemory,
    cue: torch.Tensor,
    prior: torch.Tensor,
    beta: float,
    gamma: float,
    max_iter: int = 12,
    tol: float = 1e-8,
    formulation: str = "per_pattern",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Hopfield retrieval with prior bias. Two formulations are supported
    for the **decision #5 spike** (phase-5-unified-design.md §Open decisions):

    formulation='per_pattern' (default):
        s_i = β · Re(⟨X_i, q⟩) + γ · Re(⟨X_i, prior⟩)
        q_{t+1} = normalize(Σ_i softmax(s)_i · X_i)
      γ reweights *which stored patterns* the dynamics favor. q stays in
      the convex hull of stored patterns. Off-manifold priors do little.
      Matches design line 164 ("prior term added to the score").

    formulation='global_pull':
        E_k(q) = -logsumexp(β · X q*) / β  -  γ · Re(⟨q, prior⟩)
        q_{t+1} = normalize(Σ_i softmax(β · sim_i)_i · X_i  +  γ · prior)
      γ adds a constant pull on q toward the prior, *independent* of
      whether any stored pattern matches the prior. q can drift between
      basins or off-substrate. Matches design line 158's literal energy.

    Both formulations reduce exactly to the unbiased retrieve at γ=0.
    Decision #5 is which one Phase 5 commits to for graduation runs;
    the spike runs both on synthetic role-binding cues to decide.

    Convergence is gated by the formulation's own biased-energy
    (Lyapunov function under the respective dynamics). Reported
    entropies use unbiased weights so they are comparable across
    formulations and γ values.

    Returns
    -------
    q_settled : [D] complex
    telemetry : dict
      score_entropy_initial, score_entropy_final   (UNBIASED softmax, comparable)
      converged, iterations,
      energy_unbiased_final                        (the scoring energy)
      energy_biased_final                          (the dynamics' Lyapunov value at q*)
      on_substrate_alignment                       (max sim(q*, X_i) — the
        falsifier for global-pull: low alignment means q drifted off-substrate.)
      formulation                                  (echo, for tagging)
    """
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")
    if formulation not in FORMULATIONS:
        raise ValueError(
            f"unknown formulation {formulation!r}; expected one of {FORMULATIONS}"
        )
    if not memory.stored_count:
        raise ValueError("cannot settle on an empty Hopfield memory")

    patterns = memory._pattern_matrix()
    substrate = memory.substrate
    device = substrate.device
    state = cue.to(device)
    prior_dev = prior.to(device)

    # Per-formulation precomputation.
    if formulation == "per_pattern":
        prior_bias = gamma * substrate.similarity_matrix(prior_dev, patterns)  # [N]
    else:
        prior_bias = None  # global pull adds to update vector, not logits

    # Initial unbiased entropy (diagnostic).
    init_scores = substrate.similarity_matrix(state, patterns)
    init_weights_unbiased = torch.softmax(beta * init_scores, dim=0)
    score_entropy_initial = _softmax_entropy(init_weights_unbiased)

    prev_biased_energy: Optional[torch.Tensor] = None
    final_state = state
    frozen = torch.zeros((), dtype=torch.bool, device=device)
    biased_energy_tensors: List[torch.Tensor] = []
    for _ in range(max_iter):
        scores = substrate.similarity_matrix(state, patterns)
        # Compute logits (used for weighting) and biased energy (Lyapunov).
        if formulation == "per_pattern":
            biased_logits = beta * scores + prior_bias
            weights = torch.softmax(biased_logits, dim=0)
            update_vec = (patterns * weights[:, None]).sum(dim=0)
            # Biased energy under per-pattern: -logsumexp(beta*scores + prior_bias)/β
            biased_energy = -torch.logsumexp(biased_logits, dim=0) / beta
        else:  # global_pull
            unbiased_logits = beta * scores
            weights = torch.softmax(unbiased_logits, dim=0)
            update_vec = (patterns * weights[:, None]).sum(dim=0) + gamma * prior_dev
            # Biased energy under global pull: -logsumexp(beta*scores)/β - γ·Re(<q, prior>)
            q_prior_inner = (state.conj() * prior_dev).sum().real
            biased_energy = (
                -torch.logsumexp(unbiased_logits, dim=0) / beta - gamma * q_prior_inner
            )
        biased_energy_tensors.append(biased_energy)
        next_state = substrate.normalize(update_vec)
        if prev_biased_energy is not None:
            converged_now = (
                (biased_energy - prev_biased_energy).abs() < tol
            ) & (~frozen)
            final_state = torch.where(converged_now, next_state, final_state)
            frozen = frozen | converged_now
        prev_biased_energy = biased_energy
        state = next_state
    final_state = torch.where(frozen, final_state, state)
    state = final_state

    # Diagnostics at the final state (all use unbiased softmax for comparability).
    final_scores = substrate.similarity_matrix(state, patterns)
    final_weights_unbiased = torch.softmax(beta * final_scores, dim=0)
    score_entropy_final = _softmax_entropy(final_weights_unbiased)
    on_substrate_alignment = float(final_scores.max().detach().cpu())

    # Single batched sync for energies + converged flag.
    biased_energies = torch.stack(biased_energy_tensors).detach().cpu().tolist()
    converged = False
    for k in range(1, len(biased_energies)):
        if abs(biased_energies[k] - biased_energies[k - 1]) < tol:
            converged = True
            break

    energy_unbiased_final = float(
        (-torch.logsumexp(beta * final_scores, dim=0) / beta).detach().cpu()
    )
    if formulation == "per_pattern":
        energy_biased_final = float(
            (-torch.logsumexp(beta * final_scores + prior_bias, dim=0) / beta).detach().cpu()
        )
    else:
        q_prior_inner_final = float((state.conj() * prior_dev).sum().real.detach().cpu())
        energy_biased_final = energy_unbiased_final - gamma * q_prior_inner_final

    return state, {
        "score_entropy_initial": score_entropy_initial,
        "score_entropy_final": score_entropy_final,
        "converged": converged,
        "iterations": len(biased_energies),
        "energy_unbiased_final": energy_unbiased_final,
        "energy_biased_final": energy_biased_final,
        "on_substrate_alignment": on_substrate_alignment,
        "formulation": formulation,
    }


# =============================================================================
# Per-branch diagnostics (logged, NOT used for selection)
# =============================================================================

def _unbiased_energy(
    memory: TorchHopfieldMemory, state: torch.Tensor, beta: float,
) -> float:
    """E(q) = -logsumexp(β · sim(X, q)) / β as a Python float."""
    patterns = memory._pattern_matrix()
    scores = memory.substrate.similarity_matrix(state.to(memory.substrate.device), patterns)
    return float((-torch.logsumexp(beta * scores, dim=0) / beta).detach().cpu())


def compute_branch_diagnostics(
    *,
    branch: BranchState,
    memory: TorchHopfieldMemory,
    cue: torch.Tensor,
    beta: float,
    gamma: float,
    target_id: Optional[int],
    codebook: torch.Tensor,
    positions: torch.Tensor,
    decode_ids: Sequence[int],
    decode_k: int,
    masked_pos: int,
    cue_bindings: Optional[torch.Tensor] = None,
    settling_telemetry: Optional[Dict[str, float]] = None,
) -> None:
    """Fill BranchState diagnostic fields in-place.

    Anti-homunculus discipline: all fields here are *measurements* of the
    settled state. None of them feed back into the selection rule
    (branch combination uses energy_unbiased only).

    Field semantics:
      energy_unbiased         — -logsumexp(β · sim(X, q*)) / β   (scoring)
      energy_biased           — the dynamics' Lyapunov value at q*. Pulled
                                from settling_telemetry if available;
                                otherwise approximated as
                                energy_unbiased - γ · Re(⟨q*, prior⟩)
                                (the design-line-158 "global" gloss; under
                                per-pattern formulation, prefer the telemetry).
      energy_drop             — energy_unbiased(q_initial) - energy_unbiased(q*)
      prior_alignment         — Re(⟨q*, prior⟩) / (‖q*‖·‖prior‖)
      score_entropy_initial   — H(softmax(β · sim(X, q_initial)))
      score_entropy_final     — H(softmax(β · sim(X, q*)))
      entropy_collapse        — score_entropy_initial - score_entropy_final
      recall_support          — target_id in top-decode at masked_pos?
                                (False when target_id is None — caller
                                semantics, not a "missing" sentinel.)
      cap_coverage_t05        — top decode score >= 0.5 AND target in top-K
      meta_stable             — top decode score < 0.95 (Phase 2 convention)
      structural_match        — mean cosine sim between q*'s position-unbound
                                fillers and cue_bindings, if cue_bindings
                                provided. 0.0 otherwise.
      converged               — settling reached convergence (from telemetry
                                if available, else recomputed approximately).
    """
    substrate = memory.substrate
    patterns = memory._pattern_matrix()
    q_star = branch.q_settled.to(substrate.device)
    q_init = branch.q_initial.to(substrate.device)
    prior = branch.prior.to(substrate.device)

    # Energy fields ---------------------------------------------------------
    branch.energy_unbiased = _unbiased_energy(memory, q_star, beta)
    if settling_telemetry is not None and "energy_biased_final" in settling_telemetry:
        branch.energy_biased = float(settling_telemetry["energy_biased_final"])
    else:
        q_prior_inner = float((q_star.conj() * prior).sum().real.detach().cpu())
        branch.energy_biased = branch.energy_unbiased - gamma * q_prior_inner
    branch.energy_drop = _unbiased_energy(memory, q_init, beta) - branch.energy_unbiased

    # Prior alignment -------------------------------------------------------
    q_norm = float(q_star.norm().detach().cpu())
    p_norm = float(prior.norm().detach().cpu())
    inner = float((q_star.conj() * prior).sum().real.detach().cpu())
    branch.prior_alignment = (
        inner / max(q_norm * p_norm, 1e-12) if p_norm > 0.0 else 0.0
    )

    # Entropy fields --------------------------------------------------------
    if settling_telemetry is not None and "score_entropy_initial" in settling_telemetry:
        branch.score_entropy_initial = float(settling_telemetry["score_entropy_initial"])
        branch.score_entropy_final = float(settling_telemetry["score_entropy_final"])
    else:
        init_scores = substrate.similarity_matrix(q_init, patterns)
        final_scores = substrate.similarity_matrix(q_star, patterns)
        branch.score_entropy_initial = _softmax_entropy(torch.softmax(beta * init_scores, dim=0))
        branch.score_entropy_final = _softmax_entropy(torch.softmax(beta * final_scores, dim=0))
    branch.entropy_collapse = branch.score_entropy_initial - branch.score_entropy_final

    # Convergence -----------------------------------------------------------
    if settling_telemetry is not None and "converged" in settling_telemetry:
        branch.converged = bool(settling_telemetry["converged"])

    # Readout fields: decode at masked_pos ---------------------------------
    if positions is not None and codebook is not None and len(decode_ids) > 0:
        decoded = decode_position(
            substrate=substrate,
            state=q_star,
            position=positions[masked_pos],
            codebook=codebook,
            candidate_ids=list(decode_ids),
            top_k=decode_k,
        )
        if decoded:
            top_id, top_score = decoded[0]
            branch.meta_stable = bool(top_score < 0.95)
            if target_id is not None:
                ids_in_top = {tid for tid, _ in decoded}
                branch.recall_support = bool(target_id in ids_in_top)
                branch.cap_coverage_t05 = float(
                    (top_score >= 0.5) and (target_id in ids_in_top)
                )
            else:
                branch.recall_support = False
                branch.cap_coverage_t05 = 0.0
        else:
            branch.meta_stable = False
            branch.recall_support = False
            branch.cap_coverage_t05 = 0.0

    # Structural-match (role-binding decomposition) ------------------------
    if cue_bindings is not None and positions is not None:
        # positions may be a list of [D] tensors or a stacked [N, D] tensor.
        n_pos = positions.shape[0] if isinstance(positions, torch.Tensor) else len(positions)
        n_roles = min(cue_bindings.shape[0], n_pos)
        sims = []
        for r in range(n_roles):
            pos_r = positions[r]
            unbound = substrate.unbind(q_star, pos_r)
            sims.append(float(_fhrr_cosine(unbound, cue_bindings[r].to(substrate.device))))
        branch.structural_match = float(sum(sims) / len(sims)) if sims else 0.0
    else:
        branch.structural_match = 0.0


def compute_pairwise_final_state_divergence(branches: List[BranchState]) -> None:
    """Fill BranchState.final_state_divergence in-place: mean cosine-distance
    from each branch's q_settled to every other branch's q_settled.

    K=1: divergence is 0.0 by convention (no peers).
    """
    n = len(branches)
    if n <= 1:
        for b in branches:
            b.final_state_divergence = 0.0
        return
    states = [b.q_settled for b in branches]
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            cos = float(_fhrr_cosine(states[i], states[j]))
            dists.append(1.0 - cos)
        branches[i].final_state_divergence = float(sum(dists) / len(dists))


# =============================================================================
# Branch combination — three rules, run all three for comparison
# =============================================================================

def _branch_softmax_weights(
    branches: List[BranchState], temperature: float
) -> torch.Tensor:
    """w_k = softmax(-E_k_unbiased / τ) as a real tensor."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    energies = torch.tensor([b.energy_unbiased for b in branches], dtype=torch.float32)
    return torch.softmax(-energies / temperature, dim=0)


def combine_bundle_resettle(
    *,
    branches: List[BranchState],
    memory: TorchHopfieldMemory,
    beta: float,
    temperature: float = 1.0,
    max_iter: int = 12,
    tol: float = 1e-8,
) -> Tuple[torch.Tensor, List[float], bool]:
    """PREFERRED combination rule (anti-homunculus-clean).

    q_bundle = Σ_k w_k · q_k_settled,  w_k = softmax(-E_k_unbiased / τ)
    q_final  = settle_branch_with_prior(... γ=0)(q_bundle)   # UNBIASED re-settle

    Anti-homunculus reading: the bundle is an algebraic sum; the re-settle
    is a standard Hopfield gradient descent on the marginal energy with
    γ=0 (no prior contribution leaks in — the prior already shaped which
    state each branch landed in, but it does not bias the *combination*).

    Returns (q_final, softmax_weights, converged).
    """
    if not branches:
        raise ValueError("cannot combine an empty branch list")
    weights = _branch_softmax_weights(branches, temperature)
    # Bundle in FHRR space then normalize (substrate.weighted_bundle is the
    # canonical primitive; we recreate it here to keep complex dtype and
    # avoid going through a list-of-tensors conversion).
    states = torch.stack([b.q_settled.to(memory.substrate.device) for b in branches], dim=0)
    bundled = memory.substrate.normalize(
        (states * weights.to(states.device)[:, None]).sum(dim=0)
    )
    # Re-settle UNBIASED. We deliberately use γ=0; a zero prior is fine
    # because gamma=0 means it's ignored anyway. Per-pattern is the safe
    # default formulation for the re-settle regardless of which formulation
    # the original branches used — re-settle is meant to find the marginal
    # attractor under the substrate alone.
    zero_prior = torch.zeros_like(bundled)
    q_final, telem = settle_branch_with_prior(
        memory=memory, cue=bundled, prior=zero_prior,
        beta=beta, gamma=0.0, max_iter=max_iter, tol=tol,
        formulation="per_pattern",
    )
    return q_final, weights.detach().cpu().tolist(), bool(telem["converged"])


def combine_greedy_argmin(branches: List[BranchState]) -> Tuple[torch.Tensor, int]:
    """BASELINE combination: q_final = q_k* for k = argmin_k E_k_unbiased.

    Allowed only as a comparison condition (see design §atom-splitting and
    the anti-homunculus check). Returns (q_final, picked_index).
    Ties broken by torch.argmin convention (first index).
    """
    if not branches:
        raise ValueError("cannot combine an empty branch list")
    energies = torch.tensor([b.energy_unbiased for b in branches], dtype=torch.float32)
    k = int(torch.argmin(energies).item())
    return branches[k].q_settled, k


def combine_boltzmann_sample(
    branches: List[BranchState],
    temperature: float = 1.0,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, int, List[float]]:
    """COMPARISON combination (FEP-clean Boltzmann sampling).

    Sample k ~ Categorical(softmax(-E_k / τ)); return (q_k*, k, weights).
    """
    if not branches:
        raise ValueError("cannot combine an empty branch list")
    weights = _branch_softmax_weights(branches, temperature)
    # torch.multinomial respects a torch.Generator.
    k = int(torch.multinomial(weights, num_samples=1, generator=rng).item())
    return branches[k].q_settled, k, weights.detach().cpu().tolist()


# =============================================================================
# Atom-splitting joint diagnostic
# =============================================================================

def atom_split_signal(
    branches: List[BranchState],
    *,
    delta_energy: float = 0.1,
    delta_state: float = 0.3,
) -> Tuple[bool, int, float]:
    """Return (split_eligible, n_in_low_energy_set, max_state_distance).

    Joint criterion: both conditions must hold for split_eligible=True.
    1. At least 2 branches with E_k - E_min < delta_energy
       (comparable low energies).
    2. Among those, the maximum pairwise cosine-distance among settled
       states exceeds delta_state (substantial geometric divergence).

    A signal under both axes is the FHRR-space signature of an atom
    that supports two genuinely different meanings, not a redundant
    pair of paths to one meaning.
    """
    if len(branches) < 2:
        return False, 0, 0.0
    energies = torch.tensor([b.energy_unbiased for b in branches])
    e_min = energies.min().item()
    low_energy_mask = (energies - e_min) < delta_energy
    n_low = int(low_energy_mask.sum().item())
    if n_low < 2:
        return False, n_low, 0.0
    low_states = [branches[i].q_settled for i in range(len(branches)) if low_energy_mask[i].item()]
    max_dist = 0.0
    for i in range(len(low_states)):
        for j in range(i + 1, len(low_states)):
            a = low_states[i]
            b = low_states[j]
            cos = (torch.dot(a.conj(), b).real / (a.norm() * b.norm())).item()
            dist = 1.0 - cos
            if dist > max_dist:
                max_dist = dist
    return max_dist > delta_state, n_low, max_dist


# =============================================================================
# Headline experiment driver
# =============================================================================

def run_branched_retrieval(
    *,
    cue: torch.Tensor,
    cue_id: int,
    target_id: Optional[int],
    memory: TorchHopfieldMemory,
    codebook: Optional[torch.Tensor],
    positions: Optional[Sequence[torch.Tensor]],
    decode_ids: Sequence[int],
    decode_k: int,
    masked_pos: int,
    schema_store: torch.Tensor,
    schema_atom_idx: Optional[torch.Tensor],
    consolidation: ConsolidationState,
    prior_type: str,
    k_main: int,
    gamma: float,
    beta: float,
    temperature: float,
    delta_energy: float,
    delta_state: float,
    delta_redundant: float,
    formulation: str = "per_pattern",
    cue_bindings: Optional[torch.Tensor] = None,
    schema_bindings: Optional[torch.Tensor] = None,
    include_surprise_branch: bool = True,
    max_settling_iter: int = 12,
    boltzmann_rng: Optional[torch.Generator] = None,
    random_prior_rng: Optional[torch.Generator] = None,
) -> BranchedRetrievalResult:
    """One full branched retrieval over a single cue.

    Steps (per design §"Architectural diagram"):
      1. Pick K_main schemas (via prior_type) + optional surprise branch.
      2. Settle each branch with γ-biased energy (formulation per
         decision #5).
      3. Score unbiased; log per-branch diagnostics.
      4. Compute pairwise final-state divergence (post-pass).
      5. Run all three combination rules: bundle-resettle (preferred),
         greedy-argmin (baseline), Boltzmann (FEP-clean comparison).
      6. Compute atom-split signal.

    Anti-homunculus: selection (which q* is the "answer") flows through
    the bundle-resettle combiner — energy-weighted sum + unbiased
    re-settle. The greedy and Boltzmann variants are logged for
    comparison only; they are NOT the production combiner.
    """
    result = BranchedRetrievalResult(cue_id=cue_id, cue=cue, target_id=target_id)

    # --- Step 1: pick K_main schemas + surprise branch -------------------
    picks = select_schema_priors(
        cue=cue, schema_store=schema_store, k_main=k_main,
        delta_redundant=delta_redundant, prior_type=prior_type,
        cue_bindings=cue_bindings, schema_bindings=schema_bindings,
        rng=random_prior_rng,
    )

    surprise_pick: Optional[Tuple[int, torch.Tensor]] = None
    if include_surprise_branch and codebook is not None:
        surprise_pick = surprise_prior(consolidation=consolidation, codebook=codebook)

    # --- Step 2 + 3: settle each branch and fill diagnostics ------------
    branches: List[BranchState] = []
    for i, (schema_idx, prior_vec) in enumerate(picks):
        q_settled, telem = settle_branch_with_prior(
            memory=memory, cue=cue, prior=prior_vec,
            beta=beta, gamma=gamma, max_iter=max_settling_iter,
            formulation=formulation,
        )
        b = BranchState(
            branch_id=i,
            prior_source=prior_type,
            prior=prior_vec,
            q_initial=cue,
            q_settled=q_settled,
        )
        compute_branch_diagnostics(
            branch=b, memory=memory, cue=cue, beta=beta, gamma=gamma,
            target_id=target_id, codebook=codebook, positions=positions,
            decode_ids=decode_ids, decode_k=decode_k, masked_pos=masked_pos,
            cue_bindings=cue_bindings, settling_telemetry=telem,
        )
        branches.append(b)

    if surprise_pick is not None:
        _, prior_vec = surprise_pick
        q_settled, telem = settle_branch_with_prior(
            memory=memory, cue=cue, prior=prior_vec,
            beta=beta, gamma=gamma, max_iter=max_settling_iter,
            formulation=formulation,
        )
        b = BranchState(
            branch_id=len(branches),
            prior_source="surprise",
            prior=prior_vec,
            q_initial=cue,
            q_settled=q_settled,
        )
        compute_branch_diagnostics(
            branch=b, memory=memory, cue=cue, beta=beta, gamma=gamma,
            target_id=target_id, codebook=codebook, positions=positions,
            decode_ids=decode_ids, decode_k=decode_k, masked_pos=masked_pos,
            cue_bindings=cue_bindings, settling_telemetry=telem,
        )
        branches.append(b)

    result.branches = branches

    # --- Step 4: pairwise final-state divergence -------------------------
    compute_pairwise_final_state_divergence(branches)

    # --- Step 5: all three combination rules ----------------------------
    if branches:
        q_bundle, weights, _conv = combine_bundle_resettle(
            branches=branches, memory=memory, beta=beta,
            temperature=temperature, max_iter=max_settling_iter,
        )
        result.q_bundle = q_bundle
        result.softmax_weights = weights
        result.softmax_entropy = _softmax_entropy(torch.tensor(weights))

        q_greedy, _k_greedy = combine_greedy_argmin(branches)
        result.q_greedy = q_greedy

        q_boltz, _k_boltz, _w_boltz = combine_boltzmann_sample(
            branches, temperature=temperature, rng=boltzmann_rng,
        )
        result.q_boltzmann = q_boltz

    # --- Step 6: atom-split signal --------------------------------------
    ok, n_low, max_dist = atom_split_signal(
        branches, delta_energy=delta_energy, delta_state=delta_state,
    )
    result.split_eligible = ok
    result.n_in_low_energy_set = n_low
    result.max_state_distance_in_low_energy_set = max_dist

    return result


# =============================================================================
# Conditions and aggregator
# =============================================================================

CONDITIONS = [
    # name              prior_type  gamma  k_main  notes
    ("content_K4_g0p5", "content",   0.5,  4,     "default content-prior branching"),
    ("role_K4_g0p5",    "role",      0.5,  4,     "default role-prior branching (HEADLINE COMPARISON)"),
    ("random_K4_g0p5",  "random",    0.5,  4,     "random-schema control"),
    ("content_K1_g0p5", "content",   0.5,  1,     "no-branching control (K=1)"),
    ("content_K4_g0",   "content",   0.0,  4,     "no-prior control (γ=0)"),
    # γ sweep on content-prior branching:
    ("content_K4_g025", "content",   0.25, 4,     "γ sweep"),
    ("content_K4_g1",   "content",   1.0,  4,     "γ sweep"),
    ("content_K4_g2",   "content",   2.0,  4,     "γ sweep — supercritical, watch for prior-domination"),
    # K sweep on content-prior branching:
    ("content_K2_g0p5", "content",   0.5,  2,     "K sweep"),
    ("content_K8_g0p5", "content",   0.5,  8,     "K sweep — most expensive"),
]


# Smoke-runnable subset of CONDITIONS — no role-binding required.
# Used by main() when --mode=smoke.
SMOKE_CONDITIONS = [
    ("content_K4_g0p5", "content", 0.5, 4),
    ("random_K4_g0p5", "random", 0.5, 4),
    ("content_K1_g0p5", "content", 0.5, 1),  # no-branching control
    ("content_K4_g0", "content", 0.0, 4),    # no-prior control
]


def _build_synthetic_substrate(*, n_atoms: int, dim: int, device: str, seed: int):
    """Build a small (memory, consolidation, patterns) trio for smoke runs.

    Not a Phase 4 substrate — just enough to exercise the Phase 5 driver
    end-to-end. The real run loads a saved post-death snapshot via
    `_load_substrate_from_snapshot`.
    """
    from energy_memory.phase4.consolidation import (
        ConsolidationConfig, ConsolidationState,
    )
    torch.manual_seed(seed)
    substrate = TorchFHRR(dim=dim, device=device)
    mem = TorchHopfieldMemory(substrate)
    cons = ConsolidationState(ConsolidationConfig(m=4, alpha=0.25), device=device)
    patterns = []
    for i in range(n_atoms):
        p = substrate.normalize(torch.randn(dim, dtype=torch.complex64, device=device))
        mem.store(p, label=i)
        # Spread effective strength so top-k vs random-k differ meaningfully.
        cons.add_pattern(novelty_strength=1.0 + 0.2 * (n_atoms - i))
        patterns.append(p)
    return mem, cons, patterns


def _load_substrate_from_snapshot(*, path: str, device: str):
    """Load (memory, consolidation, patterns) from a Phase 4 snapshot.

    Constructs a TorchFHRR substrate matching the snapshot's dim. Returns
    (memory, consolidation, patterns, info) where info carries the
    snapshot's label/metadata for downstream reporting.
    """
    from energy_memory.phase4.snapshot import load_substrate_snapshot
    # Peek the dim from the saved patterns tensor.
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved_patterns = state["patterns"]
    if saved_patterns.numel() == 0:
        raise ValueError(f"snapshot {path} has zero stored patterns; nothing to load")
    dim = saved_patterns.shape[-1]
    substrate = TorchFHRR(dim=dim, device=device)
    mem, cons, info = load_substrate_snapshot(
        path=path, substrate=substrate, device=device,
    )
    patterns = list(mem._patterns)
    return mem, cons, patterns, info


def _run_condition_over_cues(
    *,
    name: str,
    prior_type: str,
    gamma: float,
    k_main: int,
    mem: TorchHopfieldMemory,
    cons: ConsolidationState,
    patterns: List[torch.Tensor],
    schema_store: torch.Tensor,
    schema_atom_idx: torch.Tensor,
    cues: List[torch.Tensor],
    beta: float,
    temperature: float,
    delta_energy: float,
    delta_state: float,
    delta_redundant: float,
    formulation: str,
    boltzmann_rng: torch.Generator,
    random_prior_rng: torch.Generator,
) -> Dict:
    """Run one condition across all cues; aggregate per-cue diagnostics."""
    per_cue_energy_unbiased_min = []
    per_cue_branch_count = []
    per_cue_split_eligible = []
    per_cue_on_substrate_alignment = []
    per_cue_softmax_entropy = []
    for cue_id, cue in enumerate(cues):
        result = run_branched_retrieval(
            cue=cue, cue_id=cue_id, target_id=None,
            memory=mem, codebook=mem._pattern_matrix(), positions=None,
            decode_ids=[], decode_k=5, masked_pos=0,
            schema_store=schema_store, schema_atom_idx=schema_atom_idx,
            consolidation=cons, prior_type=prior_type, k_main=k_main,
            gamma=gamma, beta=beta, temperature=temperature,
            delta_energy=delta_energy, delta_state=delta_state,
            delta_redundant=delta_redundant, formulation=formulation,
            include_surprise_branch=True,
            boltzmann_rng=boltzmann_rng, random_prior_rng=random_prior_rng,
        )
        if not result.branches:
            continue
        e_min = min(b.energy_unbiased for b in result.branches)
        align = max(
            float(mem.substrate.similarity(b.q_settled, p))
            for b in result.branches for p in patterns
        )
        per_cue_energy_unbiased_min.append(e_min)
        per_cue_branch_count.append(len(result.branches))
        per_cue_split_eligible.append(int(result.split_eligible))
        per_cue_on_substrate_alignment.append(align)
        per_cue_softmax_entropy.append(result.softmax_entropy)
    n = max(len(per_cue_energy_unbiased_min), 1)
    return {
        "name": name,
        "prior_type": prior_type,
        "gamma": gamma,
        "k_main": k_main,
        "formulation": formulation,
        "n_cues": len(per_cue_energy_unbiased_min),
        "mean_energy_unbiased_min": sum(per_cue_energy_unbiased_min) / n,
        "mean_branch_count": sum(per_cue_branch_count) / n,
        "split_eligibility_rate": sum(per_cue_split_eligible) / n,
        "mean_on_substrate_alignment": sum(per_cue_on_substrate_alignment) / n,
        "mean_branch_softmax_entropy": sum(per_cue_softmax_entropy) / n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=str, default="smoke",
                        choices=["smoke", "decision5_spike"],
                        help="smoke: synthetic substrate, SMOKE_CONDITIONS. "
                        "decision5_spike: both formulations × γ ∈ {0.25, 0.5, 1.0}.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--substrate-snapshot", type=str, default=None,
        help=(
            "Path to a Phase 4 substrate snapshot .pt (from exp 19 with "
            "--snapshot-steps). When provided, the synthetic substrate is "
            "bypassed and Phase 5 runs against the loaded (memory, "
            "consolidation). --dim and --n-atoms are then ignored."
        ),
    )
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-atoms", type=int, default=12)
    parser.add_argument("--n-cues", type=int, default=20)
    parser.add_argument("--k", type=int, default=8, help="schema store size (k)")
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--delta-energy", type=float, default=0.1)
    parser.add_argument("--delta-state", type=float, default=0.3)
    parser.add_argument("--delta-redundant", type=float, default=0.95)
    parser.add_argument("--output-dir", type=str, default="reports/phase5_branching_smoke")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_info: Optional[Dict[str, Any]] = None
    if args.substrate_snapshot is not None:
        mem, cons, patterns, snapshot_info = _load_substrate_from_snapshot(
            path=args.substrate_snapshot, device=args.device,
        )
        effective_dim = mem.substrate.dim
        print(
            f"[load] substrate snapshot from {args.substrate_snapshot}: "
            f"n_atoms={len(patterns)}, dim={effective_dim}, "
            f"label={snapshot_info.get('label')!r}",
            flush=True,
        )
    else:
        mem, cons, patterns = _build_synthetic_substrate(
            n_atoms=args.n_atoms, dim=args.dim, device=args.device, seed=args.seed,
        )
        effective_dim = args.dim

    schema_store, atom_idx = get_schema_store(
        consolidation=cons, patterns=mem._pattern_matrix(),
        selection_rule="top_k_by_effective_strength",
        k=min(args.k, len(patterns)),
    )

    # Cues: perturbed copies of stored patterns (so retrieval has work to do).
    cues = []
    for i in range(args.n_cues):
        base = patterns[i % len(patterns)]
        noise = torch.randn(effective_dim, dtype=torch.complex64, device=args.device)
        cues.append(mem.substrate.normalize(base + 0.15 * noise))

    boltzmann_rng = torch.Generator().manual_seed(args.seed + 10)
    random_prior_rng = torch.Generator().manual_seed(args.seed + 20)

    if args.mode == "smoke":
        runs = [
            (name, prior_type, gamma, k_main, "per_pattern")
            for (name, prior_type, gamma, k_main) in SMOKE_CONDITIONS
        ]
    else:  # decision5_spike — formulation × γ grid (per_pattern + global_pull)
        runs = []
        for formulation in FORMULATIONS:
            for gamma in (0.25, 0.5, 1.0):
                name = f"content_K4_g{gamma}_{formulation}"
                runs.append((name, "content", gamma, 4, formulation))

    results = []
    for (name, prior_type, gamma, k_main, formulation) in runs:
        print(f"[run] {name} prior={prior_type} γ={gamma} K={k_main} form={formulation}")
        agg = _run_condition_over_cues(
            name=name, prior_type=prior_type, gamma=gamma, k_main=k_main,
            mem=mem, cons=cons, patterns=patterns,
            schema_store=schema_store, schema_atom_idx=atom_idx,
            cues=cues, beta=args.beta, temperature=args.temperature,
            delta_energy=args.delta_energy, delta_state=args.delta_state,
            delta_redundant=args.delta_redundant, formulation=formulation,
            boltzmann_rng=boltzmann_rng, random_prior_rng=random_prior_rng,
        )
        results.append(agg)

    out_path = output_dir / f"phase5_{args.mode}_seed{args.seed}.json"
    payload = {
        "mode": args.mode,
        "seed": args.seed,
        "dim": effective_dim,
        "n_atoms": len(patterns),
        "n_cues": args.n_cues,
        "k": args.k,
        "beta": args.beta,
        "temperature": args.temperature,
        "substrate_snapshot": args.substrate_snapshot,
        "snapshot_info": snapshot_info,
        "conditions": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[done] wrote {out_path}")
    return payload


if __name__ == "__main__":
    main()
