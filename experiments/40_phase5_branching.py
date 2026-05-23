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
    energy_unbiased: float = 0.0    # E_k^unbiased(q_settled) — raw landscape (back-compat with reports 047-053)
    energy_unbiased_step3: float = 0.0  # E_k^step3(q_settled) = -logsumexp(β·sim − score_bias)/β.
                                    # Step-3-weighted landscape per report 046 / Phase 5 design.
                                    # Equals energy_unbiased when score_bias is None.
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
    schema_fidelities: Optional[torch.Tensor] = None,
    p: float = 1.0,
    q: float = 1.0,
    substrate: Optional[TorchFHRR] = None,
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
    prior_type : {'content', 'role', 'random', 'fidelity_weighted'}
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
        'fidelity_weighted' : Path-3 β prior. Continuous weighted sum over
                    the FULL schema_store (no top-k cutoff, no diversity
                    filter): prior = normalize(Σ_i (cue·s_i)_+^p · f_i^q · s_i)
                    where f_i = mean pairwise FHRR distance of unbound
                    fillers. Returns ONE prior regardless of k_main.
                    Requires schema_fidelities [N] and substrate; uses
                    p, q parameters. See
                    notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md
                    and src/energy_memory/phase5/role_fidelity.py.
    cue_bindings : [n_roles_cue, D] complex, optional
        Role-decomposed cue (unbound fillers per role position).
    schema_bindings : [N, n_roles_schema, D] complex, optional
        Same decomposition for each schema.
    schema_fidelities : [N] float32, optional
        Per-schema role-fidelity in [0, 1] for prior_type='fidelity_weighted'.
    p, q : float, default 1.0
        β-prior exponents for prior_type='fidelity_weighted'.
    substrate : TorchFHRR, optional
        For β-prior normalization. If omitted under
        prior_type='fidelity_weighted', falls back to element-wise unit-
        magnitude normalization.
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

    if prior_type == "fidelity_weighted":
        # β prior: one continuous-weighted vector, no per-atom branching.
        from energy_memory.phase5.role_fidelity import fidelity_weighted_prior
        if schema_fidelities is None:
            raise ValueError(
                "prior_type='fidelity_weighted' requires schema_fidelities"
            )
        if schema_fidelities.shape != (n_schemas,):
            raise ValueError(
                f"schema_fidelities must be [N={n_schemas}], got "
                f"{tuple(schema_fidelities.shape)}"
            )
        prior_vec = fidelity_weighted_prior(
            cue=cue, schemas=schema_store, fidelities=schema_fidelities,
            p=p, q=q, substrate=substrate,
        )
        # Synthetic schema_idx = -1 (β prior is not tied to any one atom).
        return [(-1, prior_vec)]

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
        # Generate on CPU (where the seeded Generator lives) then move to the
        # schema_store's device. torch.rand requires generator.device ==
        # device, and the seeded rng here is a CPU Generator for reproducibility.
        scores = torch.rand(n_schemas, generator=rng).to(schema_store.device)
    else:
        raise ValueError(
            f"unknown prior_type {prior_type!r}; expected "
            "'content' | 'role' | 'random' | 'fidelity_weighted'"
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
# Role-binding cue generation (headline experiment input)
# =============================================================================

def compute_schema_bindings(
    *,
    substrate: TorchFHRR,
    schemas: torch.Tensor,
    positions: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Role-decompose each schema into per-position fillers.

    For schema s_i and position r, the unbound filler is
    ``substrate.unbind(s_i, positions[r])``. This recovers the noisy
    filler the schema had bound at that position (assuming the schema
    was originally encoded as ``bundle(filler_r ⊛ positions[r] for r)``,
    which is how Phase 4 stored windows via
    ``phase2.encoding.encode_window``).

    Returns a [n_schemas, n_roles, D] complex tensor.
    """
    n_schemas = schemas.shape[0]
    n_roles = len(positions)
    d = schemas.shape[-1]
    out = torch.zeros(
        n_schemas, n_roles, d,
        dtype=schemas.dtype, device=schemas.device,
    )
    for i in range(n_schemas):
        for r in range(n_roles):
            out[i, r] = substrate.unbind(schemas[i], positions[r].to(schemas.device))
    return out


def generate_role_binding_cue(
    *,
    substrate: TorchFHRR,
    positions: Sequence[torch.Tensor],
    role_target_schema: torch.Tensor,
    content_distractor: Optional[torch.Tensor] = None,
    binding_noise_std: float = 0.05,
    content_distortion: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a structural-retrieval test cue.

    The cue is constructed so that *role-binding* similarity to
    `role_target_schema` is high, while *content* similarity is biased
    toward `content_distractor`. This is the headline test for Phase 5:
    a role-prior schema selector should pick the role-matched schema;
    a content-prior selector should pick the content-distractor.

    Mechanism:
        1. Extract role_target's fillers per position via unbind.
        2. Add small Gaussian noise (binding_noise_std) so the cue's
           fillers are not literally identical to the schema's.
        3. Re-bundle to form the "structural" component.
        4. Mix with content_distractor at weight `content_distortion`
           and normalize.

    Returns (cue, cue_bindings) where:
        cue          : [D] complex, the input to Phase 5
        cue_bindings : [n_roles, D] complex, the noisy fillers per position

    Anti-homunculus note: this generator is a *measurement target* for
    Phase 5's headline test. It is NOT a mechanism inside the
    architecture. Phase 5's runtime receives `cue` and (optionally)
    `cue_bindings` for the role-prior comparison condition; the
    architecture does not know how the cue was synthesized.
    """
    if content_distortion < 0.0 or content_distortion > 1.0:
        raise ValueError(
            f"content_distortion must be in [0, 1], got {content_distortion}"
        )
    if binding_noise_std < 0.0:
        raise ValueError(
            f"binding_noise_std must be non-negative, got {binding_noise_std}"
        )
    device = role_target_schema.device
    d = role_target_schema.shape[-1]
    fillers = []
    for pos in positions:
        f = substrate.unbind(role_target_schema, pos.to(device))
        if binding_noise_std > 0.0:
            noise = binding_noise_std * torch.randn(
                d, dtype=role_target_schema.dtype, device=device,
            )
            f = f + noise
        fillers.append(f)
    cue_bindings = torch.stack(fillers, dim=0)

    # Structural component: re-bundle the (noisy) fillers at the same positions.
    structural_terms = [
        substrate.bind(fillers[r], positions[r].to(device))
        for r in range(len(positions))
    ]
    structural = substrate.normalize(
        torch.stack(structural_terms, dim=0).sum(dim=0)
    )

    if content_distractor is None or content_distortion == 0.0:
        cue = structural
    else:
        cd = content_distractor.to(device)
        mixed = (1.0 - content_distortion) * structural + content_distortion * cd
        cue = substrate.normalize(mixed)

    return cue, cue_bindings


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
    score_bias: Optional[torch.Tensor] = None,
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
      energy_unbiased_initial                      (raw scoring energy at cue)
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

    # Step-3 retrieval bias (from ConsolidationState.retrieval_weight_bias()):
    # subtracted from β·scores per the trajectory.py:181 sign convention.
    # When None, dynamics + telemetry are bit-identical to pre-step-3 baseline
    # (load-bearing for back-compat with reports 047–053).
    bias_dev = score_bias.to(device) if score_bias is not None else None

    # Per-formulation precomputation.
    if formulation == "per_pattern":
        prior_bias = gamma * substrate.similarity_matrix(prior_dev, patterns)  # [N]
    else:
        prior_bias = None  # global pull adds to update vector, not logits

    # Initial unbiased entropy (diagnostic). Uses RAW β·scores so the
    # initial-vs-final entropy comparison stays comparable across runs and
    # γ values; report 053-style telemetry preserved.
    init_scores = substrate.similarity_matrix(state, patterns)
    init_weights_unbiased = torch.softmax(beta * init_scores, dim=0)
    score_entropy_initial = _softmax_entropy(init_weights_unbiased)
    energy_unbiased_initial = float(
        (-torch.logsumexp(beta * init_scores, dim=0) / beta).detach().cpu()
    )

    prev_biased_energy: Optional[torch.Tensor] = None
    final_state = state
    frozen = torch.zeros((), dtype=torch.bool, device=device)
    biased_energy_tensors: List[torch.Tensor] = []
    for _ in range(max_iter):
        scores = substrate.similarity_matrix(state, patterns)
        # Compute logits (used for weighting) and biased energy (Lyapunov).
        if formulation == "per_pattern":
            biased_logits = beta * scores + prior_bias
            if bias_dev is not None:
                biased_logits = biased_logits - bias_dev
            weights = torch.softmax(biased_logits, dim=0)
            update_vec = (patterns * weights[:, None]).sum(dim=0)
            # Biased energy under per-pattern: -logsumexp(biased_logits)/β.
            # When bias_dev is None this reduces to -logsumexp(β·scores + prior_bias)/β.
            biased_energy = -torch.logsumexp(biased_logits, dim=0) / beta
        else:  # global_pull
            unbiased_logits = beta * scores
            if bias_dev is not None:
                unbiased_logits = unbiased_logits - bias_dev
            weights = torch.softmax(unbiased_logits, dim=0)
            update_vec = (patterns * weights[:, None]).sum(dim=0) + gamma * prior_dev
            # Biased energy under global pull: -logsumexp(logits)/β - γ·Re(<q, prior>)
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
    # Parallel step-3-weighted final energy. Equals energy_unbiased_final
    # when score_bias is None (back-compat with reports 047–053). When
    # provided, this is the load-bearing headline landscape per Phase 5
    # design (report 046).
    if bias_dev is not None:
        energy_unbiased_step3_final = float(
            (-torch.logsumexp(beta * final_scores - bias_dev, dim=0) / beta).detach().cpu()
        )
    else:
        energy_unbiased_step3_final = energy_unbiased_final
    if formulation == "per_pattern":
        biased_final_logits = beta * final_scores + prior_bias
        if bias_dev is not None:
            biased_final_logits = biased_final_logits - bias_dev
        energy_biased_final = float(
            (-torch.logsumexp(biased_final_logits, dim=0) / beta).detach().cpu()
        )
    else:
        q_prior_inner_final = float((state.conj() * prior_dev).sum().real.detach().cpu())
        energy_biased_final = energy_unbiased_final - gamma * q_prior_inner_final

    return state, {
        "score_entropy_initial": score_entropy_initial,
        "score_entropy_final": score_entropy_final,
        "converged": converged,
        "iterations": len(biased_energies),
        "energy_unbiased_initial": energy_unbiased_initial,
        "energy_unbiased_final": energy_unbiased_final,
        "energy_unbiased_step3_final": energy_unbiased_step3_final,
        "energy_biased_final": energy_biased_final,
        "on_substrate_alignment": on_substrate_alignment,
        "formulation": formulation,
    }


# =============================================================================
# Per-branch diagnostics (logged, NOT used for selection)
# =============================================================================

def _unbiased_energy(
    memory: TorchHopfieldMemory,
    state: torch.Tensor,
    beta: float,
    score_bias: Optional[torch.Tensor] = None,
) -> float:
    """E(q) = -logsumexp(β · sim(X, q) − score_bias) / β as a Python float.

    With ``score_bias=None`` (default), this is the raw unbiased energy used
    by reports 047–053 (back-compat). With ``score_bias`` provided, this is
    the step-3-weighted energy that the Phase 5 design spec (per report 046)
    defines as the load-bearing landscape — low-|E_i| atoms contribute
    infinitesimally via ``softplus((ε − |E_i|)/τ)`` subtracted from
    ``β · scores`` (sign convention per trajectory.py:181).
    """
    patterns = memory._pattern_matrix()
    scores = memory.substrate.similarity_matrix(state.to(memory.substrate.device), patterns)
    logits = beta * scores
    if score_bias is not None:
        logits = logits - score_bias.to(logits.device)
    return float((-torch.logsumexp(logits, dim=0) / beta).detach().cpu())


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
    score_bias: Optional[torch.Tensor] = None,
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
    if (
        settling_telemetry is not None
        and "energy_unbiased_final" in settling_telemetry
    ):
        branch.energy_unbiased = float(settling_telemetry["energy_unbiased_final"])
    else:
        branch.energy_unbiased = _unbiased_energy(memory, q_star, beta)
    # Step-3-weighted energy: equals energy_unbiased when score_bias is None.
    # Prefer settling_telemetry's value (computed during the dynamics) over
    # a fresh recomputation when available, to keep the dynamics-and-readout
    # landscapes consistent within a single branch.
    if (
        settling_telemetry is not None
        and "energy_unbiased_step3_final" in settling_telemetry
    ):
        branch.energy_unbiased_step3 = float(settling_telemetry["energy_unbiased_step3_final"])
    else:
        branch.energy_unbiased_step3 = _unbiased_energy(
            memory, q_star, beta, score_bias=score_bias
        )
    if settling_telemetry is not None and "energy_biased_final" in settling_telemetry:
        branch.energy_biased = float(settling_telemetry["energy_biased_final"])
    else:
        q_prior_inner = float((q_star.conj() * prior).sum().real.detach().cpu())
        branch.energy_biased = branch.energy_unbiased - gamma * q_prior_inner
    if (
        settling_telemetry is not None
        and "energy_unbiased_initial" in settling_telemetry
    ):
        energy_unbiased_initial = float(settling_telemetry["energy_unbiased_initial"])
    else:
        energy_unbiased_initial = _unbiased_energy(memory, q_init, beta)
    branch.energy_drop = energy_unbiased_initial - branch.energy_unbiased

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
    score_bias: Optional[torch.Tensor] = None,
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
        score_bias=score_bias,
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
    schema_fidelities: Optional[torch.Tensor] = None,
    p: float = 1.0,
    q: float = 1.0,
    include_surprise_branch: bool = True,
    max_settling_iter: int = 12,
    boltzmann_rng: Optional[torch.Generator] = None,
    random_prior_rng: Optional[torch.Generator] = None,
    score_bias: Optional[torch.Tensor] = None,
    run_combiners: bool = True,
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

    ``run_combiners=False`` is a measurement-only fast path for K=1
    drill-down sweeps that consume per-branch energies and basin readouts
    but do not interpret the combined retrieval state.

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
        schema_fidelities=schema_fidelities, p=p, q=q,
        substrate=memory.substrate,
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
            score_bias=score_bias,
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
            score_bias=score_bias,
        )
        branches.append(b)

    if surprise_pick is not None:
        _, prior_vec = surprise_pick
        q_settled, telem = settle_branch_with_prior(
            memory=memory, cue=cue, prior=prior_vec,
            beta=beta, gamma=gamma, max_iter=max_settling_iter,
            formulation=formulation,
            score_bias=score_bias,
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
            score_bias=score_bias,
        )
        branches.append(b)

    result.branches = branches

    # --- Step 4: pairwise final-state divergence -------------------------
    compute_pairwise_final_state_divergence(branches)

    # --- Step 5: all three combination rules ----------------------------
    if branches and run_combiners:
        q_bundle, weights, _conv = combine_bundle_resettle(
            branches=branches, memory=memory, beta=beta,
            temperature=temperature, max_iter=max_settling_iter,
            score_bias=score_bias,
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
    elif branches:
        weights = _branch_softmax_weights(branches, temperature)
        result.softmax_weights = weights.detach().cpu().tolist()
        result.softmax_entropy = _softmax_entropy(weights)

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


def _build_encoded_window_substrate(
    *,
    n_atoms: int,
    dim: int,
    window_size: int,
    vocab_size: int,
    device: str,
    seed: int,
):
    """Build a synthetic substrate whose patterns are encoded windows.

    Mirrors how a real Phase 4 substrate is constructed: a codebook of
    token vectors, position vectors, and a Hopfield memory holding
    bundle(filler_i ⊛ pos_i for i in window). Used by --mode headline
    when no `--substrate-snapshot` is provided so the role-prior
    comparison has a working synthetic baseline.

    Returns (memory, consolidation, patterns, positions, codebook,
    pattern_token_ids).
    """
    from energy_memory.phase4.consolidation import (
        ConsolidationConfig, ConsolidationState,
    )
    from energy_memory.phase2.encoding import build_position_vectors, encode_window
    torch.manual_seed(seed)
    substrate = TorchFHRR(dim=dim, device=device)
    mem = TorchHopfieldMemory(substrate)
    cons = ConsolidationState(ConsolidationConfig(m=4, alpha=0.25), device=device)
    positions = build_position_vectors(substrate, count=window_size)
    codebook = substrate.normalize(
        torch.randn(vocab_size, dim, dtype=torch.complex64, device=device)
    )
    patterns: List[torch.Tensor] = []
    pattern_token_ids: List[Tuple[int, ...]] = []
    rng = torch.Generator().manual_seed(seed + 1)
    for i in range(n_atoms):
        ids = tuple(
            int(torch.randint(0, vocab_size, (1,), generator=rng).item())
            for _ in range(window_size)
        )
        p = encode_window(substrate, positions, codebook, list(ids))
        mem.store(p, label=i)
        cons.add_pattern(novelty_strength=1.0 + 0.2 * (n_atoms - i))
        patterns.append(p)
        pattern_token_ids.append(ids)
    return mem, cons, patterns, positions, codebook, pattern_token_ids


def _build_role_binding_cues(
    *,
    substrate: TorchFHRR,
    positions: Sequence[torch.Tensor],
    patterns: Sequence[torch.Tensor],
    n_cues: int,
    binding_noise_std: float = 0.05,
    content_distortion: float = 0.6,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """For each cue, pick a role-target schema and a content-distractor
    schema (different from role-target), then synthesize the cue per
    `generate_role_binding_cue`.

    Returns a list of dicts with keys:
        cue, cue_bindings, role_target_idx, content_distractor_idx
    so the headline driver can match its results to ground truth.
    """
    if len(patterns) < 2:
        raise ValueError("need at least 2 patterns to pick role/content pair")
    rng = torch.Generator().manual_seed(seed)
    n_patterns = len(patterns)
    out: List[Dict[str, Any]] = []
    for _ in range(n_cues):
        idx_pair = torch.randperm(n_patterns, generator=rng)[:2].tolist()
        role_idx, content_idx = idx_pair[0], idx_pair[1]
        cue, cue_bindings = generate_role_binding_cue(
            substrate=substrate, positions=positions,
            role_target_schema=patterns[role_idx],
            content_distractor=patterns[content_idx],
            binding_noise_std=binding_noise_std,
            content_distortion=content_distortion,
        )
        out.append({
            "cue": cue,
            "cue_bindings": cue_bindings,
            "role_target_idx": role_idx,
            "content_distractor_idx": content_idx,
        })
    return out


def _load_substrate_from_snapshot(*, path: str, device: str):
    """Load (memory, consolidation, patterns, positions, info) from a snapshot.

    Constructs a TorchFHRR substrate matching the snapshot's dim. Positions
    (when saved) are returned as a [W, D] tensor; None when the snapshot
    pre-dates positions support.
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
    positions = info.get("positions")  # [W, D] tensor or None
    return mem, cons, patterns, positions, info


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
    # Step-3 retrieval-weight bias from consolidation. When coverage_lambda=0
    # this is None and the run is bit-identical to pre-walk-back behavior
    # (back-compat with reports 047–053). When coverage_lambda > 0 (the
    # Phase 5 design-spec configuration per report 046) this is the
    # softplus((ε−|E_i|)/τ) per-atom bias, subtracted from β·scores in
    # settling logits and final-energy telemetry.
    step3_bias = (
        cons.retrieval_weight_bias()
        if cons.config.coverage_lambda > 0.0
        else None
    )
    per_cue_energy_unbiased_min = []
    per_cue_energy_step3_min = []
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
            score_bias=step3_bias,
        )
        if not result.branches:
            continue
        e_min = min(b.energy_unbiased for b in result.branches)
        e_step3_min = min(b.energy_unbiased_step3 for b in result.branches)
        align = max(
            float(mem.substrate.similarity(b.q_settled, p))
            for b in result.branches for p in patterns
        )
        per_cue_energy_unbiased_min.append(e_min)
        per_cue_energy_step3_min.append(e_step3_min)
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
        "step3_bias_active": step3_bias is not None,
        "n_cues": len(per_cue_energy_unbiased_min),
        "mean_energy_unbiased_min": sum(per_cue_energy_unbiased_min) / n,
        "mean_energy_step3_min": sum(per_cue_energy_step3_min) / n,
        "mean_branch_count": sum(per_cue_branch_count) / n,
        "split_eligibility_rate": sum(per_cue_split_eligible) / n,
        "mean_on_substrate_alignment": sum(per_cue_on_substrate_alignment) / n,
        "mean_branch_softmax_entropy": sum(per_cue_softmax_entropy) / n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=str, default="smoke",
                        choices=["smoke", "decision5_spike", "headline"],
                        help="smoke: synthetic raw substrate, SMOKE_CONDITIONS. "
                        "decision5_spike: per_pattern × global_pull × γ grid. "
                        "headline: role-binding cues, paired role-vs-content "
                        "ΔE per cue, with random-schema + γ=0 controls.")
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
    parser.add_argument("--window-size", type=int, default=3,
                        help="positions count for headline mode synthetic substrate")
    parser.add_argument("--vocab-size", type=int, default=30,
                        help="codebook size for headline mode synthetic substrate")
    parser.add_argument("--binding-noise-std", type=float, default=0.05,
                        help="noise added to extracted role fillers (headline mode)")
    parser.add_argument("--content-distortion", type=float, default=0.6,
                        help="weight of content_distractor in headline cues, "
                        "in [0, 1]; higher = more content/role disagreement")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="prior weight for headline mode")
    parser.add_argument("--k-main", type=int, default=4,
                        help="K_main for the main (non-K=1-control) headline "
                        "runs. Used for the role_K{k}, content_K{k}, "
                        "random_K{k}, role_K{k}_g0, content_K{k}_g0 "
                        "conditions. K=1 control is always run separately.")
    parser.add_argument("--formulation", type=str, default="per_pattern",
                        choices=list(FORMULATIONS),
                        help="prior formulation for headline mode")
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
    positions: Optional[Sequence[torch.Tensor]] = None
    codebook: Optional[torch.Tensor] = None
    if args.substrate_snapshot is not None:
        mem, cons, patterns, positions, snapshot_info = _load_substrate_from_snapshot(
            path=args.substrate_snapshot, device=args.device,
        )
        effective_dim = mem.substrate.dim
        print(
            f"[load] substrate snapshot from {args.substrate_snapshot}: "
            f"n_atoms={len(patterns)}, dim={effective_dim}, "
            f"positions={'yes' if positions is not None else 'no'}, "
            f"label={snapshot_info.get('label')!r}",
            flush=True,
        )
    elif args.mode == "headline":
        mem, cons, patterns, positions_list, codebook, _token_ids = (
            _build_encoded_window_substrate(
                n_atoms=args.n_atoms, dim=args.dim,
                window_size=args.window_size, vocab_size=args.vocab_size,
                device=args.device, seed=args.seed,
            )
        )
        positions = torch.stack(positions_list, dim=0)
        effective_dim = args.dim
    else:
        mem, cons, patterns = _build_synthetic_substrate(
            n_atoms=args.n_atoms, dim=args.dim, device=args.device, seed=args.seed,
        )
        effective_dim = args.dim

    if args.mode == "headline" and positions is None:
        raise ValueError(
            "headline mode requires positions; substrate snapshot did not "
            "include them (was it saved by an older exp 19 without "
            "--snapshot-steps positions support?). Re-save the snapshot, "
            "or omit --substrate-snapshot to use the synthetic encoded-window "
            "substrate."
        )

    schema_store, atom_idx = get_schema_store(
        consolidation=cons, patterns=mem._pattern_matrix(),
        selection_rule="top_k_by_effective_strength",
        k=min(args.k, len(patterns)),
    )

    # Build cues based on mode.
    cues: List[torch.Tensor] = []
    cue_specs: List[Dict[str, Any]] = []  # only populated in headline mode
    if args.mode == "headline":
        cue_specs = _build_role_binding_cues(
            substrate=mem.substrate, positions=positions, patterns=patterns,
            n_cues=args.n_cues,
            binding_noise_std=args.binding_noise_std,
            content_distortion=args.content_distortion,
            seed=args.seed + 100,
        )
        cues = [c["cue"] for c in cue_specs]
    else:
        # Perturbed copies of stored patterns (so retrieval has work to do).
        for i in range(args.n_cues):
            base = patterns[i % len(patterns)]
            noise = torch.randn(
                effective_dim, dtype=torch.complex64, device=args.device,
            )
            cues.append(mem.substrate.normalize(base + 0.15 * noise))

    boltzmann_rng = torch.Generator().manual_seed(args.seed + 10)
    random_prior_rng = torch.Generator().manual_seed(args.seed + 20)

    if args.mode == "smoke":
        runs = [
            (name, prior_type, gamma, k_main, "per_pattern")
            for (name, prior_type, gamma, k_main) in SMOKE_CONDITIONS
        ]
    elif args.mode == "decision5_spike":
        # formulation × γ grid (per_pattern + global_pull)
        runs = []
        for formulation in FORMULATIONS:
            for gamma in (0.25, 0.5, 1.0):
                name = f"content_K4_g{gamma}_{formulation}"
                runs.append((name, "content", gamma, 4, formulation))
    else:  # headline
        # The headline run: role vs content vs random; controls γ=0 and K=1.
        # Plus β (path-3) conditions: fid_K1_q0 (content-only baseline) and
        # fid_K1_q1 (β recommended). β operates over the FULL pattern matrix
        # (not the top-k pre-filter); the headline loop substitutes the full
        # matrix at the call site when the condition name starts with "fid_".
        # See notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md.
        km = args.k_main
        runs = [
            (f"role_K{km}", "role", args.gamma, km, args.formulation),
            (f"content_K{km}", "content", args.gamma, km, args.formulation),
            (f"random_K{km}", "random", args.gamma, km, args.formulation),
            (f"role_K{km}_g0", "role", 0.0, km, args.formulation),
            (f"content_K{km}_g0", "content", 0.0, km, args.formulation),
            # β conditions (path 3, the report-049 selector-layer fix).
            # K=1 because β produces ONE continuous-weighted prior; γ is
            # the same as the role/content conditions for a fair comparison.
            ("fid_K1_q0", "fidelity_weighted", args.gamma, 1, args.formulation),
            ("fid_K1_q1", "fidelity_weighted", args.gamma, 1, args.formulation),
        ]
        # K=1 control runs only if k_main != 1 (otherwise duplicates main).
        if km != 1:
            runs.extend([
                ("role_K1", "role", args.gamma, 1, args.formulation),
                ("content_K1", "content", args.gamma, 1, args.formulation),
            ])

    results = []
    if args.mode == "headline":
        # Headline mode runs cues with role-binding metadata; record per-cue
        # energies under each condition so we can compute paired ΔE.
        schema_bindings = compute_schema_bindings(
            substrate=mem.substrate, schemas=schema_store, positions=positions,
        )

        # Step-3 retrieval-weight bias for the headline. Computed once
        # per run (snapshot's consolidation state is frozen for headline
        # mode). When coverage_lambda=0, step3_bias is None and the run
        # reproduces reports 047–053 bit-identically (back-compat). When
        # coverage_lambda > 0 (the design-spec configuration per report
        # 046), the bias is softplus((ε−|E_i|)/τ) per atom, subtracted
        # from β·scores in both settling dynamics and final-energy
        # telemetry. See the 2026-05-21 STATUS walk-back.
        step3_bias = (
            cons.retrieval_weight_bias()
            if cons.config.coverage_lambda > 0.0
            else None
        )
        print(
            f"[step3] coverage_lambda={cons.config.coverage_lambda}; "
            f"score_bias active={step3_bias is not None}"
            + (
                f"; bias mean={float(step3_bias.mean()):.4f}, "
                f"max={float(step3_bias.max()):.4f}, "
                f"min={float(step3_bias.min()):.4f}"
                if step3_bias is not None else ""
            )
        )

        # β prerequisites: when any fid_* condition runs, β operates over
        # the FULL pattern matrix (not the top-k pre-filter). Pre-compute
        # full-substrate bindings + fidelities once; reuse per cue.
        full_patterns = mem._pattern_matrix()
        run_needs_full_substrate = any(
            r[1] == "fidelity_weighted" for r in runs
        )
        if run_needs_full_substrate:
            from energy_memory.phase5.role_fidelity import compute_role_fidelity
            full_schema_bindings = compute_schema_bindings(
                substrate=mem.substrate, schemas=full_patterns,
                positions=positions,
            )
            full_schema_fidelities = compute_role_fidelity(full_schema_bindings)
            print(
                f"[β] full-substrate fidelities computed: N={full_patterns.shape[0]}, "
                f"mean(f)={float(full_schema_fidelities.mean()):.4f}, "
                f"std(f)={float(full_schema_fidelities.std()):.4f}, "
                f"min={float(full_schema_fidelities.min()):.4f}, "
                f"max={float(full_schema_fidelities.max()):.4f}"
            )
        else:
            full_schema_bindings = None
            full_schema_fidelities = None

        for (name, prior_type, gamma, k_main, formulation) in runs:
            # Determine q for β conditions (encoded in the condition name).
            if prior_type == "fidelity_weighted":
                q_run = 0.0 if name.endswith("_q0") else 1.0
                p_run = 1.0
                # β uses the full schema store, not the top-k filter.
                run_schemas = full_patterns
                run_schema_bindings = full_schema_bindings
                run_schema_fidelities = full_schema_fidelities
                run_atom_idx = None  # β prior is not tied to specific atom indices
            else:
                q_run = 1.0
                p_run = 1.0
                run_schemas = schema_store
                run_schema_bindings = schema_bindings
                run_schema_fidelities = None
                run_atom_idx = atom_idx

            print(f"[run] {name} prior={prior_type} γ={gamma} K={k_main} form={formulation}"
                  + (f" (β: q={q_run})" if prior_type == "fidelity_weighted" else ""))
            per_cue_e_min = []
            per_cue_e_min_unbiased = []
            per_cue_e_min_step3 = []
            per_cue_align = []
            per_cue_softmax_entropy = []
            per_cue_state_divergence = []
            per_cue_prior_alignment = []
            per_cue_prior_pairwise_dist = []
            per_cue_energy_drop = []
            per_cue_n_branches = []
            for cue_id, spec in enumerate(cue_specs):
                result = run_branched_retrieval(
                    cue=spec["cue"], cue_id=cue_id, target_id=spec["role_target_idx"],
                    memory=mem,
                    codebook=codebook if codebook is not None else mem._pattern_matrix(),
                    positions=positions,
                    decode_ids=[], decode_k=5, masked_pos=0,
                    schema_store=run_schemas, schema_atom_idx=run_atom_idx,
                    consolidation=cons, prior_type=prior_type, k_main=k_main,
                    gamma=gamma, beta=args.beta, temperature=args.temperature,
                    delta_energy=args.delta_energy, delta_state=args.delta_state,
                    delta_redundant=args.delta_redundant,
                    formulation=formulation,
                    cue_bindings=spec["cue_bindings"],
                    schema_bindings=run_schema_bindings,
                    schema_fidelities=run_schema_fidelities,
                    p=p_run, q=q_run,
                    include_surprise_branch=False,
                    boltzmann_rng=boltzmann_rng,
                    random_prior_rng=random_prior_rng,
                    score_bias=step3_bias,
                )
                if not result.branches:
                    continue
                e_min = min(b.energy_unbiased for b in result.branches)
                e_step3_min = min(b.energy_unbiased_step3 for b in result.branches)
                align = max(
                    float(mem.substrate.similarity(b.q_settled, p))
                    for b in result.branches for p in patterns
                )
                per_cue_e_min.append(e_min)
                per_cue_e_min_unbiased.append(e_min)
                per_cue_e_min_step3.append(e_step3_min)
                per_cue_align.append(align)
                nb = len(result.branches)
                per_cue_n_branches.append(nb)
                per_cue_softmax_entropy.append(float(result.softmax_entropy))
                per_cue_state_divergence.append(
                    sum(b.final_state_divergence for b in result.branches) / nb
                )
                per_cue_prior_alignment.append(
                    sum(b.prior_alignment for b in result.branches) / nb
                )
                per_cue_energy_drop.append(
                    sum(b.energy_drop for b in result.branches) / nb
                )
                # Pairwise FHRR cos-distance among the K priors used this cue.
                if nb <= 1:
                    per_cue_prior_pairwise_dist.append(0.0)
                else:
                    priors = [b.prior for b in result.branches]
                    dists = []
                    for i in range(nb):
                        for j in range(i + 1, nb):
                            c = float(_fhrr_cosine(priors[i], priors[j]))
                            dists.append(1.0 - c)
                    per_cue_prior_pairwise_dist.append(sum(dists) / len(dists))
            n = max(len(per_cue_e_min), 1)
            results.append({
                "name": name,
                "prior_type": prior_type,
                "gamma": gamma,
                "k_main": k_main,
                "formulation": formulation,
                "step3_bias_active": step3_bias is not None,
                "n_cues": len(per_cue_e_min),
                "per_cue_energy_unbiased_min": per_cue_e_min,
                "per_cue_energy_step3_min": per_cue_e_min_step3,
                "mean_energy_unbiased_min": sum(per_cue_e_min) / n,
                "mean_energy_step3_min": (
                    sum(per_cue_e_min_step3) / n if per_cue_e_min_step3 else 0.0
                ),
                "mean_on_substrate_alignment": sum(per_cue_align) / n,
                "mean_n_branches": (
                    sum(per_cue_n_branches) / n if per_cue_n_branches else 0
                ),
                "mean_branch_softmax_entropy": (
                    sum(per_cue_softmax_entropy) / n
                    if per_cue_softmax_entropy else 0.0
                ),
                "mean_branch_state_divergence": (
                    sum(per_cue_state_divergence) / n
                    if per_cue_state_divergence else 0.0
                ),
                "mean_prior_alignment": (
                    sum(per_cue_prior_alignment) / n
                    if per_cue_prior_alignment else 0.0
                ),
                "mean_prior_pairwise_distance": (
                    sum(per_cue_prior_pairwise_dist) / n
                    if per_cue_prior_pairwise_dist else 0.0
                ),
                "mean_energy_drop": (
                    sum(per_cue_energy_drop) / n
                    if per_cue_energy_drop else 0.0
                ),
            })
        # Paired ΔE = E_content - E_role (positive = role-prior found a
        # lower-energy state, i.e. structural retrieval). Compute for
        # every matched (content_*, role_*) pair found in results.
        named = {r["name"]: r for r in results}
        deltas = {}
        tag_set = []
        for n_ in named:
            if n_.startswith("content_"):
                tag = n_[len("content_"):]
                if f"role_{tag}" in named and tag not in tag_set:
                    tag_set.append(tag)
        for tag in tag_set:
            content_name = f"content_{tag}"
            role_name = f"role_{tag}"
            if content_name not in named or role_name not in named:
                continue
            c = named[content_name]["per_cue_energy_unbiased_min"]
            r = named[role_name]["per_cue_energy_unbiased_min"]
            if len(c) != len(r) or not c:
                continue
            per_cue_delta = [ci - ri for ci, ri in zip(c, r)]
            n_pos = sum(1 for d in per_cue_delta if d > 0)
            mean_d = sum(per_cue_delta) / len(per_cue_delta)
            # Step-3-weighted paired ΔE. When coverage_lambda=0 this equals
            # the raw ΔE (bit-identical back-compat with reports 047–053).
            # When > 0, this is the headline landscape per Phase 5 design.
            c_step3 = named[content_name].get("per_cue_energy_step3_min", c)
            r_step3 = named[role_name].get("per_cue_energy_step3_min", r)
            per_cue_delta_step3 = [ci - ri for ci, ri in zip(c_step3, r_step3)]
            n_pos_step3 = sum(1 for d in per_cue_delta_step3 if d > 0)
            mean_d_step3 = (
                sum(per_cue_delta_step3) / len(per_cue_delta_step3)
                if per_cue_delta_step3 else 0.0
            )
            deltas[tag] = {
                "n_pairs": len(per_cue_delta),
                "mean_delta_e_content_minus_role": mean_d,
                "fraction_positive": n_pos / len(per_cue_delta),
                "per_cue_delta": per_cue_delta,
                "mean_delta_e_step3_content_minus_role": mean_d_step3,
                "fraction_positive_step3": (
                    n_pos_step3 / len(per_cue_delta_step3)
                    if per_cue_delta_step3 else 0.0
                ),
                "per_cue_delta_step3": per_cue_delta_step3,
            }

        # β headline: ΔE = E_unbiased(fid_K1_q0) - E_unbiased(fid_K1_q1),
        # per-cue. Positive ΔE means q=1 (fidelity-weighted) finds a
        # lower-energy state than q=0 (content-only baseline). See
        # notes/notes/2026-05-20-cue-regime-role-prior-dynamic-form.md
        # §"Pre-committed falsification criteria" — criterion #2 here.
        if "fid_K1_q0" in named and "fid_K1_q1" in named:
            q0_e = named["fid_K1_q0"]["per_cue_energy_unbiased_min"]
            q1_e = named["fid_K1_q1"]["per_cue_energy_unbiased_min"]
            if len(q0_e) == len(q1_e) and q0_e:
                per_cue_beta_delta = [a - b for a, b in zip(q0_e, q1_e)]
                n_pos_beta = sum(1 for d in per_cue_beta_delta if d > 0)
                mean_beta = sum(per_cue_beta_delta) / len(per_cue_beta_delta)
                deltas["beta_q0_minus_q1"] = {
                    "n_pairs": len(per_cue_beta_delta),
                    "mean_delta_e_q0_minus_q1": mean_beta,
                    "fraction_positive": n_pos_beta / len(per_cue_beta_delta),
                    "per_cue_delta": per_cue_beta_delta,
                }
    else:
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
        deltas = None

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
        "snapshot_info": (
            {k: v for k, v in (snapshot_info or {}).items() if k != "positions"}
            if snapshot_info is not None else None
        ),
        "conditions": results,
    }
    if args.mode == "headline":
        payload["headline_deltas"] = deltas
        payload["binding_noise_std"] = args.binding_noise_std
        payload["content_distortion"] = args.content_distortion
        payload["formulation"] = args.formulation
        payload["gamma"] = args.gamma
        payload["k_main"] = args.k_main
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[done] wrote {out_path}")
    return payload


if __name__ == "__main__":
    main()
