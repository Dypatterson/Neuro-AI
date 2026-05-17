"""Phase 5 — HAM × Energy-Guided Structural Branching.

Headline experiment for the Phase 5 unified design
(see notes/emergent-codebook/phase-5-unified-design.md).

Status: **SKELETON / OUTLINE**. The schema-source plugin
(get_schema_store(...)) is a TODO that resolves once decision #1 in the
design note is settled (currently blocked on the freq-weighted-α
experiment result).

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
# Schema-source plugin — TBD pending decision #1
# =============================================================================

def get_schema_store(
    *,
    consolidation: ConsolidationState,
    codebook: torch.Tensor,
    source: str = "filtered_slow_store",
    top_n: int = 500,
) -> torch.Tensor:
    """Return a [N_schemas, D] FHRR tensor of schema priors.

    PLUGGABLE. Implementation depends on decision #1 in the Phase 5 design.
    Candidate sources:

        'filtered_slow_store'  — top-N patterns by u_m magnitude. PREFERRED
                                  once freq-α experiment confirms the slow
                                  store is meaningfully filtered.
        'co_occurrence'        — cluster patterns by co-replay; one
                                  schema per cluster.
        'ham_layer2'           — use exp 21 HAM layer-2 store directly.
        'hybrid'               — slow-store ∪ ham_layer2 with
                                  deduplication.

    Currently raises NotImplementedError; resolved post-freq-α.
    """
    raise NotImplementedError(
        f"Schema source '{source}' not yet wired. "
        "Resolution blocks on the freq-α experiment result; "
        "see notes/emergent-codebook/phase-5-unified-design.md §'Open decisions'."
    )


# =============================================================================
# Branch seeding
# =============================================================================

def select_schema_priors(
    *,
    cue: torch.Tensor,
    schema_store: torch.Tensor,
    k_main: int,
    delta_redundant: float = 0.95,
    prior_type: str = "content",
    cue_bindings: Optional[torch.Tensor] = None,
) -> List[Tuple[int, torch.Tensor]]:
    """Pick K_main schemas with diversity-filter.

    prior_type='content'    : rank schemas by FHRR cosine to cue.
    prior_type='role'       : rank schemas by binding-similarity to cue_bindings.
                              Requires cue_bindings to be the unbound role-fillers.
    prior_type='random'     : random sample of K_main schemas (control).

    Diversity filter: greedy walk down ranked schemas, skip any candidate
    whose cosine similarity to a previously-picked schema exceeds
    delta_redundant.
    """
    raise NotImplementedError("select_schema_priors: not yet implemented")


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

def settle_branch_with_prior(
    *,
    memory: TorchHopfieldMemory,
    cue: torch.Tensor,
    prior: torch.Tensor,
    beta: float,
    gamma: float,
    max_iter: int = 12,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run Hopfield retrieval with the prior added as a per-stored-pattern bias.

    Score is `β · Re(⟨X, q⟩) + γ · Re(⟨X, prior⟩)`, where the prior term
    biases every settling step toward stored patterns aligned with the
    prior. At gamma=0 this reduces to standard retrieval.

    Returns (settled_state, telemetry) where telemetry includes
    score_entropy_initial, score_entropy_final, converged.
    """
    raise NotImplementedError(
        "settle_branch_with_prior: wire to TorchHopfieldMemory.retrieve(...) "
        "with a score_bias argument equal to γ · Re(⟨X, prior⟩). "
        "(Existing retrieve() already supports score_bias for the Saighi "
        "A_k mechanism; reuse that hook.)"
    )


# =============================================================================
# Per-branch diagnostics (logged, NOT used for selection)
# =============================================================================

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
) -> None:
    """Fill out the BranchState diagnostic fields in-place."""
    # energy_unbiased: -logsumexp(beta * <X, q_settled>)
    # energy_biased  : energy_unbiased - gamma * Re(<q_settled, prior>)
    # energy_drop    : energy_unbiased(q_initial) - energy_unbiased(q_settled)
    # prior_alignment: cos(q_settled, prior)
    # score_entropy_*: H(softmax(beta * <X, q>)) at initial vs final state
    # entropy_collapse = initial - final
    # recall_support : decode q_settled at masked_pos; target_id in top-decode?
    # cap_coverage_t05: top decode score >= 0.5 AND target_id in top-K
    # meta_stable     : top decode score < 0.95
    # structural_match: unbind position vectors from q_settled, compare to cue_bindings
    # converged       : retrieve() reached convergence below max_iter
    raise NotImplementedError("compute_branch_diagnostics: not yet implemented")


def compute_pairwise_final_state_divergence(branches: List[BranchState]) -> None:
    """Fill BranchState.final_state_divergence for each branch (mean
    cosine-distance to every other branch's q_settled)."""
    raise NotImplementedError


# =============================================================================
# Branch combination — three rules, run all three for comparison
# =============================================================================

def combine_bundle_resettle(
    *,
    branches: List[BranchState],
    memory: TorchHopfieldMemory,
    beta: float,
    temperature: float = 1.0,
    max_iter: int = 12,
) -> Tuple[torch.Tensor, List[float], bool]:
    """PREFERRED combination rule (anti-homunculus-clean).

    q_bundle = Σ_k w_k * q_k_settled,  w_k = softmax(-E_k_unbiased / τ)
    q_final  = HopfieldRetrieve(q_bundle, beta=beta, max_iter=max_iter,
                                score_bias=None)   # UNBIASED re-settle

    Returns (q_final, softmax_weights, converged).

    Anti-homunculus reading: the bundle is an algebraic sum; the re-settle
    is a standard gradient descent on the marginal energy. The whole
    pipeline is one continuous dynamic, not an if-X-then-Y rule.
    """
    raise NotImplementedError


def combine_greedy_argmin(branches: List[BranchState]) -> Tuple[torch.Tensor, int]:
    """BASELINE combination rule.

    q_final = q_k_settled for k = argmin_k E_k_unbiased.

    NOTE per anti-homunculus check in the design doc: this combination
    rule is only allowed as a *comparison condition*, never the
    production architecture. The argmin output controlling downstream
    code is borderline-arbitration even though the argmin computation
    itself is a measurement.
    """
    raise NotImplementedError


def combine_boltzmann_sample(
    branches: List[BranchState], temperature: float = 1.0, rng: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, int, List[float]]:
    """COMPARISON combination rule (FEP-clean).

    Sample k ~ Categorical(softmax(-E_k / τ)); return q_k_settled.
    """
    raise NotImplementedError


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
    codebook: torch.Tensor,
    positions: torch.Tensor,
    decode_ids: Sequence[int],
    decode_k: int,
    masked_pos: int,
    schema_store: torch.Tensor,
    consolidation: ConsolidationState,
    prior_type: str,            # 'content' | 'role' | 'random' (HEADLINE COMPARISON)
    k_main: int,
    gamma: float,
    beta: float,
    temperature: float,
    delta_energy: float,
    delta_state: float,
    delta_redundant: float,
    cue_bindings: Optional[torch.Tensor] = None,
) -> BranchedRetrievalResult:
    """One full branched retrieval over a single cue.

    Steps:
      1. Pick K_main schemas (via prior_type) + 1 surprise branch.
      2. Settle each branch with γ-biased energy.
      3. Score (unbiased), log diagnostics.
      4. Pairwise final-state divergence.
      5. Run all three combination rules (bundle, greedy, Boltzmann).
      6. Atom-split signal.
    """
    raise NotImplementedError(
        "Full driver depends on the schema-source plugin and on wiring "
        "settle_branch_with_prior() to TorchHopfieldMemory.retrieve(score_bias=...). "
        "Both unblocked once decision #1 (schema source) lands."
    )


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--n-cues", type=int, default=300)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--k-main", type=int, default=4)
    parser.add_argument("--delta-energy", type=float, default=0.1)
    parser.add_argument("--delta-state", type=float, default=0.3)
    parser.add_argument("--delta-redundant", type=float, default=0.95)
    parser.add_argument("--schema-source", type=str, default="filtered_slow_store",
                        choices=["filtered_slow_store", "co_occurrence", "ham_layer2", "hybrid"])
    parser.add_argument("--codebook-path", type=str,
                        default="reports/phase3c_reconstruction/phase3c_codebook_reconstruction.pt")
    parser.add_argument("--output-dir", type=str, default="reports/phase5_branching_smoke")
    # Restrict to a single condition for debugging; default = all
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="condition names to run (default: all in CONDITIONS table)")
    args = parser.parse_args()

    raise NotImplementedError(
        "Phase 5 branching experiment driver awaiting implementation. "
        "Open dependencies:\n"
        "  1. Schema source (decision #1) — blocks get_schema_store(...)\n"
        "  2. settle_branch_with_prior(...) wired to retrieve(score_bias=...)\n"
        "  3. compute_branch_diagnostics(...) fills the per-branch diagnostic panel\n"
        "  4. role-prior selection — defines binding-similarity for headline comparison\n"
        "Design spec: notes/emergent-codebook/phase-5-unified-design.md"
    )


if __name__ == "__main__":
    main()
