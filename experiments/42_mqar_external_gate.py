"""MQAR — external architecture-gate discriminator.

This experiment is NOT a Phase 5 graduation experiment. The Phase 5
headline per
[phase-5-unified-design.md:280-299](../notes/emergent-codebook/phase-5-unified-design.md)
remains `ΔE = E_content - E_role` with magnitude floor 5.5e-3 and CI > 0.
This script does not measure ΔE.

What this script measures
-------------------------
MQAR-style key-only associative recall on the existing FHRR + Modern
Hopfield substrate. The question is whether the substrate has any
key-only basins at all — the diagnostic question raised by the four
Phase 5 retrieval-mechanism nulls (D1/D3/E1/M1) and confirmed at smoke
scale by
[brainstorm-workspace/2026-05-24-unconsidered-paths/smoke_c_mqar.py](../brainstorm-workspace/2026-05-24-unconsidered-paths/smoke_c_mqar.py)
(0% top-1 at D=4096, N=128, n_queries=32).

Headline (this experiment, NOT Phase 5)
---------------------------------------
Hopfield key-only top-1 recall at D=4096 as a function of N, with
Wilson 95% CI across (seeds × queries). The point this is a
discriminator: if the curve stays at chance / zero, the substrate
diagnosis stands and Phase 5 stuck-state is a substrate-shape problem,
not a retrieval-tuning problem. If the curve is nontrivial, the smoke
scale was misleading.

Positive control on the same test set
-------------------------------------
HRR-bundle top-1 recall: store Σ bind(k_i, v_i), recover via
unbind(M, k_j). This proves the task is solvable on this exact
substrate at this exact scale. If HRR-bundle works and Hopfield
key-only does not, the gap is in retrieval geometry not in
representation capacity.

Negative control (sanity baseline)
----------------------------------
Hopfield perfect-cue top-1 recall: cue with the bound pair itself.
This is memorization, not associative recall. Confirms storage works.

Strategy registry
-----------------
The script is organised as a strategy registry so future entries
(GHRR substrate, Residue-HDC, bundle-first) drop in as new keys
without rewriting the harness. See `_STRATEGIES`.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import torch

from energy_memory.memory.torch_hopfield import TorchHopfieldMemory
from energy_memory.substrate.torch_fhrr import TorchFHRR


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    D: int
    N: int
    seed: int
    n_queries: int
    n_correct: int
    top1: float
    notes: str = ""
    # Diagnostics added 2026-05-24 after codex review caught that the
    # original ghrr_matrix_key_only cell was bundle-style algebraic
    # decode (top_index_hits ≈ 0, entropy ≈ 1) not basin retrieval.
    # These fields distinguish "Hopfield found the right basin" from
    # "post-settle algebraic step recovered the value from a uniform
    # mixture." Set to None for non-Hopfield strategies.
    top_index_hits: int = -1     # n queries where MHN top_index == qi
    mean_weight_entropy: float = float("nan")  # normalized [0,1]
    mean_score_margin: float = float("nan")    # top - 2nd score


@dataclass(frozen=True)
class WilsonCI:
    mean: float
    lo: float
    hi: float
    n: int


def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> WilsonCI:
    if n_total == 0:
        return WilsonCI(0.0, 0.0, 0.0, 0)
    p = n_success / n_total
    denom = 1.0 + z * z / n_total
    center = (p + z * z / (2 * n_total)) / denom
    half = (z * math.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total * n_total))) / denom
    return WilsonCI(mean=p, lo=max(0.0, center - half), hi=min(1.0, center + half), n=n_total)


def _query_indices(N: int, n_queries: int, seed: int) -> List[int]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randint(0, N, (n_queries,), generator=g).tolist()


def _hopfield_diagnostics(
    result, qi: int, N: int
) -> Tuple[int, float, float]:
    """Return (top_index_hit_int, normalized_entropy, top_minus_second_score)."""
    hit = 1 if result.top_index == qi else 0
    weights = torch.tensor(result.weights)
    eps = 1e-12
    w_safe = weights.clamp(min=eps)
    if N > 1:
        ent = float(-(w_safe * w_safe.log()).sum() / math.log(N))
    else:
        ent = 0.0
    scores = torch.tensor(result.scores)
    sorted_, _ = torch.sort(scores, descending=True)
    margin = float(sorted_[0] - sorted_[1]) if scores.numel() >= 2 else 0.0
    return hit, ent, margin


def _hrr_bundle(fhrr: TorchFHRR, N: int, n_queries: int, *, seed: int) -> StrategyResult:
    keys = fhrr.random_vectors(N)
    values = fhrr.random_vectors(N)
    M = fhrr.normalize((keys * values).sum(dim=0))
    correct = 0
    for qi in _query_indices(N, n_queries, seed):
        rec = fhrr.unbind(M, keys[qi])
        pred = int(fhrr.similarity_matrix(rec, values).argmax().item())
        if pred == qi:
            correct += 1
    return StrategyResult(
        strategy="hrr_bundle",
        D=fhrr.dim,
        N=N,
        seed=seed,
        n_queries=n_queries,
        n_correct=correct,
        top1=correct / n_queries,
    )


def _hopfield_perfect_cue(
    fhrr: TorchFHRR, N: int, n_queries: int, *, seed: int, beta: float
) -> StrategyResult:
    keys = fhrr.random_vectors(N)
    values = fhrr.random_vectors(N)
    bound = keys * values
    hop = TorchHopfieldMemory(substrate=fhrr)
    for i in range(N):
        hop.store(bound[i], label=f"pair_{i}")
    correct = 0
    top_idx_hits = 0
    ents: List[float] = []
    margins: List[float] = []
    for qi in _query_indices(N, n_queries, seed):
        result = hop.retrieve(bound[qi], beta=beta)
        hit, ent, margin = _hopfield_diagnostics(result, qi, N)
        top_idx_hits += hit
        ents.append(ent)
        margins.append(margin)
        rec_val = fhrr.unbind(result.state, keys[qi])
        pred = int(fhrr.similarity_matrix(rec_val, values).argmax().item())
        if pred == qi:
            correct += 1
    return StrategyResult(
        strategy="hopfield_perfect_cue",
        D=fhrr.dim,
        N=N,
        seed=seed,
        n_queries=n_queries,
        n_correct=correct,
        top1=correct / n_queries,
        notes=f"beta={beta}",
        top_index_hits=top_idx_hits,
        mean_weight_entropy=sum(ents) / len(ents),
        mean_score_margin=sum(margins) / len(margins),
    )


def _hopfield_key_only(
    fhrr: TorchFHRR, N: int, n_queries: int, *, seed: int, beta: float
) -> StrategyResult:
    keys = fhrr.random_vectors(N)
    values = fhrr.random_vectors(N)
    bound = keys * values
    hop = TorchHopfieldMemory(substrate=fhrr)
    for i in range(N):
        hop.store(bound[i], label=f"pair_{i}")
    correct = 0
    top_idx_hits = 0
    ents: List[float] = []
    margins: List[float] = []
    for qi in _query_indices(N, n_queries, seed):
        result = hop.retrieve(keys[qi], beta=beta)
        hit, ent, margin = _hopfield_diagnostics(result, qi, N)
        top_idx_hits += hit
        ents.append(ent)
        margins.append(margin)
        rec_val = fhrr.unbind(result.state, keys[qi])
        pred = int(fhrr.similarity_matrix(rec_val, values).argmax().item())
        if pred == qi:
            correct += 1
    return StrategyResult(
        strategy="hopfield_key_only",
        D=fhrr.dim,
        N=N,
        seed=seed,
        n_queries=n_queries,
        n_correct=correct,
        top1=correct / n_queries,
        notes=f"beta={beta}",
        top_index_hits=top_idx_hits,
        mean_weight_entropy=sum(ents) / len(ents),
        mean_score_margin=sum(margins) / len(margins),
    )


def _random_unitary_matrices(n: int, m: int, *, seed: int, device: torch.device) -> torch.Tensor:
    """Return [n, m, m] complex unitary matrices via QR of complex-normal samples."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    A_real = torch.randn(n, m, m, generator=g)
    A_imag = torch.randn(n, m, m, generator=g)
    A = torch.complex(A_real, A_imag)
    Q, _ = torch.linalg.qr(A)
    return Q.to(device)


def _ghrr_matrix_key_only(
    fhrr: TorchFHRR, N: int, n_queries: int, *, seed: int, beta: float
) -> StrategyResult:
    """GHRR matrix-binding pretending to use FHRR's Hopfield substrate.

    **WARNING — codex review 2026-05-24:** this cell is structurally
    confounded. It stores flattened bound matrices in TorchHopfieldMemory,
    which uses FHRR per-element normalization in its settling loop and
    FHRR mean-real-product similarity. With unitary matrices flattened
    to D-dim vectors, both of those primitives are wrong-shape for the
    GHRR algebra. Empirically the MHN converges to near-uniform weights
    (entropy ≈ 1.0) and the recovered top-1 is driven by the post-settle
    algebraic step `K^H @ state_mat`, NOT by Hopfield basin retrieval.
    Kept in the registry as a documented negative-control showing
    "bundle-style algebraic decode masquerading as recall." For the
    actual GHRR key-only Hopfield test see `_ghrr_matrix_key_only_native`.
    """
    D = fhrr.dim
    m = int(math.isqrt(D))
    if m * m != D:
        raise ValueError(f"GHRR matrix variant needs D = m^2; got D={D}")
    device = fhrr.device

    keys_mat = _random_unitary_matrices(N, m, seed=seed * 7919 + 1, device=device)
    values_mat = _random_unitary_matrices(N, m, seed=seed * 7919 + 2, device=device)
    bound_mat = torch.matmul(keys_mat, values_mat)
    bound_flat = bound_mat.reshape(N, D)
    values_flat = values_mat.reshape(N, D)

    hop = TorchHopfieldMemory(substrate=fhrr)
    for i in range(N):
        hop.store(bound_flat[i], label=f"pair_{i}")

    correct = 0
    top_idx_hits = 0
    ents: List[float] = []
    margins: List[float] = []
    for qi in _query_indices(N, n_queries, seed):
        cue_flat = keys_mat[qi].reshape(D)
        result = hop.retrieve(cue_flat, beta=beta)
        hit, ent, margin = _hopfield_diagnostics(result, qi, N)
        top_idx_hits += hit
        ents.append(ent)
        margins.append(margin)
        state_mat = result.state.reshape(m, m)
        rec_val_mat = torch.matmul(keys_mat[qi].conj().transpose(-1, -2), state_mat)
        rec_val_flat = rec_val_mat.reshape(D)
        sims = (values_flat.conj() * rec_val_flat[None, :]).real.mean(dim=1)
        pred = int(sims.argmax().item())
        if pred == qi:
            correct += 1
    return StrategyResult(
        strategy="ghrr_matrix_key_only",
        D=fhrr.dim,
        N=N,
        seed=seed,
        n_queries=n_queries,
        n_correct=correct,
        top1=correct / n_queries,
        notes=f"beta={beta},m={m},CONFOUNDED_see_docstring",
        top_index_hits=top_idx_hits,
        mean_weight_entropy=sum(ents) / len(ents),
        mean_score_margin=sum(margins) / len(margins),
    )


def _unitary_retract(state_mat: "torch.Tensor") -> "torch.Tensor":
    """Polar-decomposition retraction to nearest unitary matrix.

    state_mat is m×m complex. Returns U @ V^H where state_mat = U @ S @ V^H.
    The polar factor U @ V^H is the nearest unitary matrix to state_mat
    in Frobenius norm.
    """
    U, _, Vh = torch.linalg.svd(state_mat, full_matrices=False)
    return torch.matmul(U, Vh)


def _ghrr_matrix_key_only_native(
    fhrr: TorchFHRR, N: int, n_queries: int, *, seed: int, beta: float
) -> StrategyResult:
    """GHRR-native key-only Hopfield retrieval with proper algebra primitives.

    Settles on the manifold of m×m unitary matrices using:
      - Frobenius inner-product similarity: sim(A, B) = Re(tr(A^H B)) / m
      - Polar-decomposition retraction (nearest-unitary) instead of
        FHRR per-element normalization

    This is the actual test of "GHRR algebra rescues key-only Hopfield
    basin retrieval." If top_index_hits is high and entropy collapses to
    a sparse weight distribution, the rescue is real. If diagnostics
    still look uniform, GHRR-as-stated does not help and the bundle-first
    path is the only positive Report 066 finding.

    The MHN settling loop is inlined (rather than reusing
    TorchHopfieldMemory) because GHRR's similarity and normalization are
    structurally incompatible with the FHRR substrate's primitives.
    """
    D = fhrr.dim
    m = int(math.isqrt(D))
    if m * m != D:
        raise ValueError(f"GHRR matrix variant needs D = m^2; got D={D}")
    device = fhrr.device
    max_iter = 10
    tol = 1e-8

    keys_mat = _random_unitary_matrices(N, m, seed=seed * 7919 + 1, device=device)
    values_mat = _random_unitary_matrices(N, m, seed=seed * 7919 + 2, device=device)
    bound_mat = torch.matmul(keys_mat, values_mat)  # [N, m, m]

    def frobenius_scores(state_mat: "torch.Tensor") -> "torch.Tensor":
        """[N] scores: Re(tr(B_i^H @ state)) / m for each stored B_i."""
        prod = bound_mat.conj().transpose(-1, -2) @ state_mat.unsqueeze(0)
        diag = torch.diagonal(prod, dim1=-2, dim2=-1)
        return diag.sum(dim=-1).real / m

    correct = 0
    top_idx_hits = 0
    ents: List[float] = []
    margins: List[float] = []

    for qi in _query_indices(N, n_queries, seed):
        cue_mat = keys_mat[qi]
        state = cue_mat
        prev_energy = None
        for _ in range(max_iter):
            scores = frobenius_scores(state)
            weights = torch.softmax(beta * scores, dim=0)
            # Weighted mixture of bound matrices.
            mix = (bound_mat * weights[:, None, None]).sum(dim=0)
            # Retract to nearest unitary (GHRR-native normalization).
            new_state = _unitary_retract(mix)
            energy = float(-(torch.logsumexp(beta * scores, dim=0) / beta).detach().cpu())
            if prev_energy is not None and abs(energy - prev_energy) < tol:
                break
            prev_energy = energy
            state = new_state
        final_scores = frobenius_scores(state)
        final_weights = torch.softmax(beta * final_scores, dim=0)
        top_index = int(torch.argmax(final_scores).detach().cpu())

        # Diagnostics
        if top_index == qi:
            top_idx_hits += 1
        eps = 1e-12
        w_safe = final_weights.clamp(min=eps)
        ent = float(-(w_safe * w_safe.log()).sum() / math.log(N) if N > 1 else 0.0)
        ents.append(ent)
        sorted_, _ = torch.sort(final_scores, descending=True)
        margins.append(float(sorted_[0] - sorted_[1]) if final_scores.numel() >= 2 else 0.0)

        # Algebraic recovery for the headline metric (top-1 over values).
        rec_val_mat = torch.matmul(cue_mat.conj().transpose(-1, -2), state)
        # Frobenius-based top-1 over value codebook.
        val_scores = torch.zeros(N, device=device)
        for j in range(N):
            val_scores[j] = (values_mat[j].conj() * rec_val_mat).real.sum() / m
        pred = int(val_scores.argmax().detach().cpu())
        if pred == qi:
            correct += 1

    return StrategyResult(
        strategy="ghrr_matrix_key_only_native",
        D=fhrr.dim,
        N=N,
        seed=seed,
        n_queries=n_queries,
        n_correct=correct,
        top1=correct / n_queries,
        notes=f"beta={beta},m={m},retract=polar,sim=frobenius",
        top_index_hits=top_idx_hits,
        mean_weight_entropy=sum(ents) / len(ents),
        mean_score_margin=sum(margins) / len(margins),
    )


def _bundle_first_key_only(
    fhrr: TorchFHRR, N: int, n_queries: int, *, seed: int, beta: float
) -> StrategyResult:
    """Bundle-first: storage is the single bundled vector M = Σ bind(k_i, v_i);
    retrieval is `unbind(M, k_j)` followed by a Hopfield clean-up settling
    over the value codebook (NOT over bound pairs). Tests whether the
    bundle-with-MHN-cleanup hybrid that the brainstorm test-results.md
    finding (3) suggested is the natural Phase-5 architecture.

    The MHN landscape here is the **value codebook**, not bound pairs.
    The unbind produces a noisy estimate of v_j; the MHN settles it to
    the nearest stored value. This is the discriminator: does
    Hopfield-as-codebook-cleanup work, even when Hopfield-as-bound-pair-
    associative-memory does not?
    """
    keys = fhrr.random_vectors(N)
    values = fhrr.random_vectors(N)
    M = fhrr.normalize((keys * values).sum(dim=0))

    hop = TorchHopfieldMemory(substrate=fhrr)
    for i in range(N):
        hop.store(values[i], label=f"val_{i}")

    correct = 0
    top_idx_hits = 0
    ents: List[float] = []
    margins: List[float] = []
    for qi in _query_indices(N, n_queries, seed):
        rec = fhrr.unbind(M, keys[qi])
        rec_unit = fhrr.normalize(rec)
        result = hop.retrieve(rec_unit, beta=beta)
        hit, ent, margin = _hopfield_diagnostics(result, qi, N)
        top_idx_hits += hit
        ents.append(ent)
        margins.append(margin)
        sims = (values.conj() * result.state[None, :]).real.mean(dim=1)
        pred = int(sims.argmax().item())
        if pred == qi:
            correct += 1
    return StrategyResult(
        strategy="bundle_first_key_only",
        D=fhrr.dim,
        N=N,
        seed=seed,
        n_queries=n_queries,
        n_correct=correct,
        top1=correct / n_queries,
        notes=f"beta={beta}",
        top_index_hits=top_idx_hits,
        mean_weight_entropy=sum(ents) / len(ents),
        mean_score_margin=sum(margins) / len(margins),
    )


StrategyFn = Callable[..., StrategyResult]

_STRATEGIES: Dict[str, StrategyFn] = {
    "hrr_bundle": _hrr_bundle,
    "hopfield_perfect_cue": _hopfield_perfect_cue,
    "hopfield_key_only": _hopfield_key_only,
    "ghrr_matrix_key_only": _ghrr_matrix_key_only,
    "ghrr_matrix_key_only_native": _ghrr_matrix_key_only_native,
    "bundle_first_key_only": _bundle_first_key_only,
    # Future: "residue_hdc_key_only" plugs in here.
}


def run_cell(
    strategy: str, D: int, N: int, *, seed: int, n_queries: int, device: str, beta: float
) -> StrategyResult:
    fhrr = TorchFHRR(dim=D, seed=seed, device=device)
    fn = _STRATEGIES[strategy]
    if strategy in (
        "hopfield_perfect_cue",
        "hopfield_key_only",
        "ghrr_matrix_key_only",
        "ghrr_matrix_key_only_native",
        "bundle_first_key_only",
    ):
        return fn(fhrr, N, n_queries, seed=seed, beta=beta)
    return fn(fhrr, N, n_queries, seed=seed)


def aggregate(
    cell_results: Sequence[StrategyResult],
) -> Tuple[WilsonCI, List[float], int, int]:
    n_total = sum(r.n_queries for r in cell_results)
    n_correct = sum(r.n_correct for r in cell_results)
    ci = wilson_ci(n_correct, n_total)
    per_seed = [r.top1 for r in cell_results]
    return ci, per_seed, n_correct, n_total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "hrr_bundle",
            "hopfield_perfect_cue",
            "hopfield_key_only",
            "ghrr_matrix_key_only",
            "ghrr_matrix_key_only_native",
            "bundle_first_key_only",
        ],
        choices=sorted(_STRATEGIES.keys()),
    )
    parser.add_argument("--Ds", nargs="+", type=int, default=[4096])
    parser.add_argument(
        "--Ns",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 512],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 11, 23])
    parser.add_argument("--n_queries", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=30.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="reports/phase5_mqar_external_gate/results.json")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"device={device}  strategies={args.strategies}  Ds={args.Ds}  "
          f"Ns={args.Ns}  seeds={args.seeds}  n_queries={args.n_queries}  beta={args.beta}")

    raw_results: List[StrategyResult] = []
    aggregates: Dict[str, Dict] = {}

    for strategy in args.strategies:
        aggregates[strategy] = {}
        for D in args.Ds:
            aggregates[strategy][str(D)] = {}
            for N in args.Ns:
                cell: List[StrategyResult] = []
                for seed in args.seeds:
                    r = run_cell(
                        strategy=strategy,
                        D=D,
                        N=N,
                        seed=seed,
                        n_queries=args.n_queries,
                        device=device,
                        beta=args.beta,
                    )
                    cell.append(r)
                    raw_results.append(r)
                ci, per_seed, n_correct, n_total = aggregate(cell)
                # Aggregate diagnostics (Hopfield-based strategies only — others log -1 / NaN).
                tix_total = sum(r.top_index_hits for r in cell if r.top_index_hits >= 0)
                tix_denom = sum(r.n_queries for r in cell if r.top_index_hits >= 0)
                ent_values = [
                    r.mean_weight_entropy for r in cell
                    if not math.isnan(r.mean_weight_entropy)
                ]
                margin_values = [
                    r.mean_score_margin for r in cell
                    if not math.isnan(r.mean_score_margin)
                ]
                aggregates[strategy][str(D)][str(N)] = {
                    "top1_mean": ci.mean,
                    "wilson_lo": ci.lo,
                    "wilson_hi": ci.hi,
                    "n_total": n_total,
                    "n_correct": n_correct,
                    "per_seed_top1": per_seed,
                    "top_index_hits": tix_total if tix_denom > 0 else None,
                    "top_index_hits_denom": tix_denom if tix_denom > 0 else None,
                    "top_index_hit_rate": (tix_total / tix_denom) if tix_denom > 0 else None,
                    "mean_weight_entropy": (
                        sum(ent_values) / len(ent_values) if ent_values else None
                    ),
                    "mean_score_margin": (
                        sum(margin_values) / len(margin_values) if margin_values else None
                    ),
                }
                tix_str = (
                    f"  tix={tix_total}/{tix_denom}"
                    if tix_denom > 0 else ""
                )
                ent_str = (
                    f"  ent={sum(ent_values)/len(ent_values):.3f}"
                    if ent_values else ""
                )
                print(
                    f"  {strategy:>28s}  D={D:>5d}  N={N:>4d}  "
                    f"top1={ci.mean:.4f}  CI=[{ci.lo:.4f},{ci.hi:.4f}]"
                    f"{tix_str}{ent_str}  "
                    f"per_seed={[f'{x:.3f}' for x in per_seed]}"
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "strategies": args.strategies,
            "Ds": args.Ds,
            "Ns": args.Ns,
            "seeds": args.seeds,
            "n_queries": args.n_queries,
            "beta": args.beta,
            "device": device,
        },
        "aggregates": aggregates,
        "raw": [
            {
                "strategy": r.strategy,
                "D": r.D,
                "N": r.N,
                "seed": r.seed,
                "n_queries": r.n_queries,
                "n_correct": r.n_correct,
                "top1": r.top1,
                "notes": r.notes,
                "top_index_hits": r.top_index_hits,
                "mean_weight_entropy": (
                    None if math.isnan(r.mean_weight_entropy) else r.mean_weight_entropy
                ),
                "mean_score_margin": (
                    None if math.isnan(r.mean_score_margin) else r.mean_score_margin
                ),
            }
            for r in raw_results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
