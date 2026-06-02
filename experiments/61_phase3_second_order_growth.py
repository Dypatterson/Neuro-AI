"""Phase-3 SECOND-ORDER codebook-growth re-scope: does a SPPMI second-order
context-profile operator grow PARADIGMATIC geometry the first-order centroid failed?

Spec (THE precommit; implement exactly):
  notes/emergent-codebook/phase-3-second-order-growth-precommit.md
Forked from:
  experiments/60_phase3_structure_gate_3b.py  (corpus loading, windows/cooc,
  pair-cosine, bootstrap scaffolding, semantic/paradigmatic split)
Re-scope target opened by Report 121 (first-order learner = paradigmatic-NULL).

ARCHITECTURE (precommit §0): grows a SEPARATE PARADIGMATIC codebook G (fresh random
FHRR init via sub.random_vectors) — NOT the validated value/content codebook. Reads
the 3b king/queen test directly on cos(Gi, Gj). The 055-058 recall FLOOR is untouched
by construction (this file does not import phase34/online_codebook.py or phase4/*).

THE OPERATOR (precommit §2, build counts ONCE per arm = Guard-2 per-arm S):
  SPPMI[i,j] = max( log( #(i,j)*|D| / (#(i)*#(j)) ) - log k , 0 ), windowed cooc.
  k is PINNED BY SPPMI DENSITY (target nnz fraction ~0.3-0.5 of V^2); NOT hardcoded
  to 5 (grill G1: k=5 zeroed a small-vocab matrix -> dead). --k and density reported.
  S  = rownorm(SPPMI @ SPPMI^T)   (real VxV second-order context-profile similarity)
  VARIANT B' (PRIMARY): S' = S - rowmean(S) ; centroid = S' @ G   (row-centering
    removes the near-uniform leading eigenvector that causes global contraction).
  VARIANT B  (control) : centroid = S @ G                          (uncentered;
    expected collapse or inert -> shows why B' row-centers).
  VARIANT A  (1st-order control): P = rownorm(C) ; S_A = rownorm(P@P^T) ;
    S_A' = S_A - rowmean ; centroid = S_A' @ G.
  VARIANT C  (robustness, flagged): top-k per-row of S' with fixed uniform k.
  GROWTH: G <- substrate.normalize( a*centroid + (1-a)*G ), decaying a, ~10-30 epochs.
    (G is complex unit-modulus; S' is real; S'@G is a real-weighted bundle -> normalize
    re-projects. Complex matmul forced to CPU mirroring codebook_learner.)

THE REVISED 4-CONDITION GATE (precommit §3 — pass = ALL FOUR):
  1. HEADLINE = demeaned matched specificity, real - shuffle, CI > 0:
     per-pair paradigmatic drift MINUS per-token global-mean drift (demean so global
     contraction cancels) MINUS the matched random-pair arm (random pairs at the SAME
     low-cooc regime), real minus shuffle. Raw drift is NON-DIAGNOSTIC (report only
     beside the collapse panel).
  2. COLLAPSE FLOOR (hard, co-equal): d_eff_end/d_eff_init >= 0.5 AND max-off-diag
     cosine < 0.99 AND global off-diag mean-cosine drift below a frozen ceiling.
  3. DECORRELATION: corr(log cooc, paradigmatic drift) bootstrap 95% CI upper < +0.15
     (WIRED into the pass conjunction; the 60-harness computed corr but did NOT gate).
  4. GAUGE-VALIDITY: corr(S_real_offdiag, S_shuffle_offdiag) < 0.40 else INVALID.
  Pass = (1) AND (2) AND (3) AND (4). HIERARCHICAL seed x pair cluster bootstrap
  (resample seeds AND pairs), not the flat pair-bootstrap.

PAIR SOURCE (precommit §5): SimLex-999 (Hill et al. 2015), filtered to similarity
>= 5.0, in-vocab, within-window cooc <= paradigmatic_max_cooc; obtained from
https://fh295.github.io/SimLex-999.zip and saved to experiments/data/simlex999.txt.
Fallback (if absent) = experiments/60 SEMANTIC_PAIRS, flagged.

SMOKE: --corpus-source synthetic_planted plants a paradigmatic pair (king/queen with
near-identical contexts that NEVER co-occur), a distractor with disjoint contexts, and
filler. B' should lift cos(G_king,G_queen) SPECIFICALLY (> distractor, > random)
WITHOUT collapse; B uncentered should collapse or be inert.

ANTI-HOMUNCULUS (precommit §7): S, S', SPPMI, top-k are fixed offline batch statistics
(same AH class as SPPMI's -log k shift). The update is a local geometric write. The
collapse/corr/gauge gates only halt/label; they never feed back into the update. PASS.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch

from energy_memory.phase2.corpus import (
    build_vocabulary, encode_texts, load_corpus_splits, make_windows,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR

REPO = pathlib.Path(__file__).resolve().parents[1]
SIMLEX_PATH = REPO / "experiments" / "data" / "simlex999.txt"
SIMLEX_URL = "https://fh295.github.io/SimLex-999.zip"

# Fallback semantic pairs (from experiments/60) — used only if SimLex is unavailable.
SEMANTIC_PAIRS = [
    ("king", "queen"), ("man", "woman"), ("men", "women"), ("boy", "girl"),
    ("he", "she"), ("his", "her"), ("father", "mother"), ("son", "daughter"),
    ("brother", "sister"), ("husband", "wife"), ("good", "bad"), ("great", "good"),
    ("big", "large"), ("small", "little"), ("high", "low"), ("old", "new"),
    ("first", "second"), ("one", "two"), ("two", "three"), ("day", "night"),
    ("year", "month"), ("month", "week"), ("north", "south"), ("east", "west"),
    ("city", "town"), ("war", "peace"), ("black", "white"), ("left", "right"),
    ("up", "down"), ("life", "death"), ("house", "home"), ("car", "road"),
    ("water", "river"), ("school", "university"), ("game", "match"), ("team", "club"),
    ("world", "country"), ("president", "government"), ("music", "song"),
    ("book", "story"), ("film", "movie"), ("army", "military"), ("church", "god"),
    ("summer", "winter"), ("morning", "evening"), ("hand", "head"), ("eyes", "face"),
    ("red", "blue"), ("hot", "cold"), ("fast", "slow"),
]


# =====================================================================
# Pair source: SimLex-999 (primary) with frozen filter + hash
# =====================================================================

def load_simlex_pairs(min_sim: float):
    """Return [(w1, w2, sim)] from SimLex-999 at SimLex >= min_sim, plus the file hash.
    Returns (pairs, sha256, source_str) or (None, None, reason) if unavailable."""
    if not SIMLEX_PATH.exists():
        return None, None, f"SimLex not found at {SIMLEX_PATH}"
    import hashlib
    raw = SIMLEX_PATH.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    lines = raw.decode("utf-8").splitlines()
    pairs = []
    for ln in lines[1:]:  # skip header
        parts = ln.split("\t")
        if len(parts) < 4:
            continue
        w1, w2 = parts[0].lower(), parts[1].lower()
        try:
            sim = float(parts[3])
        except ValueError:
            continue
        if sim >= min_sim:
            pairs.append((w1, w2, sim))
    return pairs, sha, f"SimLex-999 (>= {min_sim}); {SIMLEX_URL}"


# =====================================================================
# Synthetic planted corpus (the MANDATORY smoke — validates the mechanism)
# =====================================================================

def make_planted_corpus(seed: int):
    """Token stream with a PLANTED paradigmatic pair.

    king and queen have NEARLY IDENTICAL context distributions over a shared context
    set {ctx0..ctxN} but NEVER co-occur with each other. distractor draws contexts
    from a DISJOINT set {dctx0..dctxM}. filler/noise tokens provide a frequency-
    balanced background. This is the gauge-clean toy from precommit §8: the
    paradigmatic signal lives in second-order context overlap, NOT in any king/queen
    co-occurrence.

    Returns a single long text string (corpus.encode_texts tokenizes it).
    """
    rng = torch.Generator().manual_seed(seed * 31 + 5)

    def rint(hi):
        return int(torch.randint(0, hi, (1,), generator=rng).item())

    n_shared = 12       # shared context tokens for king/queen
    n_distract = 12     # disjoint context tokens for distractor
    n_filler = 30
    shared = [f"ctx{i}" for i in range(n_shared)]
    dctx = [f"dctx{i}" for i in range(n_distract)]
    filler = [f"fill{i}" for i in range(n_filler)]

    tokens = []
    # Each "sentence" places a target token surrounded by a few of its contexts.
    n_sentences = 1600
    for _ in range(n_sentences):
        r = rint(100)
        if r < 30:                       # king sentence
            tgt = "king"
            ctxs = [shared[rint(n_shared)] for _ in range(4)]
        elif r < 60:                     # queen sentence (same context distribution)
            tgt = "queen"
            ctxs = [shared[rint(n_shared)] for _ in range(4)]
        elif r < 80:                     # distractor sentence (disjoint contexts)
            tgt = "distractor"
            ctxs = [dctx[rint(n_distract)] for _ in range(4)]
        else:                            # pure filler sentence (no target)
            tgt = None
            ctxs = [filler[rint(n_filler)] for _ in range(4)]
        # interleave target into the middle of its contexts; king/queen NEVER share a
        # sentence (each sentence has at most one target) -> they never co-occur.
        if tgt is None:
            sent = ctxs
        else:
            sent = ctxs[:2] + [tgt] + ctxs[2:]
        # sprinkle a little filler noise so marginals are not degenerate
        if rint(100) < 40:
            sent = sent + [filler[rint(n_filler)]]
        tokens.extend(sent)

    return " ".join(tokens)


def load_corpus(source: str, args, seed: int):
    """Returns splits dict {train, validation?, test?} of List[str]."""
    if source == "synthetic_planted":
        text = make_planted_corpus(seed)
        # one big doc; held-out PMI not used in the planted smoke
        return {"train": [text], "validation": [text], "test": [text]}
    if source in ("tinystories", "ag_news"):
        # Phase-B generality corpora (different domains from encyclopedic WikiText). ADDITIVE — existing
        # wikitext/repo_sample/synthetic_planted paths unchanged. Only splits["train"] is used downstream
        # (exp73/74). n_docs slice from args.corpus_docs (default 40000 = the exp75 probe slice).
        import itertools
        from datasets import load_dataset
        n = int(getattr(args, "corpus_docs", 40000))
        if source == "tinystories":
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            texts = [r["text"] for r in itertools.islice(ds, n)]
        else:
            ds = load_dataset("fancyzhx/ag_news", split=f"train[:{n}]")
            texts = [r["text"] for r in ds]
        return {"train": texts, "validation": texts[:1], "test": texts[:1]}
    return load_corpus_splits(source, REPO, wikitext_name=args.wikitext_name)


# =====================================================================
# Stream shuffle (gauge-safe control) — reused from experiments/60
# =====================================================================

def stream_shuffle(token_ids, seed):
    g = torch.Generator().manual_seed(seed * 100003 + 7)
    perm = torch.randperm(len(token_ids), generator=g).tolist()
    return [token_ids[i] for i in perm]


# =====================================================================
# Co-occurrence + SPPMI (the operator). Build counts ONCE per arm.
# =====================================================================

def build_cooccurrence(windows, V, special, device="cpu", chunk=50000):
    """Symmetric within-window co-occurrence count matrix C[i,j] (#(i,j)),
    unigram marginals uni[i] (#(i), window-presence based), and |D| = total presence.

    Presence-based: each unordered pair counted once per window where both appear
    (matches the 60-harness _cooc_counts / _heldout_pmi semantics). VECTORIZED as a
    chunked binary-presence Gram matrix: C = sum_chunks(Pᵀ P) with P[w,t]=1 iff token t
    is present in window w (specials zeroed), diagonal (self-presence) zeroed. Counts are
    exact small integers (< 2^24), so float32 chunks accumulated in float64 are
    BIT-IDENTICAL to the prior per-window Python loop. The matmul runs on `device`
    (GPU under CUDA), where each build is ~1s instead of minutes over millions of
    windows — this is the headline-run's dominant cost when it is rebuilt per grid cell.
    """
    if not windows:
        return (torch.zeros((V, V), dtype=torch.float64), torch.zeros(V, dtype=torch.float64), 0.0)
    wt = torch.tensor(windows, dtype=torch.long, device=device)
    n_win = wt.shape[0]
    spec = torch.zeros(V, dtype=torch.bool, device=device)
    for s in special:
        if s < V:
            spec[s] = True
    C = torch.zeros((V, V), dtype=torch.float64, device=device)
    uni = torch.zeros(V, dtype=torch.float64, device=device)
    for start in range(0, n_win, chunk):
        wb = wt[start:start + chunk]                                   # (b, W)
        P = torch.zeros((wb.shape[0], V), dtype=torch.float32, device=device)
        P.scatter_(1, wb, 1.0)                                         # presence (idempotent on dups)
        P[:, spec] = 0.0
        uni += P.sum(dim=0).double()
        C += (P.t() @ P).double()                                      # windows where both present
    C.fill_diagonal_(0.0)
    total = float(uni.sum().item())
    return C.cpu(), uni.cpu(), total


def build_sppmi(C, uni, total, k):
    """SPPMI[i,j] = max( log( #(i,j)*|D| / (#(i)*#(j)) ) - log k , 0 ).
    #(i,j) = C[i,j] (symmetric cooc count). Diagonal zeroed."""
    V = C.shape[0]
    eps = 1e-12
    num = C * total
    denom = uni[:, None] * uni[None, :]
    ratio = num / denom.clamp(min=eps)
    with torch.no_grad():
        pmi = torch.log(ratio.clamp(min=eps)) - math.log(max(k, 1e-9))
    sppmi = pmi.clamp(min=0.0)
    # zero where there is no co-occurrence (log of 0 ratio -> handled, but force clean)
    sppmi = torch.where(C > 0, sppmi, torch.zeros_like(sppmi))
    sppmi.fill_diagonal_(0.0)
    return sppmi


def sppmi_density(sppmi):
    V = sppmi.shape[0]
    off = V * V - V
    nnz = int((sppmi > 0).sum().item()) - int((sppmi.diagonal() > 0).sum().item())
    return nnz / max(off, 1)


def pick_k_by_density(C, uni, total, target_lo, target_hi, k_grid=None):
    """Pin k by SPPMI density: choose k whose nnz-fraction lands in [target_lo, target_hi].
    Returns (k, density, density_report[list of (k, density)])."""
    if k_grid is None:
        k_grid = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120]
    report = []
    best_k, best_d, best_gap = None, None, float("inf")
    target_mid = 0.5 * (target_lo + target_hi)
    for k in k_grid:
        d = sppmi_density(build_sppmi(C, uni, total, k))
        report.append((k, d))
        in_band = target_lo <= d <= target_hi
        gap = 0.0 if in_band else abs(d - target_mid)
        # prefer in-band; among those, prefer largest k (sparsest still-in-band);
        # if none in band, minimize gap to target midpoint.
        if in_band:
            if best_k is None or (best_gap == 0.0 and k > best_k):
                best_k, best_d, best_gap = k, d, 0.0
        elif best_gap > 0.0 and gap < best_gap:
            best_k, best_d, best_gap = k, d, gap
    return best_k, best_d, report


def rownorm(M):
    rs = M.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return M / rs


def build_S(sppmi):
    """S = rownorm(SPPMI @ SPPMI^T): second-order context-profile similarity."""
    G = sppmi @ sppmi.T
    return rownorm(G)


def build_S_first_order(C, special):
    """Variant A inner: P = rownorm(C); S_A = rownorm(P @ P^T)."""
    Cc = C.clone()
    for s in special:
        if s < Cc.shape[0]:
            Cc[s, :] = 0.0
            Cc[:, s] = 0.0
    P = rownorm(Cc)
    return rownorm(P @ P.T)


def row_center(S):
    """S' = S - rowmean(S): removes the near-uniform leading eigenvector."""
    return S - S.mean(dim=1, keepdim=True)


def topk_rows(Sp, k):
    """Variant C: keep top-k per row of S' (fixed uniform k), zero the rest."""
    if k >= Sp.shape[1]:
        return Sp
    out = torch.zeros_like(Sp)
    vals, idx = torch.topk(Sp, k, dim=1)
    out.scatter_(1, idx, vals)
    return out


# =====================================================================
# Growth: G <- normalize( a*centroid + (1-a)*G ), decaying a
# =====================================================================

def grow_G(sub, G_init, operator_matrix, epochs, alpha0, alpha_decay, device, eta_sep=0.0):
    """operator_matrix is the REAL VxV matrix M; centroid = M @ G (a real-weighted bundle
    of the complex unit-modulus rows of G) each epoch. MPS lacks complex matmul, so on MPS
    the matmul runs on CPU (mirrors codebook_learner); CUDA/CPU run it natively on-device —
    a large speedup on GPU, where this matmul is the dominant cost of the headline run.
    (CPU path is numerically identical to the prior CPU-forced version: float32 either way.)

    H_anti KEEP-APART (precommit §GR): if eta_sep > 0 AND sub.alpha_anti > 0, after each pull
    step apply the substrate's energy-native anti-collapse force, FORCE-NORMALIZED so eta_sep
    is a scale-invariant RELATIVE step (the toy's absolute scale won't transfer to D=4096):
    G <- normalize(G + eta_sep * force/mean|force|), force = sub.repulsion_force(G) — the
    -alpha*log(d_eff) gradient on
    the CENTERED Gram (torch_fhrr.py:141,166-182), which removes the common-mode the bundle
    injects (the diagnosed smush cause). alpha_anti is fixed at substrate construction (the
    gradient IS the actuator, not a feedback loop on d_eff) -> anti-homunculus clean.
    eta_sep=0 OR alpha_anti=0 -> repulsion_force returns zeros -> BYTE-IDENTICAL to B' alone.
    (repulsion_force uses complex autograd/matmul; supported on cuda/cpu, not mps.)"""
    G = G_init.clone()
    mps = (str(device) == "mps")
    M = operator_matrix.to(dtype=G.real.dtype)
    M = M.cpu() if mps else M.to(device)
    alpha = alpha0
    for _ in range(epochs):
        Gsrc = G.cpu() if mps else G
        # real-weighted bundle: (V,V) real @ (V,D) complex, split over real/imag
        centroid = torch.matmul(M, Gsrc.real) + 1j * torch.matmul(M, Gsrc.imag)
        if mps:
            centroid = centroid.to(device)
        centroid_norm = sub.normalize(centroid)
        G = sub.normalize(alpha * centroid_norm + (1.0 - alpha) * G)  # B' pull-similar
        if eta_sep > 0.0:                                             # H_anti keep-apart
            force = sub.repulsion_force(G)
            fmag = force.abs().mean().clamp(min=1e-12)                # FORCE-NORMALIZE: scale-invariant
            G = sub.normalize(G + eta_sep * (force / fmag))           # eta_sep = relative step size
        alpha *= alpha_decay
    return G


# =====================================================================
# Pair-cosine + metrics (reused/extended from experiments/60)
# =====================================================================

def pair_cos(cb, pairs):
    if pairs.numel() == 0:
        return torch.zeros(0)
    a = cb[pairs[:, 0]]
    b = cb[pairs[:, 1]]
    return (a.conj() * b).real.mean(dim=1)


def cooc_counts_for_pairs(C, pairs):
    """Co-occurrence count #(i,j) for each pair, from the (already built) C matrix."""
    if pairs.numel() == 0:
        return torch.zeros(0)
    counts = []
    for a, b in pairs.tolist():
        counts.append(float(C[a, b].item()))
    return torch.tensor(counts, dtype=torch.float64)


def offdiag_mean_max_cos(cb, sample_idx):
    """Mean and max off-diagonal pairwise cosine over a sample of rows of cb (FHRR
    similarity = mean real of conj product / D). sample_idx limits cost on large V."""
    sub_cb = cb[sample_idx]
    D = sub_cb.shape[1]
    sim = (sub_cb.conj() @ sub_cb.conj().T.conj()).real / D  # (n,n) real
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    off = sim[mask]
    return float(off.mean().item()), float(off.max().item())


def d_eff(sub, cb):
    return float(sub.d_eff(cb).detach().cpu().real if torch.is_complex(sub.d_eff(cb)) else sub.d_eff(cb).detach().cpu())


# =====================================================================
# Hierarchical seed x pair cluster bootstrap (precommit §5)
# =====================================================================

def hierarchical_bootstrap_ci(per_seed_per_pair, n_boot=2000, seed=0):
    """per_seed_per_pair: list (over seeds) of 1-D tensors (over pairs), all same length.
    Resample SEEDS (with replacement) AND PAIRS (with replacement), recompute the grand
    mean each draw. Returns (mean, lo, hi). Returns nan triple if empty."""
    seeds = [t for t in per_seed_per_pair if t is not None and t.numel() > 0]
    if not seeds:
        return (float("nan"), float("nan"), float("nan"))
    n_seeds = len(seeds)
    n_pairs = seeds[0].numel()
    if n_pairs == 0:
        return (float("nan"), float("nan"), float("nan"))
    stacked = torch.stack(seeds)  # (n_seeds, n_pairs)
    g = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(n_boot):
        s_idx = torch.randint(0, n_seeds, (n_seeds,), generator=g)
        p_idx = torch.randint(0, n_pairs, (n_pairs,), generator=g)
        sample = stacked[s_idx][:, p_idx]
        means.append(float(sample.mean()))
    means.sort()
    grand = float(stacked.mean())
    return (grand, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def flat_bootstrap_ci(values, n_boot=2000, seed=0):
    if values is None or values.numel() == 0:
        return (float("nan"), float("nan"), float("nan"))
    g = torch.Generator().manual_seed(seed)
    n = values.numel()
    means = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        means.append(float(values[idx].mean()))
    means.sort()
    return (float(values.mean()), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def corr_bootstrap_ci(x, y, n_boot=2000, seed=0):
    """Bootstrap CI of Pearson corr(x, y) by resampling pair-index with replacement."""
    if x.numel() < 3:
        return (float("nan"), float("nan"), float("nan"))
    x = x.double()
    y = y.double()
    g = torch.Generator().manual_seed(seed)
    n = x.numel()

    def _corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = (a.norm() * b.norm()).clamp(min=1e-12)
        return float((a @ b) / denom)

    point = _corr(x, y)
    cs = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        xi, yi = x[idx], y[idx]
        if xi.std() < 1e-9 or yi.std() < 1e-9:
            continue
        cs.append(_corr(xi, yi))
    if not cs:
        return (point, float("nan"), float("nan"))
    cs.sort()
    return (point, cs[int(0.025 * len(cs))], cs[int(0.975 * len(cs))])


# =====================================================================
# Random matched pairs (same low-cooc regime as paradigmatic subset)
# =====================================================================

def random_matched_pairs(V, n, seed, special, C, max_cooc):
    """Draw n random in-vocab pairs whose within-window cooc <= max_cooc (matched to the
    paradigmatic low-cooc regime). Frequency is implicitly controlled by drawing from
    the same vocab; the low-cooc filter is the explicit match."""
    g = torch.Generator().manual_seed(seed * 7919 + 3)
    ids = [i for i in range(V) if i not in special]
    pairs = set()
    tries = 0
    max_tries = n * 200 + 1000
    while len(pairs) < n and tries < max_tries:
        tries += 1
        i = ids[int(torch.randint(0, len(ids), (1,), generator=g).item())]
        j = ids[int(torch.randint(0, len(ids), (1,), generator=g).item())]
        if i == j:
            continue
        a, b = min(i, j), max(i, j)
        if (a, b) in pairs:
            continue
        if float(C[a, b].item()) <= max_cooc:
            pairs.add((a, b))
    return list(pairs)


# =====================================================================
# Build the operator matrix M for a given variant + an arm's counts
# =====================================================================

def build_operator(variant, C, uni, total, k, special, topk_c):
    """Return the REAL VxV matrix M such that centroid = M @ G, plus the S/S' used and
    the SPPMI density (for reporting). Variant in {B_prime, B, A, C}."""
    if variant == "A":
        S_A = build_S_first_order(C, special)
        Sp = row_center(S_A)
        return Sp, S_A, float("nan")
    sppmi = build_sppmi(C, uni, total, k)
    dens = sppmi_density(sppmi)
    S = build_S(sppmi)
    if variant == "B":
        return S, S, dens                 # uncentered
    Sp = row_center(S)                    # B' / C share row-centering
    if variant == "C":
        Sp = topk_rows(Sp, topk_c)
    return Sp, S, dens


# =====================================================================
# Main run: one variant across the k x alpha grid, full gate panel
# =====================================================================

def run_variant(args, variant, k, alpha0, pair_words, vocab, real_arm, C_shuf_per_seed,
                special, V, fallback_used):
    """Run `variant` at (k, alpha0) across seeds; return the full gate panel dict.
    `real_arm` = (C_real, uni_real, total_real) and `C_shuf_per_seed[si]` = the seed's
    (C, uni, total) are precomputed ONCE in run() and reused across the grid."""
    device = args.device

    # resolve pairs to ids (in-vocab) once; cooc filter done per the REAL arm's C below.
    def tid(tok):
        return vocab.token_to_id.get(tok)

    # real-arm counts (precomputed once in run(); was rebuilt per cell here).
    C_real, uni_real, total_real = real_arm

    # paradigmatic pair candidates: in vocab, distinct, non-special, cooc <= max
    para_pairs = []
    para_words = []
    for entry in pair_words:
        a_tok, b_tok = entry[0], entry[1]
        ai, bi = tid(a_tok), tid(b_tok)
        if ai is None or bi is None or ai == bi or ai in special or bi in special:
            continue
        a, b = min(ai, bi), max(ai, bi)
        if float(C_real[a, b].item()) <= args.paradigmatic_max_cooc:
            para_pairs.append((a, b))
            para_words.append((a_tok, b_tok))
    para_t = torch.tensor(para_pairs, dtype=torch.long) if para_pairs else torch.zeros((0, 2), dtype=torch.long)

    # matched random pairs (same low-cooc regime), same count as paradigmatic
    n_para = max(len(para_pairs), 1)
    rand_pairs = random_matched_pairs(V, max(n_para, 20), seed=123, special=special,
                                      C=C_real, max_cooc=args.paradigmatic_max_cooc)
    rand_t = torch.tensor(rand_pairs, dtype=torch.long) if rand_pairs else torch.zeros((0, 2), dtype=torch.long)

    # ---- planted-smoke probe: explicit king/queen vs king/distractor vs random ----
    planted = (args.corpus_source == "synthetic_planted")
    pk, pq, pd = tid("king"), tid("queen"), tid("distractor")
    planted_probe = {"kq": [], "kd": [], "qd": [], "rand": []} if planted else None

    # all-vocab off-diag sample for the global-mean demean + collapse panel
    g_samp = torch.Generator().manual_seed(777)
    samp_n = min(args.offdiag_sample, V)
    samp_idx = torch.randperm(V, generator=g_samp)[:samp_n]
    # token-level global-mean drift: per-row mean cosine to all others — approximate by
    # the demean of per-pair drift using the sampled-offdiag mean drift below.

    # per-seed accumulators
    para_real_drift, para_shuf_drift = [], []      # per-seed tensors over para pairs
    rand_real_drift, rand_shuf_drift = [], []
    glob_real_drift, glob_shuf_drift = [], []      # per-seed scalar (sampled off-diag mean)
    deff_init, deff_real, deff_shuf = [], [], []
    offmax_real, offmax_shuf = [], []
    offmean_init, offmean_real, offmean_shuf = [], [], []
    S_real_off_cat, S_shuf_off_cat = [], []         # for gauge-validity corr
    cooc_para = cooc_counts_for_pairs(C_real, para_t)
    # real-arm operator is seed-independent (C_real constant) — build it ONCE.
    M_real, S_real, density_real = build_operator(variant, C_real, uni_real, total_real,
                                                  k, special, args.topk_c)

    for si, seed in enumerate(range(args.seeds)):
        sub = TorchFHRR(dim=args.D, seed=seed, device=device, alpha_anti=args.alpha_anti)
        G_init = sub.random_vectors(V)

        # shuffle-arm operator: use the per-seed counts precomputed once in run()
        # (Guard-2 per-arm S). Real-arm operator was built once above.
        C_shuf, uni_shuf, total_shuf = C_shuf_per_seed[si]
        M_shuf, S_shuf, dens_shuf = build_operator(variant, C_shuf, uni_shuf, total_shuf,
                                                   k, special, args.topk_c)

        # grow G on each arm
        G_real = grow_G(sub, G_init, M_real, args.epochs, alpha0, args.alpha_decay, device, eta_sep=args.eta_sep)
        G_shuf = grow_G(sub, G_init, M_shuf, args.epochs, alpha0, args.alpha_decay, device, eta_sep=args.eta_sep)

        cb_init = G_init.cpu()
        cb_real = G_real.cpu()
        cb_shuf = G_shuf.cpu()

        # ---- planted-smoke probe (drift = end - init, on the REAL arm) ----
        if planted and pk is not None and pq is not None and pd is not None:
            def _c(cb, i, j):
                return float((cb[i].conj() * cb[j]).real.mean().item())
            planted_probe["kq"].append(_c(cb_real, pk, pq) - _c(cb_init, pk, pq))
            planted_probe["kd"].append(_c(cb_real, pk, pd) - _c(cb_init, pk, pd))
            planted_probe["qd"].append(_c(cb_real, pq, pd) - _c(cb_init, pq, pd))
            # random non-target pair drift, matched count
            g_rp = torch.Generator().manual_seed(seed * 13 + 1)
            non_targets = [i for i in range(V) if i not in special and i not in (pk, pq, pd)]
            rdrift = []
            for _ in range(min(30, len(non_targets) // 2)):
                i = non_targets[int(torch.randint(0, len(non_targets), (1,), generator=g_rp).item())]
                j = non_targets[int(torch.randint(0, len(non_targets), (1,), generator=g_rp).item())]
                if i != j:
                    rdrift.append(_c(cb_real, i, j) - _c(cb_init, i, j))
            planted_probe["rand"].append(sum(rdrift) / max(len(rdrift), 1))

        # pair drifts (end - init)
        c0_para = pair_cos(cb_init, para_t)
        para_real_drift.append(pair_cos(cb_real, para_t) - c0_para)
        para_shuf_drift.append(pair_cos(cb_shuf, para_t) - c0_para)
        c0_rand = pair_cos(cb_init, rand_t)
        rand_real_drift.append(pair_cos(cb_real, rand_t) - c0_rand)
        rand_shuf_drift.append(pair_cos(cb_shuf, rand_t) - c0_rand)

        # global-mean off-diag drift (per-token global structure) on the sample
        om_init, omx_init = offdiag_mean_max_cos(cb_init, samp_idx)
        om_real, omx_real = offdiag_mean_max_cos(cb_real, samp_idx)
        om_shuf, omx_shuf = offdiag_mean_max_cos(cb_shuf, samp_idx)
        glob_real_drift.append(om_real - om_init)
        glob_shuf_drift.append(om_shuf - om_init)
        offmean_init.append(om_init)
        offmean_real.append(om_real)
        offmean_shuf.append(om_shuf)
        offmax_real.append(omx_real)
        offmax_shuf.append(omx_shuf)

        deff_init.append(d_eff(sub, G_init))
        deff_real.append(d_eff(sub, G_real))
        deff_shuf.append(d_eff(sub, G_shuf))

        # gauge-validity: off-diagonal of S_real vs S_shuf on the sample
        Sr = S_real[samp_idx][:, samp_idx]
        Ss = S_shuf[samp_idx][:, samp_idx]
        n = Sr.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool)
        S_real_off_cat.append(Sr[mask].flatten())
        S_shuf_off_cat.append(Ss[mask].flatten())

        print(f"[{variant} k={k} a={alpha0} seed {seed}] "
              f"para real Δ̄={float(para_real_drift[-1].mean()):+.4f} "
              f"shuf Δ̄={float(para_shuf_drift[-1].mean()):+.4f} | "
              f"glob real Δ̄={glob_real_drift[-1]:+.4f} | "
              f"d_eff i/r/s={deff_init[-1]:.0f}/{deff_real[-1]:.0f}/{deff_shuf[-1]:.0f} | "
              f"offmax r={offmax_real[-1]:.3f}",
              file=sys.stderr, flush=True)

    # ---------- HEADLINE: demeaned matched specificity, real - shuffle ----------
    # per-seed per-pair: demeaned paradigmatic drift = para_drift - glob_drift (per seed),
    # then minus the matched-random-pair arm (mean over random pairs, per seed),
    # then real minus shuffle.
    def demeaned_matched(per_seed_para, per_seed_glob, per_seed_rand):
        out = []
        for s in range(len(per_seed_para)):
            para = per_seed_para[s]                      # (n_para,)
            glob = per_seed_glob[s]                      # scalar
            rand_mean = float(per_seed_rand[s].mean()) if per_seed_rand[s].numel() else 0.0
            out.append(para - glob - rand_mean)          # (n_para,) demeaned + matched
        return out

    real_dmm = demeaned_matched(para_real_drift, glob_real_drift, rand_real_drift)
    shuf_dmm = demeaned_matched(para_shuf_drift, glob_shuf_drift, rand_shuf_drift)
    # real - shuffle per seed per pair
    headline_per_seed = [real_dmm[s] - shuf_dmm[s] for s in range(len(real_dmm))]
    h_mean, h_lo, h_hi = hierarchical_bootstrap_ci(headline_per_seed, seed=31)
    # GAUGE-FREE headline (precommit §GR): para-vs-random specificity on the REAL arm =
    # mean(para_drift) - mean(random_drift), matching the §10 SVD oracle's gauge-free read.
    # NOTE (2026-06-01 fix): the matched-RANDOM arm is the contraction control here, so do NOT
    # ALSO subtract the global-mean drift (`real_dmm` does both -> a DOUBLE-subtraction that
    # biased the gauge-free headline negative whenever the codebook contracts). Contraction is
    # handled by the SEPARATE collapse floor, not by the headline. The stream-shuffle gauge is
    # RETIRED for 2nd-order operators (it leaks ~0.79: SPPMI is frequency-dominated).
    def matched_only(per_seed_para, per_seed_rand):
        out = []
        for s in range(len(per_seed_para)):
            rand_mean = float(per_seed_rand[s].mean()) if per_seed_rand[s].numel() else 0.0
            out.append(per_seed_para[s] - rand_mean)     # para - random (NO glob double-subtract)
        return out
    real_para_vs_rand = matched_only(para_real_drift, rand_real_drift)
    gf_mean, gf_lo, gf_hi = hierarchical_bootstrap_ci(real_para_vs_rand, seed=34)
    # also the flat version for reporting
    if headline_per_seed and headline_per_seed[0].numel() > 0:
        flat_vals = torch.stack(headline_per_seed).mean(dim=0)
        hf_mean, hf_lo, hf_hi = flat_bootstrap_ci(flat_vals, seed=32)
    else:
        hf_mean = hf_lo = hf_hi = float("nan")

    # ---------- RAW drift (NON-DIAGNOSTIC; report beside collapse) ----------
    if para_real_drift and para_real_drift[0].numel() > 0:
        raw_real = torch.stack(para_real_drift).mean(dim=0)
        raw_shuf = torch.stack(para_shuf_drift).mean(dim=0)
        raw_per_seed = [para_real_drift[s] - para_shuf_drift[s] for s in range(len(para_real_drift))]
        raw_mean, raw_lo, raw_hi = hierarchical_bootstrap_ci(raw_per_seed, seed=33)
    else:
        raw_mean = raw_lo = raw_hi = float("nan")

    # ---------- COLLAPSE FLOOR ----------
    deff_init_m = sum(deff_init) / len(deff_init)
    deff_real_m = sum(deff_real) / len(deff_real)
    deff_shuf_m = sum(deff_shuf) / len(deff_shuf)
    deff_ratio = deff_real_m / max(deff_init_m, 1e-9)
    offmax_real_m = max(offmax_real)        # worst-case
    offmean_drift_real = (sum(offmean_real) / len(offmean_real)) - (sum(offmean_init) / len(offmean_init))
    collapse_ok = bool(deff_ratio >= 0.5 and offmax_real_m < 0.99
                       and offmean_drift_real < args.offdiag_drift_ceiling)

    # ---------- DECORRELATION: corr(log cooc, paradigmatic drift) ----------
    if cooc_para.numel() >= 3 and para_real_drift and para_real_drift[0].numel() > 0:
        log_cooc = torch.log(cooc_para.float() + 1.0)
        para_drift_mean = torch.stack(para_real_drift).mean(dim=0).double()
        corr_pt, corr_lo, corr_hi = corr_bootstrap_ci(log_cooc, para_drift_mean, seed=41)
    else:
        corr_pt = corr_lo = corr_hi = float("nan")
    decorr_ok = bool(corr_hi == corr_hi and corr_hi < 0.15)

    # ---------- GAUGE-VALIDITY: corr(S_real_offdiag, S_shuffle_offdiag) ----------
    if S_real_off_cat:
        sr = torch.cat(S_real_off_cat).double()
        ss = torch.cat(S_shuf_off_cat).double()
        # subsample for speed if huge
        if sr.numel() > 200000:
            g = torch.Generator().manual_seed(55)
            idx = torch.randperm(sr.numel(), generator=g)[:200000]
            sr, ss = sr[idx], ss[idx]
        a = sr - sr.mean()
        b = ss - ss.mean()
        denom = (a.norm() * b.norm()).clamp(min=1e-12)
        gauge_corr = float((a @ b) / denom) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0
    else:
        gauge_corr = float("nan")
    # Signed corr < 0.40 per the ratified spec (leakage = POSITIVE corr: shuffle S
    # retaining real co-occurrence structure). Strong negative corr is not leakage.
    gauge_valid = bool(gauge_corr == gauge_corr and gauge_corr < 0.40)

    # ---------- PASS CONJUNCTION ----------
    # --gate shuffle (legacy): demeaned-matched real-MINUS-shuffle headline + gauge-validity.
    # --gate gauge_free (precommit §GR, the growth-redesign default): para-vs-random
    #   specificity on the REAL arm; the stream-shuffle gauge is RETIRED (it leaks ~0.79 for
    #   the 2nd-order operator), so the headline drops the shuffle subtraction and the
    #   gauge-validity gate is dropped from the conjunction (still reported).
    if getattr(args, "gate", "shuffle") == "gauge_free":
        headline_ci_pos = bool(gf_lo == gf_lo and gf_lo > 0.0)
        variant_pass = bool(headline_ci_pos and collapse_ok and decorr_ok)
    else:
        headline_ci_pos = bool(h_lo == h_lo and h_lo > 0.0)
        variant_pass = bool(headline_ci_pos and collapse_ok and decorr_ok and gauge_valid)

    # ---------- planted-smoke probe summary ----------
    planted_summary = None
    if planted and planted_probe is not None and planted_probe["kq"]:
        def _m(lst):
            return sum(lst) / len(lst) if lst else float("nan")
        kq_m, kd_m, qd_m, rnd_m = (_m(planted_probe["kq"]), _m(planted_probe["kd"]),
                                   _m(planted_probe["qd"]), _m(planted_probe["rand"]))
        # THREADS THE NEEDLE: king/queen lifts SPECIFICALLY (> distractor pairs AND
        # > random) WITHOUT collapse (d_eff ratio >= 0.5).
        specific = bool(kq_m > kd_m and kq_m > qd_m and kq_m > rnd_m)
        threads_needle = bool(specific and deff_ratio >= 0.5 and offmax_real_m < 0.99)
        planted_summary = {
            "drift_king_queen": kq_m,
            "drift_king_distractor": kd_m,
            "drift_queen_distractor": qd_m,
            "drift_random_nontarget_mean": rnd_m,
            "kq_per_seed": planted_probe["kq"],
            "kq_gt_distractor_and_random": specific,
            "d_eff_ratio": deff_ratio,
            "no_collapse": bool(deff_ratio >= 0.5 and offmax_real_m < 0.99),
            "THREADS_NEEDLE": threads_needle,
        }

    return {
        "variant": variant, "k": k, "alpha0": alpha0,
        "n_paradigmatic_pairs": int(para_t.shape[0]),
        "n_random_matched_pairs": int(rand_t.shape[0]),
        "sppmi_density_real": density_real,
        "fallback_pairs_used": fallback_used,
        "gate_used": getattr(args, "gate", "shuffle"),
        # GATE 1a (legacy shuffle headline — gauge-INVALID for SPPMI, reported only)
        "HEADLINE_demeaned_matched_real_minus_shuffle": {
            "hierarchical_mean": h_mean, "hierarchical_ci": [h_lo, h_hi],
            "flat_mean": hf_mean, "flat_ci": [hf_lo, hf_hi],
            "CI_gt_0": bool(h_lo == h_lo and h_lo > 0.0),
        },
        # GATE 1b (gauge_free, precommit §GR — the growth-redesign HEADLINE)
        "HEADLINE_gauge_free_para_vs_random": {
            "hierarchical_mean": gf_mean, "hierarchical_ci": [gf_lo, gf_hi],
            "CI_gt_0": bool(gf_lo == gf_lo and gf_lo > 0.0),
        },
        "raw_drift_real_minus_shuffle_NONDIAGNOSTIC": {
            "mean": raw_mean, "ci": [raw_lo, raw_hi],
        },
        # GATE 2
        "collapse_floor": {
            "d_eff_init": deff_init_m, "d_eff_real": deff_real_m, "d_eff_shuffle": deff_shuf_m,
            "d_eff_ratio_real": deff_ratio, "ratio_ge_0p5": bool(deff_ratio >= 0.5),
            "offdiag_max_real": offmax_real_m, "offmax_lt_0p99": bool(offmax_real_m < 0.99),
            "offdiag_mean_drift_real": offmean_drift_real,
            "drift_ceiling": args.offdiag_drift_ceiling,
            "drift_below_ceiling": bool(offmean_drift_real < args.offdiag_drift_ceiling),
            "COLLAPSE_OK": collapse_ok,
        },
        # GATE 3
        "decorrelation": {
            "corr_logcooc_drift_point": corr_pt, "corr_ci": [corr_lo, corr_hi],
            "ci_hi_lt_0p15": decorr_ok,
        },
        # GATE 4
        "gauge_validity": {
            "corr_S_real_shuffle_offdiag": gauge_corr, "lt_0p40": gauge_valid,
            "VALID": gauge_valid,
        },
        "VARIANT_PASS": variant_pass,
        "planted_probe": planted_summary,
        "para_words_sample": para_words[:8],
    }


def run(args):
    device = args.device
    fallback_used = False
    simlex_status = ""

    # ---- pair source ----
    if args.corpus_source == "synthetic_planted":
        # The planted smoke uses its OWN pair set: king/queen is the planted
        # paradigmatic pair; the standalone planted-probe (below) reports king/queen
        # vs king/distractor vs random directly.
        simlex_status = "synthetic_planted: planted pair king/queen (probe reports specificity)"
        pair_words = [("king", "queen", 99.0)]
        fallback_used = True
    elif args.pair_source == "simlex":
        pairs, sha, src = load_simlex_pairs(args.simlex_min_sim)
        if pairs is None:
            fallback_used = True
            simlex_status = f"FALLBACK to SEMANTIC_PAIRS ({src})"
            pair_words = [(a, b, 99.0) for a, b in SEMANTIC_PAIRS]
        else:
            simlex_status = f"{src}; n_at_threshold={len(pairs)}; sha256={sha[:16]}"
            pair_words = pairs
    else:
        fallback_used = True
        simlex_status = "SEMANTIC_PAIRS (forced)"
        pair_words = [(a, b, 99.0) for a, b in SEMANTIC_PAIRS]

    # ---- corpus + vocab ----
    splits = load_corpus(args.corpus_source, args, seed=0)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    special = {vocab.unk_id, vocab.mask_id}
    V = len(vocab.id_to_token)
    train_ids = encode_texts(splits["train"], vocab)
    train_windows = make_windows(train_ids, args.W)

    # precompute per-seed stream-shuffled windows (shuffle the FLAT stream per seed)
    sh_windows_per_seed = []
    for seed in range(args.seeds):
        sh_ids = stream_shuffle(train_ids, seed)
        sh_windows_per_seed.append(make_windows(sh_ids, args.W))

    # ---- co-occurrence: build ONCE per arm and reuse across the whole variant x k x
    # alpha grid. Previously C_real was rebuilt per cell (~28x) and C_shuf per cell x seed
    # (~140x) inside run_variant — the dominant cost of the headline run. Now: real once,
    # shuffle once per seed, on-device (GPU). ----
    C_real, uni_real, total_real = build_cooccurrence(train_windows, V, special, device=args.device)
    real_arm = (C_real, uni_real, total_real)
    C_shuf_per_seed = [build_cooccurrence(sh_windows_per_seed[si], V, special, device=args.device)
                       for si in range(args.seeds)]

    # ---- pin k by density (on the real arm) unless --k given ----
    if args.k is not None:
        k_chosen = args.k
        dens = sppmi_density(build_sppmi(C_real, uni_real, total_real, k_chosen))
        density_report = [(k_chosen, dens)]
    else:
        k_chosen, dens, density_report = pick_k_by_density(
            C_real, uni_real, total_real, args.density_lo, args.density_hi)
        if k_chosen is None:
            # No grid k hit the density band. Pick the DENSEST grid k (not a silent
            # k=5 — grill-G1 flagged k=5 as the dead config that zeroed a small-vocab
            # SPPMI to 26/1681 nnz). Warn loudly so a degenerate-density run is visible.
            k_chosen, dens = max(density_report, key=lambda kd: kd[1])
            print(f"!! WARNING: no k hit the density band [{args.density_lo},{args.density_hi}]; "
                  f"falling back to the densest grid k={k_chosen} (density={dens:.3f}). "
                  f"Low density risks the grill-G1 'dead/inert' regime — inspect the "
                  f"density report and the gauge-validity gate before banking a verdict.",
                  file=sys.stderr, flush=True)

    print(f"=== V={V} train_windows={len(train_windows)} | k_chosen={k_chosen} "
          f"density={dens:.3f} (target {args.density_lo}-{args.density_hi}) ===",
          file=sys.stderr, flush=True)
    print(f"=== density report (k, nnz_frac): {density_report} ===", file=sys.stderr, flush=True)

    # k x alpha grid (centered on k_chosen)
    if args.k_grid:
        k_grid = [int(x) for x in args.k_grid.split(",")]
    else:
        k_grid = sorted(set([max(1, k_chosen // 2), k_chosen, k_chosen * 2]))
    if args.alpha_grid:
        alpha_grid = [float(x) for x in args.alpha_grid.split(",")]
    else:
        alpha_grid = [args.alpha0]

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    grid_results = []
    for variant in variants:
        # A ignores k (first-order); run once at k_grid[0]
        v_kgrid = [k_chosen] if variant == "A" else k_grid
        for k in v_kgrid:
            for a0 in alpha_grid:
                res = run_variant(args, variant, k, a0, pair_words, vocab, real_arm,
                                  C_shuf_per_seed, special, V, fallback_used)
                grid_results.append(res)

    # ---- pick the headline cell per primary variant (B') for the verdict ----
    def best_cell(variant):
        cells = [r for r in grid_results if r["variant"] == variant]
        # prefer a PASS; else the one with the highest headline mean among collapse-ok
        passes = [r for r in cells if r["VARIANT_PASS"]]
        if passes:
            return passes[0]
        ok = [r for r in cells if r["collapse_floor"]["COLLAPSE_OK"]]
        pool = ok or cells
        return max(pool, key=lambda r: (r["HEADLINE_demeaned_matched_real_minus_shuffle"]["hierarchical_mean"]
                                        if r["HEADLINE_demeaned_matched_real_minus_shuffle"]["hierarchical_mean"] == r["HEADLINE_demeaned_matched_real_minus_shuffle"]["hierarchical_mean"]
                                        else -1e9))

    b_prime_cell = best_cell("B_prime") if any(r["variant"] == "B_prime" for r in grid_results) else None
    any_pass = any(r["VARIANT_PASS"] for r in grid_results)

    summary = {
        "config": {
            "corpus_source": args.corpus_source, "D": args.D, "W": args.W,
            "epochs": args.epochs, "alpha0": args.alpha0, "alpha_decay": args.alpha_decay,
            "seeds": args.seeds, "max_vocab": args.max_vocab,
            "paradigmatic_max_cooc": args.paradigmatic_max_cooc,
            "k_chosen_by_density": k_chosen, "density": dens,
            "density_target": [args.density_lo, args.density_hi],
            "offdiag_drift_ceiling": args.offdiag_drift_ceiling,
            "alpha_anti": args.alpha_anti, "eta_sep": args.eta_sep, "gate": args.gate,
            "variants": variants, "k_grid": k_grid, "alpha_grid": alpha_grid,
        },
        "pair_source": simlex_status,
        "fallback_pairs_used": fallback_used,
        "density_report_k_nnzfrac": density_report,
        "grid_results": grid_results,
        "B_prime_headline_cell": b_prime_cell,
        "ANY_VARIANT_PASS": any_pass,
        "verdict": ("PASS — a second-order variant clears the 4-condition paradigmatic gate"
                    if any_pass else
                    "NULL on the 4-condition gate — per precommit §10 pre-registered "
                    "disposition: run SVD-of-SPPMI + FHRR-port oracles; do NOT scale vocab/epochs."),
    }

    print(json.dumps(summary, indent=2))
    if args.out:
        outp = pathlib.Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(outp, "w"), indent=2)
        print(f"wrote {args.out}", file=sys.stderr)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source",
                    choices=["repo_sample", "wikitext", "synthetic_planted"],
                    default="repo_sample", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--pair-source", choices=["simlex", "semantic"], default="simlex",
                    dest="pair_source")
    ap.add_argument("--simlex-min-sim", type=float, default=5.0, dest="simlex_min_sim")
    ap.add_argument("--variants", default="B_prime,B,A",
                    help="comma list of B_prime,B,A,C")
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--alpha0", type=float, default=0.3)
    ap.add_argument("--alpha-decay", type=float, default=0.9, dest="alpha_decay")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")
    ap.add_argument("--k", type=int, default=None,
                    help="SPPMI shift k; if unset, pinned by density")
    ap.add_argument("--k-grid", default="", dest="k_grid",
                    help="comma list to override the k grid")
    ap.add_argument("--alpha-grid", default="", dest="alpha_grid",
                    help="comma list to override the alpha grid")
    ap.add_argument("--density-lo", type=float, default=0.3, dest="density_lo")
    ap.add_argument("--density-hi", type=float, default=0.5, dest="density_hi")
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2,
                    dest="paradigmatic_max_cooc")
    ap.add_argument("--offdiag-sample", type=int, default=400, dest="offdiag_sample",
                    help="rows sampled for the all-vocab off-diag collapse/gauge panel")
    ap.add_argument("--offdiag-drift-ceiling", type=float, default=0.10,
                    dest="offdiag_drift_ceiling",
                    help="frozen ceiling on global off-diag mean-cosine drift (collapse gate)")
    ap.add_argument("--topk-c", type=int, default=20, dest="topk_c",
                    help="Variant C: fixed uniform top-k per row of S'")
    ap.add_argument("--alpha-anti", type=float, default=0.0, dest="alpha_anti",
                    help="H_anti = -alpha*log(d_eff) energy strength, FIXED at substrate "
                         "construction (anti-homunculus: not adapted from observed d_eff). "
                         "0.0 -> inert -> byte-identical to B' alone (precommit §GR).")
    ap.add_argument("--eta-sep", type=float, default=0.0, dest="eta_sep",
                    help="FORCE-NORMALIZED relative step for the H_anti keep-apart in grow_G "
                         "(scale-invariant: the step has mean magnitude eta_sep relative to "
                         "the atom magnitude ~1); 0.0 -> off. Sweep ~{0.02..0.4} (precommit §GR).")
    ap.add_argument("--gate", choices=["shuffle", "gauge_free"], default="shuffle",
                    help="shuffle (legacy): real-minus-shuffle headline + gauge-validity. "
                         "gauge_free (precommit §GR): para-vs-random specificity on the REAL "
                         "arm; the stream-shuffle gauge is retired (it leaks ~0.79 for SPPMI).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
