"""§10 oracles for the experiment-61 NULL (precommit §10, grill G6).

Experiment 61 (row-centered SPPMI second-order growth on a SEPARATE paradigmatic
codebook G) returned a NULL on the WikiText-2 4-condition gate (A ≈ B′; SPPMI gauge
leaks 0.79; no collapse-free decorrelated signal). The pre-registered disposition is
NOT to scale, but to run two gauge-FREE oracles that disambiguate WHERE the null comes
from:

  ORACLE 1 — SVD-of-SPPMI (substrate-free SIGNAL oracle). Truncated SVD of the SPPMI
    matrix (Levy-Goldberg's OWN word-embedding construction, W = U_r · Σ_r^0.5) read on
    the curated paradigmatic SimLex pairs (non-co-occurring) vs matched-random pairs.
    Tests: does the paradigmatic signal EXIST in WikiText-2's SPPMI statistics at this
    scale AT ALL — with zero FHRR re-bundle and no stream-shuffle control?
      • SVD NULL (para ≈ random) -> signal genuinely insufficient at this scale ->
        Option-B latent layer licensed.
      • SVD PASS (para ≫ random) -> the signal exists; the FHRR growth/gauge destroyed it.

  ORACLE 2 — FHRR-port (single-shot). Take the SAME real SPPMI, build the project's
    operator S' = rownorm(SPPMI@SPPMIᵀ) − rowmean, do ONE port step centroid = S'@G,
    normalize (no multi-epoch growth), read the same paradigmatic vs random arms in
    FHRR space. Isolates whether the single FHRR re-bundle preserves what SVD finds.

SANITY arm: collocational SimLex pairs (sim≥5 but co-occurring) — a working SVD should
place these close (co-occurrence -> similar). If even collocational is flat, the corpus
is too small for ANY distributional signal (a scale verdict, not a mechanism verdict).

Reuses experiment 61's EXACT functions via importlib (no drift). Offline; no growth grid.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]

# ---- load experiment 61's functions (module name starts with a digit) ----
_spec = importlib.util.spec_from_file_location(
    "exp61", REPO / "experiments" / "61_phase3_second_order_growth.py")
exp61 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp61)

from energy_memory.phase2.corpus import (  # noqa: E402
    build_vocabulary, encode_texts, load_corpus_splits, make_windows,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402


def _cos_real(W, pairs):
    """Standard cosine on real rows of W for each [a,b] pair."""
    if not pairs:
        return torch.zeros(0)
    a = W[[p[0] for p in pairs]]
    b = W[[p[1] for p in pairs]]
    num = (a * b).sum(dim=1)
    den = (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-12)
    return num / den


def _fcos(G, pairs):
    """FHRR similarity (mean real of conj-product) for each [a,b] pair."""
    if not pairs:
        return torch.zeros(0)
    a = G[[p[0] for p in pairs]]
    b = G[[p[1] for p in pairs]]
    return (a.conj() * b).real.mean(dim=1)


def _boot_diff(x, y, n_boot=4000, seed=7):
    """Bootstrap 95% CI of mean(x) - mean(y) (independent resample of each)."""
    if x.numel() == 0 or y.numel() == 0:
        return (float("nan"), float("nan"), float("nan"))
    g = torch.Generator().manual_seed(seed)
    diffs = []
    for _ in range(n_boot):
        xi = x[torch.randint(0, x.numel(), (x.numel(),), generator=g)]
        yi = y[torch.randint(0, y.numel(), (y.numel(),), generator=g)]
        diffs.append(float(xi.mean() - yi.mean()))
    diffs.sort()
    return (float(x.mean() - y.mean()), diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source", default="wikitext", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")
    ap.add_argument("--k", type=int, default=None, help="SPPMI shift; default pin by density")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2, dest="paradigmatic_max_cooc")
    ap.add_argument("--simlex-min-sim", type=float, default=5.0, dest="simlex_min_sim")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # ---- corpus + vocab + windows + co-occurrence + SPPMI (match the exp-61 headline) ----
    splits = load_corpus_splits(args.corpus_source, REPO, wikitext_name=args.wikitext_name)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    special = {vocab.unk_id, vocab.mask_id}
    V = len(vocab.id_to_token)
    train_ids = encode_texts(splits["train"], vocab)
    windows = make_windows(train_ids, args.W)
    print(f"V={V} windows={len(windows)}", file=sys.stderr, flush=True)

    C, uni, total = exp61.build_cooccurrence(windows, V, special, device=args.device)
    if args.k is not None:
        k = args.k
        dens = exp61.sppmi_density(exp61.build_sppmi(C, uni, total, k))
    else:
        k, dens, _ = exp61.pick_k_by_density(C, uni, total, 0.3, 0.5)
        if k is None:
            k = 1
            dens = exp61.sppmi_density(exp61.build_sppmi(C, uni, total, k))
    sppmi = exp61.build_sppmi(C, uni, total, k)
    print(f"k={k} sppmi_density={dens:.3f}", file=sys.stderr, flush=True)

    # ---- SimLex pairs -> ids; split paradigmatic (non-cooc) vs collocational (cooc) ----
    pairs, sha, src = exp61.load_simlex_pairs(args.simlex_min_sim)
    if pairs is None:
        print("SimLex unavailable:", src, file=sys.stderr); raise SystemExit(1)

    def tid(t):
        return vocab.token_to_id.get(t)

    para, collo = [], []
    for a_tok, b_tok, _sim in pairs:
        ai, bi = tid(a_tok), tid(b_tok)
        if ai is None or bi is None or ai == bi or ai in special or bi in special:
            continue
        a, b = min(ai, bi), max(ai, bi)
        (para if float(C[a, b]) <= args.paradigmatic_max_cooc else collo).append((a, b))
    rand = exp61.random_matched_pairs(V, max(len(para), 20), seed=123, special=special,
                                      C=C, max_cooc=args.paradigmatic_max_cooc)
    kq = [(min(tid("king"), tid("queen")), max(tid("king"), tid("queen")))] \
        if tid("king") is not None and tid("queen") is not None else []
    print(f"n_para={len(para)} n_collo={len(collo)} n_rand={len(rand)} "
          f"king/queen_in_vocab={bool(kq)}", file=sys.stderr, flush=True)

    # ======================= ORACLE 1: SVD-of-SPPMI =======================
    r = min(args.svd_rank, V)
    U, Sg, _ = torch.linalg.svd(sppmi, full_matrices=False)
    Wsvd = U[:, :r] * Sg[:r].clamp(min=0).sqrt()           # L&G symmetric embedding

    svd_para = _cos_real(Wsvd, para)
    svd_collo = _cos_real(Wsvd, collo)
    svd_rand = _cos_real(Wsvd, rand)
    svd_kq = _cos_real(Wsvd, kq)
    spec_mean, spec_lo, spec_hi = _boot_diff(svd_para, svd_rand)
    coll_mean, coll_lo, coll_hi = _boot_diff(svd_collo, svd_rand)
    svd_signal_exists = bool(spec_lo == spec_lo and spec_lo > 0.0)

    # ======================= ORACLE 2: FHRR-port (single-shot) =======================
    sub = TorchFHRR(dim=args.D, seed=0, device=args.device)
    G0 = sub.random_vectors(V)
    S = exp61.build_S(sppmi)                                # rownorm(SPPMI@SPPMIᵀ)
    Sp = exp61.row_center(S)                                # B′ operator: S − rowmean
    M = Sp.to(dtype=G0.real.dtype)
    centroid = (M @ G0.real) + 1j * (M @ G0.imag)          # single real-weighted bundle
    Gp = sub.normalize(centroid)
    Gp = Gp.cpu()
    fhrr_para = _fcos(Gp, para)
    fhrr_collo = _fcos(Gp, collo)
    fhrr_rand = _fcos(Gp, rand)
    fhrr_kq = _fcos(Gp, kq)
    fspec_mean, fspec_lo, fspec_hi = _boot_diff(fhrr_para, fhrr_rand)
    fhrr_signal_exists = bool(fspec_lo == fspec_lo and fspec_lo > 0.0)

    def m(t):
        return float(t.mean()) if t.numel() else float("nan")

    # ---- verdict ----
    if not svd_signal_exists:
        verdict = ("SVD-of-SPPMI NULL: the paradigmatic signal does NOT exist in "
                   "WikiText-2's SPPMI statistics at this scale (para ≈ random). The "
                   "re-scope is FALSIFIED at this scale -> the §10 disposition licenses "
                   "the Option-B latent/predictor layer. (Mechanism is not the issue; the "
                   "flat corpus statistics do not carry paradigmatic structure here.)")
    elif not fhrr_signal_exists:
        verdict = ("SVD-of-SPPMI PASS but FHRR-port NULL: the paradigmatic signal EXISTS "
                   "in the SPPMI statistics, but the FHRR S'@G re-bundle/normalize destroys "
                   "it -> fix the PORT (e.g. diagonal-zeroing / spectral control), not the "
                   "corpus. Re-scope is NOT falsified.")
    else:
        verdict = ("SVD-of-SPPMI PASS and FHRR-port PASS: signal exists and survives a "
                   "single FHRR port -> the multi-epoch growth dynamics / gauge config (not "
                   "the corpus or the port) are the issue; the growth schedule is the lever.")

    out = {
        "config": {"V": V, "windows": len(windows), "k": k, "sppmi_density": dens,
                   "svd_rank": r, "D": args.D, "n_para": len(para), "n_collo": len(collo),
                   "n_rand": len(rand), "simlex": f"{src}; sha256={sha[:16] if sha else None}"},
        "ORACLE1_svd_of_sppmi": {
            "para_cos_mean": m(svd_para), "collo_cos_mean": m(svd_collo),
            "random_cos_mean": m(svd_rand), "king_queen_cos": m(svd_kq),
            "specificity_para_minus_random": {"mean": spec_mean, "ci": [spec_lo, spec_hi],
                                              "SIGNAL_EXISTS": svd_signal_exists},
            "collocational_minus_random": {"mean": coll_mean, "ci": [coll_lo, coll_hi]},
        },
        "ORACLE2_fhrr_port_single_shot": {
            "para_cos_mean": m(fhrr_para), "collo_cos_mean": m(fhrr_collo),
            "random_cos_mean": m(fhrr_rand), "king_queen_cos": m(fhrr_kq),
            "specificity_para_minus_random": {"mean": fspec_mean, "ci": [fspec_lo, fspec_hi],
                                              "SIGNAL_SURVIVES_PORT": fhrr_signal_exists},
        },
        "VERDICT": verdict,
    }
    import json
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
