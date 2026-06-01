"""R3 directional-successor SPECTRAL-REACHABILITY oracle (DRILL-DOWN, not graduation).

Frozen pre-commit: notes/emergent-codebook/phase-3-r3-directional-oracle-precommit.md.

THE QUESTION (Report 123 §5 R3 conjecture). Report 123 showed the paradigmatic
("substitutability", king/queen) signal EXISTS in WikiText-2's SPPMI statistics but lives
in the SUBDOMINANT modes of the SYMMETRIC second-order operator — a GLOBAL SVD reaches it
(+0.109) but the LOCAL iterative growth S'@G (power-iteration toward the dominant modes)
cannot (local best +0.021). The pre-registered next fork (R3) is a DIRECTIONAL / predictive
/ successor-context local growth, on the HOPE that directional asymmetry reshapes the
operator spectrum so the paradigmatic signal lands in the DOMINANT (locally-reachable)
modes. This oracle tests EXACTLY that hope — substrate-free, gauge-free — BEFORE any R3
build (the directional analog of the §10 oracles in experiments/62_exp61_oracles.py).

THE READ. For a symmetric-PSD operator S, the Levy-Goldberg embedding W_r = U[:,1:r+1]·Σ^0.5
(drop mode 1 = the common mode). spec(r) = mean_cos(para, W_r) − mean_cos(random, W_r),
means-based (NO glob double-subtraction — the dc642a0 fix). The DECISIVE comparison is
DIFFERENTIAL and at LOW r: does the DIRECTIONAL operator put the paradigmatic signal in the
low-r (locally-reachable) modes that the matched SYMMETRIC operator does not?

BATCH-OFFLINE / ANTI-HOMUNCULUS. The directional successor statistic F is ONE offline pass
over the frozen window stream with a fixed γ-discount weight — NOT a learned next-token
predictor and NOT a reconstruction-error term (Reports 017/018 killed that first-order
online class: error_driven/reconstruction both below random Recall@1). The SVD is a
read-only diagnostic flashlight on a FIXED operator; the gates only halt/label, they never
feed back into the operator. ONLINE TD IS BANNED. The local S_dir@G is the (future)
mechanism — this oracle does not build it.

OPERATORS (precommit §1; symmetric-baseline design corrected from the frozen draft — see
the BASELINE NOTE in build_operators(): build_S confounds directionality with sum-vs-ℓ2
normalization and is NOT the source of the +0.109 anchor, so the GATED differential baseline
is the apples-to-apples ℓ2 symmetric Gram; the raw-SPPMI SVD anchor + build_S diagnostic are
reported separately):
  V1  S_dir  = ℓ2rows([ℓ2rows(Mf) ‖ ℓ2rows(Mfᵀ)]) @ (·)ᵀ           [PRIMARY, directional]
  V2  S_fwd  = ℓ2rows(Mf) @ ℓ2rows(Mf)ᵀ                            [forward-only asymmetry probe]
      S_bwd  = ℓ2rows(Mfᵀ) @ ℓ2rows(Mfᵀ)ᵀ                          [backward-only diagnostic]
  V3  S_sym  = ℓ2rows(SPPMI) @ ℓ2rows(SPPMI)ᵀ   [GATED differential baseline, matched build]
      anchor = SVD of raw SPPMI (exp62 ORACLE1)  [calibration: must reproduce +0.109/0.222]
      S_2nd  = build_S(SPPMI) = rownorm(SPPMI@SPPMIᵀ)  [the exact Report-123 operator, diag]
  (HELD IN RESERVE, flashlight-only: Ψ=(I−γT)⁻¹ proper SR — run only if V1/V2 ambiguous.)

Reuses experiment 61's EXACT functions via importlib (no drift), exactly as exp-62 does.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402

# ---- load experiment 61 + 62 functions (module names start with a digit) ----
def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, REPO / "experiments" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

exp61 = _load("exp61", "61_phase3_second_order_growth.py")
exp62 = _load("exp62", "62_exp61_oracles.py")

from energy_memory.phase2.corpus import (  # noqa: E402
    build_vocabulary, encode_texts, load_corpus_splits, make_windows,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402

EPS = 1e-12


# =====================================================================
# THE ONE NEW STATISTIC: γ-discounted DIRECTIONAL within-window co-occurrence
# =====================================================================

def build_directional_cooccurrence(windows, V, special, gamma, W, device="cpu"):
    """Directional cooc F[i,j] += γ^(d−1) for token j at distance d AFTER token i within a
    window (ordered pairs p<q, d=q−p). B=Fᵀ. Specials and i==j skipped. Vectorized over the
    15 (p<q) offset pairs for W=6 — one batched index_add per offset. Batch-offline: a single
    pass over the FIXED window stream, no running estimate, no gradient (anti-homunculus)."""
    F = torch.zeros((V, V), dtype=torch.float64, device=device)
    if not windows:
        return F
    wt = torch.tensor(windows, dtype=torch.long, device=device)          # (n_win, W)
    spec = torch.zeros(V, dtype=torch.bool, device=device)
    for s in special:
        if s < V:
            spec[s] = True
    flatF = F.view(-1)
    for p in range(W):
        for q in range(p + 1, W):
            d = q - p
            i = wt[:, p]
            j = wt[:, q]
            keep = (i != j) & (~spec[i]) & (~spec[j])
            if not bool(keep.any()):
                continue
            ii = i[keep]
            jj = j[keep]
            wgt = torch.full((ii.numel(),), gamma ** (d - 1), dtype=torch.float64, device=device)
            flatF.index_add_(0, ii * V + jj, wgt)
    F.fill_diagonal_(0.0)
    return F.cpu()


def build_directional_sppmi(F, k):
    """Directional SPPMI on F with DIRECTIONAL marginals (row=Σⱼ F[i,j], col=Σᵢ F[i,j]):
    Mf[i,j] = max( log( F[i,j]·|F| / (rowF[i]·colF[j]) ) − log k , 0 ). Zeroed where F==0,
    diagonal zeroed. (Mb = Mfᵀ since B=Fᵀ swaps the marginals — proven in the precommit.)"""
    rowF = F.sum(dim=1, keepdim=True)          # (V,1) outgoing mass
    colF = F.sum(dim=0, keepdim=True)          # (1,V) incoming mass
    total = float(F.sum().item())
    denom = (rowF * colF).clamp(min=EPS)
    ratio = (F * total) / denom
    with torch.no_grad():
        pmi = torch.log(ratio.clamp(min=EPS)) - math.log(max(k, 1e-9))
    M = pmi.clamp(min=0.0)
    M = torch.where(F > 0, M, torch.zeros_like(M))
    M.fill_diagonal_(0.0)
    return M


def directional_density(M):
    V = M.shape[0]
    off = V * V - V
    nnz = int((M > 0).sum().item()) - int((M.diagonal() > 0).sum().item())
    return nnz / max(off, 1)


def l2norm_rows(M):
    return M / M.norm(dim=1, keepdim=True).clamp(min=EPS)


# =====================================================================
# Spectral read: Levy-Goldberg embedding W_r = U[:,1:r+1]·Σ[1:r+1]^0.5 (drop mode 1)
# =====================================================================

def effective_rank(Sg):
    """Participation ratio of the eigenvalue spectrum: (ΣΣ)² / Σ(Σ²)."""
    s = Sg.clamp(min=0)
    return float((s.sum() ** 2) / (s.pow(2).sum().clamp(min=1e-30)))


def spectral_read(S, para, rand, collo, kq, r_grid, full_r, n_boot, boot_seed):
    """SVD a symmetric-PSD operator S; for each r compute means-based spec(r) over the
    drop-mode-1 embedding W_r. Returns per-r dict + diagnostics + the W at every grid r."""
    sym_err = float((S - S.T).abs().max())
    U, Sg, _ = torch.linalg.svd(S)
    Sg = Sg.clamp(min=0)
    neg_eig = float((torch.linalg.eigvalsh(S).clamp(max=0).abs().max())) if S.shape[0] <= 4096 else float("nan")
    # common-mode check: does the top singular vector align with the population mean direction?
    mean_dir = S.mean(dim=0)
    mean_dir = mean_dir / mean_dir.norm().clamp(min=EPS)
    cos_u1_mean = float(torch.dot(U[:, 0], mean_dir).abs())

    rand_cos_cache = {}
    per_r = {}
    W_at = {}
    for r in r_grid:
        rr = max(1, min(r, full_r))
        W = U[:, 1:1 + rr] * Sg[1:1 + rr].sqrt()
        W_at[r] = W
        pc = exp62._cos_real(W, para)
        rc = exp62._cos_real(W, rand)
        rand_cos_cache[r] = rc
        mean, lo, hi = exp62._boot_diff(pc, rc, n_boot=n_boot, seed=boot_seed)
        per_r[r] = {
            "spec": mean, "ci": [lo, hi],
            "para_cos_mean": float(pc.mean()) if pc.numel() else float("nan"),
            "rand_cos_mean": float(rc.mean()) if rc.numel() else float("nan"),
            "eff_rank_keptmodes": effective_rank(Sg[1:1 + rr]),
        }
    # collocational sanity + king/queen at full rank
    Wf = W_at[full_r] if full_r in W_at else (U[:, 1:1 + full_r] * Sg[1:1 + full_r].sqrt())
    rc_full = rand_cos_cache.get(full_r, exp62._cos_real(Wf, rand))
    cc = exp62._cos_real(Wf, collo)
    coll_mean, coll_lo, coll_hi = exp62._boot_diff(cc, rc_full, n_boot=n_boot, seed=boot_seed)
    kq_full = float(exp62._cos_real(Wf, kq).mean()) if kq else float("nan")
    return {
        "sym_err": sym_err, "neg_eig_abs_max": neg_eig,
        "eff_rank_full": effective_rank(Sg), "cos_u1_meandir": cos_u1_mean,
        "top5_sv": [float(x) for x in Sg[:5].tolist()],
        "per_r": per_r,
        "collo_spec_full": {"mean": coll_mean, "ci": [coll_lo, coll_hi]},
        "king_queen_cos_full": kq_full,
    }, W_at


# =====================================================================
# Pair selection (paradigmatic / collocational / random) — real corpus + planted smoke
# =====================================================================

def select_pairs(args, vocab, C, V, special):
    """Returns (para, collo, rand, kq, sha, src) as lists of (a,b) tuples (a<b)."""
    def tid(t):
        return vocab.token_to_id.get(t)

    kq = []
    if tid("king") is not None and tid("queen") is not None and tid("king") != tid("queen"):
        kq = [(min(tid("king"), tid("queen")), max(tid("king"), tid("queen")))]

    if args.corpus_source == "synthetic_planted":
        # SimLex tokens are absent from the planted vocab; build the planted probe directly.
        # para = the planted paradigmatic pair (king/queen: identical contexts, zero cooc);
        # collo = (king, ctx_i) co-occurring; random = matched low-cooc.
        para = list(kq)
        collo = []
        k_id = tid("king")
        for i in range(12):
            ci = tid(f"ctx{i}")
            if k_id is not None and ci is not None and k_id != ci:
                collo.append((min(k_id, ci), max(k_id, ci)))
        rand = exp61.random_matched_pairs(V, max(len(para) + 10, 20), seed=123,
                                          special=special, C=C, max_cooc=args.paradigmatic_max_cooc)
        return para, collo, rand, kq, "PLANTED", "synthetic_planted"

    pairs, sha, src = exp61.load_simlex_pairs(args.simlex_min_sim)
    if pairs is None:
        print("SimLex unavailable:", src, file=sys.stderr)
        raise SystemExit(1)
    para, collo = [], []
    for a_tok, b_tok, _sim in pairs:
        ai, bi = tid(a_tok), tid(b_tok)
        if ai is None or bi is None or ai == bi or ai in special or bi in special:
            continue
        a, b = min(ai, bi), max(ai, bi)
        (para if float(C[a, b]) <= args.paradigmatic_max_cooc else collo).append((a, b))
    rand = exp61.random_matched_pairs(V, max(len(para), 20), seed=123, special=special,
                                      C=C, max_cooc=args.paradigmatic_max_cooc)
    return para, collo, rand, kq, sha, src


# =====================================================================
# Operator construction
# =====================================================================

def build_operators(F, sppmi_sym, C, uni_sym, total_sym, k_sym):
    """BASELINE NOTE (correction to the frozen draft, applied BEFORE the run, documented):
    the frozen pre-commit named the gated differential baseline `S_sym = build_S = rownorm(
    SPPMI@SPPMIᵀ)`. That object differs from S_dir in TWO ways (symmetric-vs-directional AND
    sum-rownorm-vs-ℓ2), confounding the directional differential; and it is NOT the source of
    the +0.1092 anchor (that is exp62 ORACLE1 = SVD of RAW sppmi). So we use THREE labeled
    symmetric references: (a) S_sym = ℓ2(sppmi)@ℓ2(sppmi)ᵀ — the apples-to-apples GATED
    differential baseline (same ℓ2-PMI-signature→Gram construction as S_dir, directionality
    the ONLY difference); (b) the raw-SPPMI SVD ANCHOR (reproduces +0.109/0.222, INVALID
    check); (c) S_2nd = build_S(sppmi) — the exact Report-123 operator, reported as a
    diagnostic. This sharpens the gate; it does not loosen it."""
    Mf = build_directional_sppmi(F, build_operators._k_dir)
    Lf = l2norm_rows(Mf)
    Lb = l2norm_rows(Mf.T.contiguous())
    Sig = l2norm_rows(torch.cat([Lf, Lb], dim=1))
    ops = {
        "S_dir": Sig @ Sig.T,                                   # V1 PRIMARY (directional)
        "S_fwd": Lf @ Lf.T,                                     # V2 forward-only
        "S_bwd": Lb @ Lb.T,                                     # backward-only (diagnostic)
        "S_sym": l2norm_rows(sppmi_sym) @ l2norm_rows(sppmi_sym).T,   # V3 GATED differential baseline
        "S_2nd": exp61.build_S(sppmi_sym),                      # exact Report-123 operator (diag)
    }
    arm_cos = float((Lf * Lb).sum(dim=1).mean())               # per-token fwd/bwd signature overlap
    return ops, Mf, arm_cos


def raw_sppmi_svd_anchor(sppmi_sym, para, rand, collo, kq, svd_rank, n_boot, boot_seed):
    """exp62 ORACLE1 reproduction: SVD of RAW sppmi (NO mode-1 removal), W=U·Σ^0.5. This is
    the +0.1092 / king-queen 0.222 calibration anchor — NOT a gated operator."""
    r = min(svd_rank, sppmi_sym.shape[0])
    U, Sg, _ = torch.linalg.svd(sppmi_sym)
    W = U[:, :r] * Sg[:r].clamp(min=0).sqrt()
    pc = exp62._cos_real(W, para); rc = exp62._cos_real(W, rand)
    cc = exp62._cos_real(W, collo)
    s_mean, s_lo, s_hi = exp62._boot_diff(pc, rc, n_boot=n_boot, seed=boot_seed)
    c_mean, c_lo, c_hi = exp62._boot_diff(cc, rc, n_boot=n_boot, seed=boot_seed)
    return {
        "specificity_para_minus_random": {"mean": s_mean, "ci": [s_lo, s_hi]},
        "collocational_minus_random": {"mean": c_mean, "ci": [c_lo, c_hi]},
        "king_queen_cos": float(exp62._cos_real(W, kq).mean()) if kq else float("nan"),
        "para_cos_mean": float(pc.mean()) if pc.numel() else float("nan"),
        "rand_cos_mean": float(rc.mean()) if rc.numel() else float("nan"),
    }


# =====================================================================

def local_iteration_read(D, V, ops, para, rand, para_t, C, epochs, alpha0, alpha_decay,
                         eta_grid, n_boot, boot_seed):
    """FAITHFUL reachability read: run the ACTUAL local dynamic (exp61.grow_G, the Report-123
    S'@G power-iteration with optional H_anti) on row_center(operator) and read FHRR
    para-vs-random specificity — directly comparable to Report-123's +0.021 collapse-free
    best. Unlike the idealized SVD-rank projection, this is what a LOCAL growth actually
    reaches; it is the un-shortcut test (the user's 'do it right' bar). Same growth schedule,
    same G0, same η_sep grid for every operator → the directional−symmetric difference is the
    clean signal. α_anti=1 so H_anti is active for η_sep>0 (else byte-identical to pull-only)."""
    sub = TorchFHRR(dim=D, seed=0, device="cpu", alpha_anti=1.0)
    G0 = sub.random_vectors(V)
    deff0 = exp61.d_eff(sub, G0)
    cooc_x = torch.log1p(exp61.cooc_counts_for_pairs(C, para_t)) if para_t.numel() else None
    res = {}
    # S_2nd = build_S(sppmi) = the EXACT Report-123 B' operator -> calibrates this faithful
    # read against Report-123's +0.021 collapse-free best (the local analog of the SVD anchor).
    for name in ("S_dir", "S_fwd", "S_sym", "S_2nd"):
        M = exp61.row_center(ops[name])
        cells = []
        best = None
        for eta in eta_grid:
            G = exp61.grow_G(sub, G0, M, epochs, alpha0, alpha_decay, "cpu", eta_sep=eta)
            fpc = exp62._fcos(G, para); frc = exp62._fcos(G, rand)
            mean, lo, hi = exp62._boot_diff(fpc, frc, n_boot=n_boot, seed=boot_seed)
            deff_ratio = exp61.d_eff(sub, G) / max(deff0, 1e-9)
            if cooc_x is not None:
                drift = exp62._fcos(G, para) - exp62._fcos(G0, para)
                corr = exp61.corr_bootstrap_ci(cooc_x, drift, n_boot=n_boot, seed=boot_seed)
            else:
                corr = (float("nan"),) * 3
            cell = {"eta_sep": eta, "spec": mean, "ci": [lo, hi], "deff_ratio": deff_ratio,
                    "corr_hi": corr[2], "collapse_free": bool(deff_ratio >= 0.5)}
            cells.append(cell)
            # "best" = highest spec among collapse-free cells (d_eff_ratio >= 0.5)
            if cell["collapse_free"] and (best is None or mean > best["spec"]):
                best = cell
        res[name] = {"cells": cells, "best_collapse_free": best}
        b = best if best else {"spec": float("nan"), "deff_ratio": float("nan"), "corr_hi": float("nan")}
        print(f"[local-iter {name}] best collapse-free spec={b['spec']:+.4f} "
              f"deff_ratio={b['deff_ratio']:.2f} corr_hi={b['corr_hi']:+.3f}",
              file=sys.stderr, flush=True)
    # the faithful headline: directional best vs symmetric best (comparable to Report-123 +0.021)
    d = res["S_dir"]["best_collapse_free"]; s = res["S_sym"]["best_collapse_free"]
    res["_faithful_headline"] = {
        "dir_best": d["spec"] if d else float("nan"),
        "sym_best": s["spec"] if s else float("nan"),
        "dir_minus_sym": (d["spec"] - s["spec"]) if (d and s) else float("nan"),
        "report123_anchor": 0.021,
        "note": "best collapse-free FHRR para-vs-random specificity under the ACTUAL local "
                "grow_G dynamic; comparable to Report-123's +0.021 symmetric best.",
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source", default="wikitext", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--D", type=int, default=4096)
    ap.add_argument("--W", type=int, default=6)                       # FROZEN
    ap.add_argument("--gamma", type=float, default=0.9)               # FROZEN
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")   # FROZEN
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2, dest="paradigmatic_max_cooc")  # FROZEN
    ap.add_argument("--simlex-min-sim", type=float, default=5.0, dest="simlex_min_sim")            # FROZEN
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--r-lo", type=int, default=5, dest="r_lo")       # FROZEN
    ap.add_argument("--pass-frac", type=float, default=0.50, dest="pass_frac")   # FROZEN f
    ap.add_argument("--margin", type=float, default=0.04)            # FROZEN
    ap.add_argument("--corr-gate", type=float, default=0.15, dest="corr_gate")  # FROZEN
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--planted-seed", type=int, default=0, dest="planted_seed")
    ap.add_argument("--device", default="cpu")
    # faithful local-iteration read (the actual grow_G dynamic; Report-123 schedule)
    ap.add_argument("--local-iter", action="store_true", dest="local_iter",
                    help="run the faithful grow_G local-iteration read (directional vs symmetric)")
    ap.add_argument("--li-epochs", type=int, default=20, dest="li_epochs")
    ap.add_argument("--li-alpha0", type=float, default=0.3, dest="li_alpha0")
    ap.add_argument("--li-alpha-decay", type=float, default=0.9, dest="li_alpha_decay")
    ap.add_argument("--li-eta-grid", default="0,0.05,0.1,0.2", dest="li_eta_grid")
    args = ap.parse_args()

    # ---- corpus + vocab + windows ----
    splits = exp61.load_corpus(args.corpus_source, args, args.planted_seed)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    special = {vocab.unk_id, vocab.mask_id}
    V = len(vocab.id_to_token)
    train_ids = encode_texts(splits["train"], vocab)
    windows = make_windows(train_ids, args.W)
    print(f"V={V} windows={len(windows)} gamma={args.gamma} W={args.W}", file=sys.stderr, flush=True)

    # ---- symmetric presence co-occurrence (for S_sym/anchor/S_2nd + pair-cooc) ----
    C, uni_sym, total_sym = exp61.build_cooccurrence(windows, V, special, device=args.device)
    k_sym, dens_sym, _ = exp61.pick_k_by_density(C, uni_sym, total_sym, 0.3, 0.5)
    if k_sym is None:
        k_sym = 1
        dens_sym = exp61.sppmi_density(exp61.build_sppmi(C, uni_sym, total_sym, k_sym))
    sppmi_sym = exp61.build_sppmi(C, uni_sym, total_sym, k_sym)

    # ---- directional co-occurrence + k pinned on the symmetrized F+B ----
    F = build_directional_cooccurrence(windows, V, special, args.gamma, args.W, device=args.device)
    FB = F + F.T
    uni_fb = FB.sum(dim=1)
    total_fb = float(uni_fb.sum().item())
    k_dir, dens_fb, _ = exp61.pick_k_by_density(FB, uni_fb, total_fb, 0.3, 0.5)
    if k_dir is None:
        k_dir = 1
    build_operators._k_dir = k_dir
    Mf_dens = directional_density(build_directional_sppmi(F, k_dir))
    print(f"k_sym={k_sym} dens_sym={dens_sym:.3f} | k_dir={k_dir} dens_FB={dens_fb:.3f} "
          f"Mf_dens={Mf_dens:.3f}", file=sys.stderr, flush=True)

    # ---- pairs ----
    para, collo, rand, kq, sha, src = select_pairs(args, vocab, C, V, special)
    para_t = torch.tensor(para, dtype=torch.long) if para else torch.zeros((0, 2), dtype=torch.long)
    print(f"n_para={len(para)} n_collo={len(collo)} n_rand={len(rand)} kq={bool(kq)} "
          f"src={src}", file=sys.stderr, flush=True)

    # ---- build operators + spectral read ----
    ops, Mf, arm_cos = build_operators(F, sppmi_sym, C, uni_sym, total_sym, k_sym)
    full_r = min(args.svd_rank, V - 1)
    r_grid = sorted({2, 3, 5, 10, 20, full_r, args.r_lo})
    reads = {}
    W_caches = {}
    for name, S in ops.items():
        reads[name], W_caches[name] = spectral_read(
            S.double(), para, rand, collo, kq, r_grid, full_r, args.n_boot, args.boot_seed)
        rr = reads[name]["per_r"]
        print(f"[{name}] specs " + " ".join(
            f"r{r}={rr[r]['spec']:+.3f}" for r in r_grid), file=sys.stderr, flush=True)

    anchor = raw_sppmi_svd_anchor(sppmi_sym, para, rand, collo, kq, args.svd_rank,
                                  args.n_boot, args.boot_seed)
    print(f"[anchor raw-SPPMI-SVD] spec={anchor['specificity_para_minus_random']['mean']:+.4f} "
          f"kq={anchor['king_queen_cos']:.3f}", file=sys.stderr, flush=True)

    # ---- corr(log cooc, drift) gate on S_dir @ R_lo (the SAME object the headline reads) ----
    def corr_gate_for(name):
        if para_t.numel() == 0:
            return (float("nan"), float("nan"), float("nan"))
        W_rlo = W_caches[name][args.r_lo]
        y = exp62._cos_real(W_rlo, para)
        cooc = exp61.cooc_counts_for_pairs(C, para_t)
        x = torch.log1p(cooc)
        return exp61.corr_bootstrap_ci(x, y, n_boot=args.n_boot, seed=args.boot_seed)
    corr_dir = corr_gate_for("S_dir")

    # ---- FHRR single-shot port arm (substrate-damage isolation; row_center(S_dir)) ----
    sub = TorchFHRR(dim=args.D, seed=0, device="cpu")
    G0 = sub.random_vectors(V)
    M = exp61.row_center(ops["S_dir"]).to(dtype=G0.real.dtype)
    centroid = (M @ G0.real) + 1j * (M @ G0.imag)
    Gp = sub.normalize(centroid).cpu()
    fpc = exp62._fcos(Gp, para); frc = exp62._fcos(Gp, rand)
    fp_mean, fp_lo, fp_hi = exp62._boot_diff(fpc, frc, n_boot=args.n_boot, seed=args.boot_seed)
    # port corr on port_drift = fcos(Gp) − fcos(G0)
    if para_t.numel():
        port_drift = exp62._fcos(Gp, para) - exp62._fcos(G0, para)
        corr_port = exp61.corr_bootstrap_ci(torch.log1p(exp61.cooc_counts_for_pairs(C, para_t)),
                                            port_drift, n_boot=args.n_boot, seed=args.boot_seed)
    else:
        corr_port = (float("nan"),) * 3
    print(f"[FHRR-port S_dir] spec={fp_mean:+.4f} CI[{fp_lo:+.3f},{fp_hi:+.3f}]",
          file=sys.stderr, flush=True)

    # ---- FAITHFUL local-iteration read (the actual grow_G dynamic; cmp Report-123 +0.021) ----
    local_iter = None
    if args.local_iter:
        eta_grid = [float(x) for x in args.li_eta_grid.split(",") if x != ""]
        local_iter = local_iteration_read(
            args.D, V, ops, para, rand, para_t, C, args.li_epochs, args.li_alpha0,
            args.li_alpha_decay, eta_grid, args.n_boot, args.boot_seed)

    # ---- GATE EVALUATION (frozen §2) ----
    dir_r = reads["S_dir"]["per_r"]
    sym_r = reads["S_sym"]["per_r"]
    spec5 = dir_r[args.r_lo]["spec"]; spec5_lo = dir_r[args.r_lo]["ci"][0]
    spec_full = dir_r[full_r]["spec"]
    spec_sym5 = sym_r[args.r_lo]["spec"]
    c1 = bool(spec5_lo == spec5_lo and spec5_lo > 0.0)
    c2 = bool(spec_full > 0 and spec5 >= args.pass_frac * spec_full)
    c3 = bool((spec5 - spec_sym5) >= args.margin)
    c4 = bool(corr_dir[2] == corr_dir[2] and corr_dir[2] < args.corr_gate)
    directional_pass = bool(c1 and c2 and c3 and c4)

    # ---- INVALID vs NULL bright line (frozen §6) ----
    anchor_spec = anchor["specificity_para_minus_random"]["mean"]
    a_lo, a_hi = anchor["specificity_para_minus_random"]["ci"]
    hw = (a_hi - a_lo) / 2 if (a_hi == a_hi) else 0.0
    calib_spec_ok = bool((0.082 - hw) <= anchor_spec <= (0.137 + hw))
    calib_kq_ok = bool(abs(anchor["king_queen_cos"] - 0.222) <= 0.03)
    dens_ratio = max(Mf_dens, dens_sym) / max(min(Mf_dens, dens_sym), 1e-9)
    dens_ok = bool(dens_ratio <= 1.5)
    collo_ok = all(reads[n]["collo_spec_full"]["ci"][0] > 0 for n in ("S_dir", "S_fwd", "S_sym"))
    n_para_ok = bool(len(para) >= 30 or args.corpus_source == "synthetic_planted")
    invalid_reasons = []
    if not calib_spec_ok:
        invalid_reasons.append(f"anchor spec {anchor_spec:+.4f} off the +0.109 CI[0.082,0.137]±hw")
    if not calib_kq_ok:
        invalid_reasons.append(f"anchor king/queen {anchor['king_queen_cos']:.3f} != 0.222±0.03")
    if not dens_ok:
        invalid_reasons.append(f"density mismatch dir/sym ratio {dens_ratio:.2f} > 1.5")
    if not collo_ok:
        invalid_reasons.append("collocational sanity floor fails (scale verdict)")
    if not n_para_ok:
        invalid_reasons.append(f"n_para={len(para)} < 30 (apply frozen fallback)")
    is_valid = (len(invalid_reasons) == 0)

    if not is_valid:
        verdict = "INVALID — " + "; ".join(invalid_reasons) + " -> fix the build / apply fallback, re-run (NOT a true NULL)."
    elif directional_pass:
        verdict = ("PASS — directionality moves the paradigmatic signal into the locally-reachable "
                   "DOMINANT modes (spec_dir(R_lo) real, >=50% of full, beats symmetric by >=margin, "
                   "not collocational). -> green-light R3 grounding + pre-commit (NOT a direct build; §9 fence).")
    else:
        fail = [n for n, c in [("c1 spec5>0", c1), ("c2 >=0.5*full", c2),
                               ("c3 dir-sym>=margin", c3), ("c4 corr<0.15", c4)] if not c]
        verdict = ("NULL — the directional operator hides the paradigmatic signal in the SAME "
                   "subdominant modes as symmetric (failed: " + ", ".join(fail) + "). The local-vs-global "
                   "bound holds for flat codes regardless of directional asymmetry -> escalate to the "
                   "latent/hierarchical layer (Option B; NOT more S'@G/gamma/window knobs).")

    out = {
        "experiment": "63_directional_successor_oracle (DRILL-DOWN feasibility oracle, NOT graduation)",
        "precommit": "notes/emergent-codebook/phase-3-r3-directional-oracle-precommit.md",
        "config": {
            "corpus_source": args.corpus_source, "V": V, "windows": len(windows),
            "gamma": args.gamma, "W": args.W, "max_vocab": args.max_vocab,
            "paradigmatic_max_cooc": args.paradigmatic_max_cooc, "simlex_min_sim": args.simlex_min_sim,
            "k_sym": k_sym, "dens_sym": dens_sym, "k_dir": k_dir, "Mf_dens": Mf_dens,
            "n_para": len(para), "n_collo": len(collo), "n_rand": len(rand),
            "D": args.D, "svd_rank": args.svd_rank, "r_lo": args.r_lo, "full_r": full_r,
            "pass_frac": args.pass_frac, "margin": args.margin, "corr_gate": args.corr_gate,
            "simlex": f"{src}; sha256={sha[:16] if sha and sha != 'PLANTED' else sha}",
            "fwd_bwd_signature_overlap_mean": arm_cos,
        },
        "r_grid": r_grid,
        "operators": reads,
        "anchor_raw_sppmi_svd": anchor,
        "corr_logcooc_drift_S_dir_at_Rlo": {"point": corr_dir[0], "ci": [corr_dir[1], corr_dir[2]]},
        "fhrr_port_S_dir": {
            "spec_para_minus_random": {"mean": fp_mean, "ci": [fp_lo, fp_hi]},
            "corr_logcooc_portdrift": {"point": corr_port[0], "ci": [corr_port[1], corr_port[2]]},
        },
        "local_iteration_read": local_iter,
        "GATE": {
            "spec_dir_Rlo": spec5, "spec_dir_Rlo_ci_lo": spec5_lo, "spec_dir_full": spec_full,
            "spec_sym_Rlo": spec_sym5, "spec_dir_minus_sym_Rlo": spec5 - spec_sym5,
            "corr_ci_hi": corr_dir[2],
            "c1_spec5_real": c1, "c2_half_of_full": c2, "c3_beats_sym_by_margin": c3,
            "c4_not_collocational": c4, "directional_pass": directional_pass,
        },
        "validity": {"is_valid": is_valid, "invalid_reasons": invalid_reasons,
                     "calib_spec_ok": calib_spec_ok, "calib_kq_ok": calib_kq_ok,
                     "dens_ratio": dens_ratio, "collo_floor_ok": collo_ok, "n_para_ok": n_para_ok},
        "VERDICT": verdict,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
