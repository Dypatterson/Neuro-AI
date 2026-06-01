"""Phase-3 escape-route oracle TRIAGE (4 substrate-free DRILL-DOWN oracles, NOT graduation).

Frozen pre-commit: notes/emergent-codebook/phase-3-escape-route-oracle-triage-precommit.md.

After the flat-code growth family was exhausted (Reports 121/123/124 — paradigmatic structure
lives in the SUBDOMINANT modes of the co-occurrence operator, unreachable by local iteration),
two workflows + a completeness critic converged on FOUR distinct structural escape-routes. This
triages all four with cheap substrate-free oracles BEFORE any substrate build (the exp63 pattern):

  B  SFA/SR slowness     : M_trans=(T+Tᵀ)/2, M_SR=(Ψ+Ψᵀ)/2 (Ψ=Σγ^t Tᵗ). Dominant modes = slow/SR
                           low-freq = the substitutability axis. Verdict = grow_G + H_anti (the
                           LOCAL whitening) reaching them — the critic's locality trap.   [LEAD]
  C  eligibility×surprise: M_elig = cooc · novelty (multiplicative freq-suppression, distinct from
                           SPPMI's marginal division). MUST beat plain SPPMI (else = 123). [missing family]
  D  order channel       : M_order = avg over distance-d of the cosine-Gram of position-typed cooc
                           profiles (the position SPECTRUM, not k=±1 = 124).               [Route IV]
  E  TEM slots           : NMF of the symmetrized transition op into k relational slots; paradigmatic
                           = shared slot-distribution. Static read on slot-vectors.        [Route II]

Verdict-bearing read = the FAITHFUL exp61.grow_G dynamic (Report 124 §4 proved the SVD low-rank
spectral read is a CONTRACTION ARTIFACT — retained only as a labeled upper-bound diagnostic).
Every run carries the raw-SPPMI-SVD calibration anchor (+0.1092/kq 0.222 or INVALID). Reuses
exp61/62/63 verbatim via importlib. Floor (055-058) untouched (separate operators, never read at recall).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, REPO / "experiments" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

exp61 = _load("exp61", "61_phase3_second_order_growth.py")
exp62 = _load("exp62", "62_exp61_oracles.py")
exp63 = _load("exp63", "63_directional_successor_oracle.py")

from energy_memory.phase2.corpus import (  # noqa: E402
    build_vocabulary, encode_texts, load_corpus_splits, make_windows,
)
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402

EPS = 1e-12


def l2norm_rows(M):
    return M / M.norm(dim=1, keepdim=True).clamp(min=EPS)


# =====================================================================
# New operators (the only new code; everything else reuses exp61/62/63)
# =====================================================================

def build_distance_coocs(windows, V, special, W, gamma, device="cpu"):
    """Per-distance directional cooc F_d[i,j] = γ^(d−1)·#(i at p, j at p+d) within a window,
    d=1..W−1. Returns a list [F_1, ..., F_{W-1}] (each VxV float64). The position SPECTRUM."""
    Fs = [torch.zeros((V, V), dtype=torch.float64, device=device) for _ in range(W - 1)]
    if not windows:
        return Fs
    wt = torch.tensor(windows, dtype=torch.long, device=device)
    spec = torch.zeros(V, dtype=torch.bool, device=device)
    for s in special:
        if s < V:
            spec[s] = True
    for p in range(W):
        for q in range(p + 1, W):
            d = q - p
            i, j = wt[:, p], wt[:, q]
            keep = (i != j) & (~spec[i]) & (~spec[j])
            if not bool(keep.any()):
                continue
            ii, jj = i[keep], j[keep]
            wgt = torch.full((ii.numel(),), gamma ** (d - 1), dtype=torch.float64, device=device)
            Fs[d - 1].view(-1).index_add_(0, ii * V + jj, wgt)
    return [F.cpu() for F in Fs]


def build_transition_operator(F):
    """M_trans = (T + Tᵀ)/2, T = rownorm(F). Top eigenvectors (post mode-1) = slow/SR modes."""
    T = exp61.rownorm(F)
    return 0.5 * (T + T.T)


def build_SR_operator(F, gamma, K):
    """M_SR = (Ψ + Ψᵀ)/2, Ψ = Σ_{t=0}^{K} γ^t Tᵗ (batch-offline truncated SR; γ-reweight INFLATES
    the slow modes — the critic's subdominant→dominant mechanism)."""
    T = exp61.rownorm(F)
    V = T.shape[0]
    Psi = torch.zeros((V, V), dtype=torch.float64)
    term = torch.eye(V, dtype=torch.float64)
    g = 1.0
    for _ in range(K + 1):
        Psi += g * term
        term = term @ T
        g *= gamma
    return 0.5 * (Psi + Psi.T)


def build_eligibility_operator(C, uni, beta):
    """M_elig = cooc · novelty, novelty[i,j] = 1/(1+min(freq_i,freq_j))^β — MULTIPLICATIVE
    frequency-suppression, distinct from SPPMI's marginal DIVISION (the two-timescale surrogate)."""
    fmin = torch.minimum(uni[:, None], uni[None, :])
    nov = 1.0 / (1.0 + fmin).pow(beta)
    M = C * nov
    M.fill_diagonal_(0.0)
    return M


def build_order_operator(Fs):
    """M_order = mean over distance-d of the cosine-Gram of position-typed cooc rows (the position
    SPECTRUM). Each F_d row = i's distance-d forward-neighbor profile; cluster by shared profiles."""
    V = Fs[0].shape[0]
    M = torch.zeros((V, V), dtype=torch.float64)
    n = 0
    for F in Fs:
        L = l2norm_rows(F)
        M += L @ L.T
        Lb = l2norm_rows(F.T.contiguous())
        M += Lb @ Lb.T
        n += 2
    return M / max(n, 1)


def nmf_slots(M, k, iters=200, seed=0):
    """Nonnegative factorization M ≈ W Hᵀ (M symmetric nonneg), multiplicative updates (Lee-Seung).
    Returns ℓ2-normalized slot-vectors (rows of W, V×k) = each token's structural slot-distribution."""
    Mc = M.clamp(min=0.0)
    V = Mc.shape[0]
    g = torch.Generator().manual_seed(seed)
    Wm = torch.rand((V, k), generator=g, dtype=torch.float64).clamp(min=1e-3)
    Hm = torch.rand((V, k), generator=g, dtype=torch.float64).clamp(min=1e-3)
    for _ in range(iters):
        Wm = Wm * ((Mc @ Hm) / ((Wm @ (Hm.T @ Hm)).clamp(min=EPS)))
        Hm = Hm * ((Mc.T @ Wm) / ((Hm @ (Wm.T @ Wm)).clamp(min=EPS)))
    return l2norm_rows(Wm)


# =====================================================================
# Reads: faithful grow_G (verdict) + static cosine (factor embeddings)
# =====================================================================

def faithful_read(sub, G0, deff0, M, para, rand, cooc_x, eta_grid, epochs, a0, adec, n_boot, seed):
    """Run exp61.grow_G on row_center(M) over η_sep; return best collapse-free cell + all cells."""
    Mc = exp61.row_center(M)
    cells, best = [], None
    for eta in eta_grid:
        G = exp61.grow_G(sub, G0, Mc, epochs, a0, adec, "cpu", eta_sep=eta)
        fpc, frc = exp62._fcos(G, para), exp62._fcos(G, rand)
        mean, lo, hi = exp62._boot_diff(fpc, frc, n_boot=n_boot, seed=seed)
        ratio = exp61.d_eff(sub, G) / max(deff0, 1e-9)
        if cooc_x is not None and para:
            drift = exp62._fcos(G, para) - exp62._fcos(G0, para)
            corr = exp61.corr_bootstrap_ci(cooc_x, drift, n_boot=n_boot, seed=seed)
        else:
            corr = (float("nan"),) * 3
        cell = {"eta": eta, "spec": mean, "ci": [lo, hi], "deff_ratio": ratio, "corr_hi": corr[2],
                "collapse_free": bool(ratio >= 0.5)}
        cells.append(cell)
        if cell["collapse_free"] and (best is None or mean > best["spec"]):
            best = cell
    return {"cells": cells, "best": best}


def static_read(Wemb, para, rand, n_boot, seed):
    """Means-based para-vs-random specificity on real embedding rows (for factor/static operators)."""
    pc, rc = exp62._cos_real(Wemb, para), exp62._cos_real(Wemb, rand)
    mean, lo, hi = exp62._boot_diff(pc, rc, n_boot=n_boot, seed=seed)
    return {"spec": mean, "ci": [lo, hi],
            "para_cos": float(pc.mean()) if pc.numel() else float("nan"),
            "rand_cos": float(rc.mean()) if rc.numel() else float("nan")}


def gate(spec, ci_lo, baseline_spec, corr_hi, deff_ratio, margin=0.04, beat=0.02):
    g1 = bool(ci_lo == ci_lo and ci_lo > margin)
    g2 = bool(spec == spec and (spec - baseline_spec) >= beat)
    g3 = bool(corr_hi == corr_hi and corr_hi < 0.15)
    g4 = bool(deff_ratio != deff_ratio or deff_ratio >= 0.5)  # nan (static) -> pass
    return {"g1_decisively_past_bound": g1, "g2_beats_flat_baseline": g2,
            "g3_not_collocational": g3, "g4_no_collapse": g4,
            "PASS": bool(g1 and g2 and g3 and g4)}


# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="all", choices=["B", "C", "D", "E", "all"])
    ap.add_argument("--corpus-source", default="wikitext", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--D", type=int, default=4096)
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2, dest="paradigmatic_max_cooc")
    ap.add_argument("--simlex-min-sim", type=float, default=5.0, dest="simlex_min_sim")
    ap.add_argument("--K-sr", type=int, default=10, dest="K_sr")
    ap.add_argument("--betas", default="0.5,1.0")
    ap.add_argument("--k-slots", default="8,16,32", dest="k_slots")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--eta-grid", default="0,0.05,0.1,0.2", dest="eta_grid")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--alpha0", type=float, default=0.3)
    ap.add_argument("--alpha-decay", type=float, default=0.9, dest="alpha_decay")
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--planted-seed", type=int, default=0, dest="planted_seed")
    args = ap.parse_args()
    eta_grid = [float(x) for x in args.eta_grid.split(",") if x != ""]
    betas = [float(x) for x in args.betas.split(",") if x != ""]
    k_slots = [int(x) for x in args.k_slots.split(",") if x != ""]

    # ---- corpus + vocab + windows + cooc + SPPMI (shared) ----
    splits = exp61.load_corpus(args.corpus_source, args, args.planted_seed)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    special = {vocab.unk_id, vocab.mask_id}
    V = len(vocab.id_to_token)
    train_ids = encode_texts(splits["train"], vocab)
    windows = make_windows(train_ids, args.W)
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    k_sym, _, _ = exp61.pick_k_by_density(C, uni, total, 0.3, 0.5)
    if k_sym is None:
        k_sym = 1
    sppmi = exp61.build_sppmi(C, uni, total, k_sym)
    M_cooc = exp61.build_S(sppmi)                       # flat-SPPMI baseline operator (known-subdominant)
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)
    para_t = torch.tensor(para, dtype=torch.long) if para else torch.zeros((0, 2), dtype=torch.long)
    cooc_x = torch.log1p(exp61.cooc_counts_for_pairs(C, para_t)) if para_t.numel() else None
    print(f"V={V} windows={len(windows)} n_para={len(para)} k_sym={k_sym} src={src}",
          file=sys.stderr, flush=True)

    # ---- calibration anchor (INVALID if it misses +0.109/0.222) ----
    anchor = exp63.raw_sppmi_svd_anchor(sppmi, para, rand, collo, kq, args.svd_rank,
                                        args.n_boot, args.boot_seed)
    a_spec = anchor["specificity_para_minus_random"]["mean"]
    a_lo, a_hi = anchor["specificity_para_minus_random"]["ci"]
    hw = (a_hi - a_lo) / 2 if a_hi == a_hi else 0.0
    calib_ok = bool((0.082 - hw) <= a_spec <= (0.137 + hw) and
                    abs(anchor["king_queen_cos"] - 0.222) <= 0.04)
    print(f"[anchor] spec={a_spec:+.4f} kq={anchor['king_queen_cos']:.3f} calib_ok={calib_ok}",
          file=sys.stderr, flush=True)

    # ---- FHRR substrate (α_anti=1 for the local-whitening H_anti) ----
    sub = TorchFHRR(dim=args.D, seed=0, device="cpu", alpha_anti=1.0)
    G0 = sub.random_vectors(V)
    deff0 = exp61.d_eff(sub, G0)

    def grow(M):
        return faithful_read(sub, G0, deff0, M, para, rand, cooc_x, eta_grid,
                             args.epochs, args.alpha0, args.alpha_decay, args.n_boot, args.boot_seed)

    # ---- baseline: flat-SPPMI under grow_G (the matched control for g2) ----
    base = grow(M_cooc)
    base_spec = base["best"]["spec"] if base["best"] else float("nan")
    print(f"[baseline flat-SPPMI] best collapse-free spec={base_spec:+.4f}", file=sys.stderr, flush=True)

    results = {}
    Fs = None
    F = None
    want = ["B", "C", "D", "E"] if args.oracle == "all" else [args.oracle]

    if any(o in want for o in ("B", "D", "E")):
        Fs = build_distance_coocs(windows, V, special, args.W, args.gamma)
        F = sum(Fs)

    if "B" in want:
        ops = {"M_trans": build_transition_operator(F), "M_SR": build_SR_operator(F, args.gamma, args.K_sr)}
        oB = {}
        for name, M in ops.items():
            r = grow(M)
            b = r["best"] or {"spec": float("nan"), "corr_hi": float("nan"), "deff_ratio": float("nan"), "ci": [float("nan")] * 2}
            oB[name] = {"grow": r, "gate": gate(b["spec"], b["ci"][0], base_spec, b["corr_hi"], b["deff_ratio"])}
            print(f"[B {name}] spec={b['spec']:+.4f} corr_hi={b['corr_hi']:+.3f} PASS={oB[name]['gate']['PASS']}",
                  file=sys.stderr, flush=True)
        results["B_sfa_sr"] = oB

    if "C" in want:
        oC = {}
        for beta in betas:
            M = build_eligibility_operator(C, uni, beta)
            Ssec = exp61.build_S(M)  # second-order similarity of eligibility-weighted profiles
            r = grow(Ssec)
            b = r["best"] or {"spec": float("nan"), "corr_hi": float("nan"), "deff_ratio": float("nan"), "ci": [float("nan")] * 2}
            oC[f"beta={beta}"] = {"grow": r, "gate": gate(b["spec"], b["ci"][0], base_spec, b["corr_hi"], b["deff_ratio"])}
            print(f"[C beta={beta}] spec={b['spec']:+.4f} corr_hi={b['corr_hi']:+.3f} PASS={oC[f'beta={beta}']['gate']['PASS']}",
                  file=sys.stderr, flush=True)
        results["C_eligibility"] = oC

    if "D" in want:
        M_order = build_order_operator(Fs)
        r = grow(M_order)
        b = r["best"] or {"spec": float("nan"), "corr_hi": float("nan"), "deff_ratio": float("nan"), "ci": [float("nan")] * 2}
        results["D_order"] = {"grow": r, "gate": gate(b["spec"], b["ci"][0], base_spec, b["corr_hi"], b["deff_ratio"])}
        print(f"[D order] spec={b['spec']:+.4f} corr_hi={b['corr_hi']:+.3f} PASS={results['D_order']['gate']['PASS']}",
              file=sys.stderr, flush=True)

    if "E" in want:
        M_trans = build_transition_operator(F)
        oE = {}
        for k in k_slots:
            slots = nmf_slots(M_trans, k)
            sr = static_read(slots, para, rand, args.n_boot, args.boot_seed)
            # corr on static slot cosine vs log cooc
            if para_t.numel():
                y = exp62._cos_real(slots, para)
                corr = exp61.corr_bootstrap_ci(cooc_x, y, n_boot=args.n_boot, seed=args.boot_seed)
            else:
                corr = (float("nan"),) * 3
            kq_cos = float(exp62._cos_real(slots, kq).mean()) if kq else float("nan")
            oE[f"k={k}"] = {"static": sr, "corr_hi": corr[2], "king_queen_cos": kq_cos,
                            "gate": gate(sr["spec"], sr["ci"][0], base_spec, corr[2], float("nan"))}
            print(f"[E k={k}] slot spec={sr['spec']:+.4f} corr_hi={corr[2]:+.3f} kq={kq_cos:+.3f} "
                  f"PASS={oE[f'k={k}']['gate']['PASS']}", file=sys.stderr, flush=True)
        results["E_tem_slots"] = oE

    any_pass = json.dumps(results).count('"PASS": true') > 0
    out = {
        "experiment": "65_escape_route_triage (DRILL-DOWN feasibility oracles, NOT graduation)",
        "precommit": "notes/emergent-codebook/phase-3-escape-route-oracle-triage-precommit.md",
        "config": {"corpus_source": args.corpus_source, "V": V, "windows": len(windows),
                   "n_para": len(para), "gamma": args.gamma, "W": args.W, "k_sym": k_sym,
                   "K_sr": args.K_sr, "D": args.D, "eta_grid": eta_grid,
                   "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
        "anchor": {"spec": a_spec, "ci": [a_lo, a_hi], "king_queen_cos": anchor["king_queen_cos"],
                   "calib_ok": calib_ok},
        "flat_sppmi_baseline_grow_spec": base_spec,
        "oracles": results,
        "VERDICT": ("INVALID — anchor missed +0.109/0.222; fix harness, re-run."
                    if not calib_ok else
                    ("PASS — at least one escape route cleared the gate (see per-oracle PASS); "
                     "greenlight that route's grounding+precommit (NOT a build; FHRR-port + phase fence remain)."
                     if any_pass else
                     "NULL — all four escape routes fail the gate at the substrate-free level → the bound is "
                     "route-invariant for single-projection operators → escalate to a multi-LAYER hierarchy / "
                     "substrate change (re-scope with user). NOT more operator knobs.")),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
