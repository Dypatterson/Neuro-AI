"""experiments/68 — Nonlinear-competition kill-test (DRILL-DOWN oracle, NOT graduation).

Frozen pre-commit: notes/emergent-codebook/phase-3-nonlinear-competition-kill-test-precommit.md.

The ONE question: does a nonlinearity INSIDE a local recurrent fixed point (NSM rectification / assembly
k-WTA) — the brain-legal LOCAL cousin of Oracle E's global NMF (+0.19) — recover the part-based
paradigmatic structure, or null toward the LINEAR grow_G floor (+0.021)? Grill Q1: nonneg
similarity-matching provably escapes PCA/dominant -> manifold-tiling part-based (Sengupta et al.
NeurIPS 2018), the class NMF used; the open question is paradigmatic ALIGNMENT (the label-shuffle
measures it). 3a CORRECTION (build-time): symmetric-NSM-on-M_trans needs sqrt(M_trans) = a banned
global eig; the genuinely-LOCAL writer is ROW-STREAMING (preserves the 2nd-order Gram, no eig), so the
operator is unified to a 2nd-order similarity (build_S = SPPMI@SPPMIᵀ primary; M_trans secondary), with
the linear floor (grow_G), the NMF ceiling (E), and the NSM/kWTA candidates all factorizing the SAME
operator in-harness. Reads: static para-vs-random on the k-dim nonneg codes (= Oracle E's static_read),
with the within-set LABEL-SHUFFLE as the headline + multiple-comparisons guard. Substrate-free verdict;
FHRR-port deferred (fence). 5 seeds. Reuses exp61/62/63/65 verbatim. Floor (055-058) untouched.

NOTE (this build): READ-2 (faithful H-coupled softmax coactivation, Ideas 2/7) is DEFERRED to a second
increment — experiments/67 already empirically closed its static-cosine reduction (Idea 1 -> the 126
NULL). g2 here = beats the grow_G linear floor by +0.02 (the READ-2 comparison is added in the follow-up).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import statistics as st

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
exp65 = _load("exp65", "65_escape_route_triage.py")

from energy_memory.phase2.corpus import build_vocabulary, encode_texts, make_windows  # noqa: E402
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402

EPS = 1e-12


def l2rows(M):
    return M / M.norm(dim=1, keepdim=True).clamp(min=EPS)


# =====================================================================
# The competitive writers (the only new mechanism code)
# =====================================================================

def nsm_features(X, k, *, iters=200, settle=20, lr=0.05, seed=0, learn=True):
    """ROW-STREAMING nonnegative similarity matching (Pehlevan-Chklovskii local rule), BATCHED over
    the offline replay pass (same local Hebbian/anti-Hebbian form; batch-offline = sleep-phase).
    Inputs = rows of the 2nd-order operator X (V×V); preserves the input Gram X Xᵀ nonnegatively
    (NO global eig). Y settles via the recurrent rectified fixed point Y ← [Xn Wᵀ − Y M_offᵀ]₊; W
    (feedforward Hebbian) and M (lateral anti-Hebbian) update from the batch. Returns Y (V×k) ℓ2-rows."""
    V = X.shape[0]
    g = torch.Generator().manual_seed(seed * 7919 + 1)
    Xn = l2rows(X.clamp(min=0.0).to(torch.float64))
    W = 0.1 * torch.randn((k, V), generator=g, dtype=torch.float64)
    M = torch.zeros((k, k), dtype=torch.float64)
    Y = (Xn @ W.T).clamp(min=0.0)
    for _ in range(iters if learn else 1):
        WX = Xn @ W.T                                   # (V×k) feedforward drive
        Y = WX.clamp(min=0.0)
        Moff = M - torch.diag(torch.diag(M))
        for _ in range(settle):
            Y = (WX - Y @ Moff.T).clamp(min=0.0)        # recurrent rectified fixed point
        if not learn:                                   # frozen-random-W architecture control
            break
        W = l2rows(W + lr * ((Y.T @ Xn) / V - W))       # Hebbian (no error/reward); rows normalized (anti-collapse)
        M = M + lr * ((Y.T @ Y) / V - M)                # anti-Hebbian (lateral inhibition)
        M.fill_diagonal_(0.0)
    return l2rows(Y)


def kwta_features(X, k, *, cap=None, iters=200, lr=0.05, seed=0, learn=True):
    """Assembly competitive learning with a FIXED PRECOMMITTED k-WTA cap (Papadimitriou-Vempala),
    BATCHED over the offline pass. a = relu(Xn Wᵀ); keep the top-`cap` units per row (hard
    rank-threshold applied uniformly — a fixed dynamic, not a metric-read branch); competitive Hebbian
    move of the winning units toward their inputs. Returns Y (V×k) ℓ2-rows."""
    V = X.shape[0]
    cap = cap if cap is not None else max(1, k // 4)
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    Xn = l2rows(X.clamp(min=0.0).to(torch.float64))
    W = l2rows(torch.rand((k, V), generator=g, dtype=torch.float64))
    Y = torch.zeros((V, k), dtype=torch.float64)
    for _ in range(iters if learn else 1):
        a = (Xn @ W.T).clamp(min=0.0)                   # (V×k)
        thr = torch.topk(a, cap, dim=1).values[:, -1:].clamp(min=EPS)
        mask = (a >= thr).to(torch.float64)             # uniform top-cap cap per row
        Y = a * mask
        if not learn:                                   # frozen-random-W architecture control
            break
        # competitive Hebbian: each unit moves toward the mean of inputs it won (batched)
        won = mask.T @ Xn                               # (k×V) sum of winning inputs
        cnt = mask.sum(0).clamp(min=1.0).unsqueeze(1)
        W = l2rows(W + lr * (won / cnt - W))
    return l2rows(Y)


def random_nonneg(V, k, seed):
    g = torch.Generator().manual_seed(seed * 99991 + 3)
    return l2rows(torch.rand((V, k), generator=g, dtype=torch.float64))


# =====================================================================
# Static read: para / rand / collo / label-shuffle specificity on a V×k code matrix
# =====================================================================

def read_specificity(Y, para, rand, collo, shuf, n_boot, seed):
    pc = exp62._cos_real(Y, para); rc = exp62._cos_real(Y, rand)
    cc = exp62._cos_real(Y, collo) if collo else torch.zeros(0)
    sc = exp62._cos_real(Y, shuf) if shuf else torch.zeros(0)
    spec = exp62._boot_diff(pc, rc, n_boot=n_boot, seed=seed)       # para − rand
    kill = exp62._boot_diff(pc, sc, n_boot=n_boot, seed=seed) if sc.numel() else (float("nan"),) * 3
    return {"para_cos": float(pc.mean()) if pc.numel() else float("nan"),
            "rand_cos": float(rc.mean()) if rc.numel() else float("nan"),
            "collo_cos": float(cc.mean()) if cc.numel() else float("nan"),
            "shuf_cos": float(sc.mean()) if sc.numel() else float("nan"),
            "spec_para_minus_rand": list(spec), "kill_para_minus_shuf": list(kill)}


def derange_partners(pairs, seed):
    if len(pairs) < 3:
        return []
    g = torch.Generator().manual_seed(seed * 13 + 5)
    while True:
        perm = torch.randperm(len(pairs), generator=g).tolist()
        if all(i != p for i, p in enumerate(perm)):
            break
    return [(pairs[i][0], pairs[perm[i]][1]) for i in range(len(pairs))]


# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source", default="wikitext", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--D", type=int, default=4096)
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2, dest="paradigmatic_max_cooc")
    ap.add_argument("--simlex-min-sim", type=float, default=5.0, dest="simlex_min_sim")
    ap.add_argument("--k-slots", default="8,16,32", dest="k_slots")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--operators", default="build_S,M_trans")
    ap.add_argument("--eta-grid", default="0,0.05,0.1,0.2", dest="eta_grid")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--alpha0", type=float, default=0.3)
    ap.add_argument("--alpha-decay", type=float, default=0.9, dest="alpha_decay")
    ap.add_argument("--nsm-iters", type=int, default=30, dest="nsm_iters")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--planted-seed", type=int, default=0, dest="planted_seed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    k_slots = [int(x) for x in args.k_slots.split(",") if x != ""]
    eta_grid = [float(x) for x in args.eta_grid.split(",") if x != ""]
    ops_want = [o for o in args.operators.split(",") if o]

    # ---- corpus / vocab / windows / operators (seed-independent) ----
    splits = exp61.load_corpus(args.corpus_source, args, args.planted_seed)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    V = len(vocab.id_to_token)
    special = {vocab.unk_id, vocab.mask_id}
    train_ids = encode_texts(splits["train"], vocab)
    windows = make_windows(train_ids, args.W)
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    k_sym, _, _ = exp61.pick_k_by_density(C, uni, total, 0.3, 0.5)
    if k_sym is None:
        k_sym = 1
    sppmi = exp61.build_sppmi(C, uni, total, k_sym)
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)
    shuf = derange_partners([(a, b) for (a, b) in para], args.boot_seed)
    cooc_x = (torch.log1p(exp61.cooc_counts_for_pairs(C, torch.tensor(para, dtype=torch.long)))
              if para else None)

    operators = {}
    if "build_S" in ops_want:
        operators["build_S"] = exp61.build_S(sppmi)                     # SPPMI 2nd-order (the +0.021 floor op)
    if "M_trans" in ops_want:
        Fs = exp65.build_distance_coocs(windows, V, special, args.W, args.gamma)
        operators["M_trans"] = exp65.build_transition_operator(sum(Fs)) # E's operator (+0.19)

    # ---- calibration anchor (once) ----
    anchor = exp63.raw_sppmi_svd_anchor(sppmi, para, rand, collo, kq, args.svd_rank,
                                        args.n_boot, args.boot_seed)
    a_spec = anchor["specificity_para_minus_random"]["mean"]
    a_lo, a_hi = anchor["specificity_para_minus_random"]["ci"]
    hw = (a_hi - a_lo) / 2 if a_hi == a_hi else 0.0
    calib_ok = bool((0.082 - hw) <= a_spec <= (0.137 + hw)
                    and abs(anchor["king_queen_cos"] - 0.222) <= 0.04)
    print(f"V={V} windows={len(windows)} n_para={len(para)} ops={list(operators)} "
          f"[anchor] spec={a_spec:+.4f} kq={anchor['king_queen_cos']:.3f} calib_ok={calib_ok}",
          file=sys.stderr, flush=True)

    def rd(Y, seed):
        return read_specificity(Y, para, rand, collo, shuf, args.n_boot, seed)

    results = {}
    for opname, M in operators.items():
        per_seed = {"linear_floor": [], "E_nmf": {k: [] for k in k_slots},
                    "NSM_3a": {k: [] for k in k_slots}, "kWTA_3a": {k: [] for k in k_slots},
                    "NSM_nolearn": {k: [] for k in k_slots}, "kWTA_nolearn": {k: [] for k in k_slots},
                    "rand_nonneg": {k: [] for k in k_slots}}
        for seed in range(args.seeds):
            # LINEAR FLOOR — grow_G FHRR read (reuse exp65.faithful_read machinery)
            sub = TorchFHRR(dim=args.D, seed=seed, device=args.device, alpha_anti=1.0)
            G0 = sub.random_vectors(V); deff0 = exp61.d_eff(sub, G0)
            fr = exp65.faithful_read(sub, G0, deff0, M, para, rand, cooc_x, eta_grid,
                                     args.epochs, args.alpha0, args.alpha_decay, args.n_boot, args.boot_seed)
            per_seed["linear_floor"].append(fr["best"]["spec"] if fr["best"] else float("nan"))
            # NONNEG STATIC reads at each k
            for k in k_slots:
                Wn = exp65.nmf_slots(M, k, seed=seed)
                per_seed["E_nmf"][k].append(rd(Wn, seed))
                per_seed["NSM_3a"][k].append(rd(nsm_features(M, k, iters=args.nsm_iters, seed=seed), seed))
                per_seed["kWTA_3a"][k].append(rd(kwta_features(M, k, iters=args.nsm_iters, seed=seed), seed))
                per_seed["NSM_nolearn"][k].append(rd(nsm_features(M, k, iters=args.nsm_iters, seed=seed, learn=False), seed))
                per_seed["kWTA_nolearn"][k].append(rd(kwta_features(M, k, iters=args.nsm_iters, seed=seed, learn=False), seed))
                per_seed["rand_nonneg"][k].append(rd(random_nonneg(V, k, seed), seed))
            print(f"[{opname} seed {seed}] linear_floor={per_seed['linear_floor'][-1]:+.4f} "
                  f"E(k32)spec={per_seed['E_nmf'][k_slots[-1]][-1]['spec_para_minus_rand'][0]:+.4f} "
                  f"NSM(k32)spec={per_seed['NSM_3a'][k_slots[-1]][-1]['spec_para_minus_rand'][0]:+.4f} "
                  f"kWTA(k32)spec={per_seed['kWTA_3a'][k_slots[-1]][-1]['spec_para_minus_rand'][0]:+.4f}",
                  file=sys.stderr, flush=True)

        # ---- aggregate + gate per writer (best over k; ≥4/5-seed label-shuffle) ----
        lin_floor = st.mean([x for x in per_seed["linear_floor"] if x == x])
        e_ceiling = max(st.mean([c["spec_para_minus_rand"][0] for c in per_seed["E_nmf"][k]])
                        for k in k_slots)

        def writer_verdict(name):
            ctrl_name = name.replace("_3a", "_nolearn")              # architecture-matched control
            cells = {}
            for k in k_slots:
                rows = per_seed[name][k]
                ctrl = per_seed[ctrl_name][k]
                rn = per_seed["rand_nonneg"][k]
                spec_mean = st.mean([c["spec_para_minus_rand"][0] for c in rows])
                spec_lo = st.mean([c["spec_para_minus_rand"][1] for c in rows])
                kill_lo = [c["kill_para_minus_shuf"][1] for c in rows]
                ctrl_spec = st.mean([c["spec_para_minus_rand"][0] for c in ctrl])
                rn_spec = st.mean([c["spec_para_minus_rand"][0] for c in rn])
                g1 = bool(spec_lo > 0.05)
                g2 = bool(spec_mean - lin_floor >= 0.02)
                g4 = bool(spec_mean - ctrl_spec >= 0.02)             # LEARNING beats frozen-random arch (anti-inflation)
                bkill = sum(1 for x in kill_lo if x == x and x > 0)
                cells[k] = {"spec_mean": spec_mean, "spec_ci_lo_mean": spec_lo,
                            "kill_lo_per_seed": [round(x, 4) for x in kill_lo],
                            "nolearn_ctrl_spec": ctrl_spec, "rand_nonneg_spec": rn_spec,
                            "recover_frac_of_E": spec_mean / e_ceiling if e_ceiling > 1e-9 else float("nan"),
                            "g1": g1, "g2": g2, "g4_beats_nolearn": g4, "B_KILL_seeds": f"{bkill}/{len(rows)}",
                            "PASS": bool(g1 and g2 and g4 and bkill >= 4)}
            best_k = max(k_slots, key=lambda k: cells[k]["spec_mean"])
            return {"cells": cells, "best_k": best_k, "PASS": any(cells[k]["PASS"] for k in k_slots)}

        results[opname] = {
            "linear_floor": lin_floor, "E_ceiling_bestk": e_ceiling,
            "NSM_3a": writer_verdict("NSM_3a"), "kWTA_3a": writer_verdict("kWTA_3a"),
            "_per_seed": {kk: per_seed[kk] for kk in
                          ("E_nmf", "NSM_3a", "kWTA_3a", "NSM_nolearn", "kWTA_nolearn", "rand_nonneg")},
        }
        for w in ("NSM_3a", "kWTA_3a"):
            bk = results[opname][w]["best_k"]
            c = results[opname][w]["cells"][bk]
            print(f"[{opname} {w}] best_k={bk} spec={c['spec_mean']:+.4f} (CIlo {c['spec_ci_lo_mean']:+.4f}) "
                  f"recoverE={c['recover_frac_of_E']:.2f} floor={lin_floor:+.4f} ceil={e_ceiling:+.4f} "
                  f"B_KILL={c['B_KILL_seeds']} PASS={results[opname][w]['PASS']}", file=sys.stderr, flush=True)

    any_pass = any(results[o][w]["PASS"] for o in results for w in ("NSM_3a", "kWTA_3a"))
    verdict = ("INVALID — calibration anchor missed +0.109/0.222." if not calib_ok else
               ("PASS — a LOCAL nonlinear-competitive writer cleared the gate (g1∧g2∧g4∧label-shuffle≥4/5) "
                "→ hard local competition breaks the linear bound; open+card the NSM/sparse-Hopfield "
                "primaries before any build (Phase-5 fence remains)." if any_pass else
                "NULL — no local competitive writer cleared the label-shuffle gate → the linear bound holds "
                "even under read-side competition; Oracle E's positive was carried by global optimization "
                "(same warning as TEM). The k-WTA/NSM/sparse family is dead route-invariantly."))
    out = {"experiment": "68_nonlinear_competition_kill_test (DRILL-DOWN, NOT graduation; READ-2-faithful + 3b deferred)",
           "precommit": "notes/emergent-codebook/phase-3-nonlinear-competition-kill-test-precommit.md",
           "config": {"D": args.D, "V": V, "max_vocab": args.max_vocab, "k_slots": k_slots,
                      "operators": list(operators), "seeds": args.seeds, "nsm_iters": args.nsm_iters,
                      "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
           "anchor": {"spec": a_spec, "kq": anchor["king_queen_cos"], "calib_ok": calib_ok},
           "results": results, "VERDICT": verdict}
    print(json.dumps(out, indent=2, default=float))
    print(f"[VERDICT] {verdict[:110]}", file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
