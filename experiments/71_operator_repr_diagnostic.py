"""experiments/71 — Same-operator / representation diagnostic (settles the exp70 verification's #1 red flag).

The 7-agent adversarial verification (precommit §7) NARROWED exp70: the partition arms ran on
L1 = l2rows(sppmi) (1st-order profiles) while the grow_G floor + NMF ceiling ran on the 2nd-order
build_S = rownorm(sppmi@sppmi^T) — a CROSS-OPERATOR comparison violating the precommit's same-operator
invariant — and offline k-means B-KILL came out NEGATIVE (contradicting 127's "k-means reproduces k-WTA").

This CHEAP, OFFLINE-ONLY diagnostic settles those two red flags before any heavier control:
  - run offline k-WTA, k-means (single-init), k-means (MULTI-RESTART, lowest-inertia), and NMF on BOTH
    representations (L1 = 1st-order l2rows(sppmi); L2 = 2nd-order l2rows(build_S)),
  - read the hubness-immune B-KILL for each, + per-pair BREADTH (how many of 40 pairs >0) + drop-top-10.
Questions settled: (a) on build_S does k-means ~= k-WTA (reproduce 127, validate the harness)? (b) is the
k-means anomaly specific to the 1st-order representation? (c) is the k-WTA B-KILL BROAD or few-pair?
Defers (heavier, to follow): online-on-build_S, n=10 multi-seed, single-pass. Reuses exp70 verbatim.
"""
from __future__ import annotations

import argparse, importlib.util, json, pathlib, statistics as st, sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch  # noqa: E402


def _load(m, f):
    spec = importlib.util.spec_from_file_location(m, REPO / "experiments" / f)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

exp61 = _load("exp61", "61_phase3_second_order_growth.py")
exp62 = _load("exp62", "62_exp61_oracles.py")
exp63 = _load("exp63", "63_directional_successor_oracle.py")
exp65 = _load("exp65", "65_escape_route_triage.py")
exp68 = _load("exp68", "68_nonlinear_competition_kill_test.py")
exp70 = _load("exp70", "70_wikitext_partition_headtohead.py")

from energy_memory.phase2.corpus import build_vocabulary, encode_texts, make_windows  # noqa: E402
EPS = 1e-12; F64 = torch.float64
l2rows = exp70.l2rows


def kmeans_multi(L, k, seed, n_init, iters=50):
    """Multi-restart Lloyd: run n_init inits, keep the LOWEST-INERTIA assignment (closes the
    'k-means underpowered' hypothesis). Returns one-hot l2-rows codes."""
    V = L.shape[0]
    best_Y, best_inertia = None, float("inf")
    for r in range(n_init):
        g = torch.Generator().manual_seed((seed * 100003 + r * 7919) % (2**31))
        cent = L[torch.randperm(V, generator=g)[:k]].clone()
        assign = torch.zeros(V, dtype=torch.long)
        for _ in range(iters):
            d = torch.cdist(L, cent); assign = d.argmin(dim=1)
            for c in range(k):
                msk = assign == c
                if bool(msk.any()):
                    cent[c] = L[msk].mean(dim=0)
        inertia = float((L - cent[assign]).pow(2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            Y = torch.zeros((V, k), dtype=F64); Y[torch.arange(V), assign] = 1.0
            best_Y = l2rows(Y)
    return best_Y


def breadth(Y, para, shuf, n_para):
    """Per-pair B-KILL breadth: count of pairs with para_cos>shuf_cos, + mean after dropping top-10
    highest-para-cos pairs (127's robustness check: must stay > ~+0.12)."""
    pc = exp62._cos_real(Y, para); sc = exp62._cos_real(Y, shuf)
    per = (pc - sc)
    n_pos = int((per > 0).sum())
    order = torch.argsort(pc, descending=True)
    keep10 = order[10:] if pc.numel() > 10 else order
    drop_top10_mean = float(per[keep10].mean()) if keep10.numel() else float("nan")
    return {"pairs_positive": f"{n_pos}/{n_para}", "mean_bkill": float(per.mean()),
            "drop_top10_mean_bkill": drop_top10_mean}


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
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--offline-epochs", type=int, default=60, dest="offline_epochs")
    ap.add_argument("--kmeans-restarts", type=int, default=10, dest="kmeans_restarts")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--planted-seed", type=int, default=0, dest="planted_seed")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    splits = exp61.load_corpus(args.corpus_source, args, args.planted_seed)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    V = len(vocab.id_to_token); special = {vocab.unk_id, vocab.mask_id}
    windows = make_windows(encode_texts(splits["train"], vocab), args.W)
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    k_sym, dens, _ = exp61.pick_k_by_density(C, uni, total, 0.3, 0.5); k_sym = k_sym or 1
    sppmi = exp61.build_sppmi(C, uni, total, k_sym)
    L1 = l2rows(sppmi)                              # 1st-order profiles (exp70's input)
    build_S = exp61.build_S(sppmi)
    L2 = l2rows(build_S)                            # 2nd-order Gram rows (127/precommit's prescribed input)
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)
    shuf = exp68.derange_partners(para, args.boot_seed)
    anchor = exp63.raw_sppmi_svd_anchor(sppmi, para, rand, collo, kq, args.svd_rank, args.n_boot, args.boot_seed)
    a_spec = anchor["specificity_para_minus_random"]["mean"]
    calib_ok = bool(0.082 <= a_spec <= 0.137 and abs(anchor["king_queen_cos"] - 0.222) <= 0.04)
    print(f"V={V} windows={len(windows)} k_sym={k_sym} n_para={len(para)} "
          f"[anchor] spec={a_spec:+.4f} kq={anchor['king_queen_cos']:.3f} calib_ok={calib_ok}",
          file=sys.stderr, flush=True)

    reps = {"L1_sppmi_1storder": L1, "L2_build_S_2ndorder": L2}
    arms = ("kwta", "kmeans_single", "kmeans_multi", "nmf")
    res = {rn: {a: {"kill": [], "kill_lo": []} for a in arms} for rn in reps}
    brd = {rn: {a: [] for a in ("kwta", "kmeans_multi")} for rn in reps}

    def kill(Y):
        r = exp68.read_specificity(Y, para, rand, collo, shuf, args.n_boot, args.boot_seed)
        return r["kill_para_minus_shuf"]

    for seed in range(args.seeds):
        for rn, L in reps.items():
            ys = {
                "kwta": exp70.kwta_offline(L, args.k, args.cap, args.offline_epochs, seed),
                "kmeans_single": exp70.kmeans_onehot(L, args.k, seed),
                "kmeans_multi": kmeans_multi(L, args.k, seed, args.kmeans_restarts),
                # NMF factorizes the operator matrix (nonneg) — use the underlying nonneg operator
                "nmf": exp65.nmf_slots(sppmi if rn == "L1_sppmi_1storder" else build_S, args.k, seed=seed),
            }
            for a in arms:
                kl = kill(ys[a]); res[rn][a]["kill"].append(kl[0]); res[rn][a]["kill_lo"].append(kl[1])
            for a in ("kwta", "kmeans_multi"):
                brd[rn][a].append(breadth(ys[a], para, shuf, len(para)))
        print(f"  [seed {seed}] "
              f"L1: kwta={res['L1_sppmi_1storder']['kwta']['kill'][-1]:+.3f} "
              f"km1={res['L1_sppmi_1storder']['kmeans_single']['kill'][-1]:+.3f} "
              f"kmX={res['L1_sppmi_1storder']['kmeans_multi']['kill'][-1]:+.3f} | "
              f"L2: kwta={res['L2_build_S_2ndorder']['kwta']['kill'][-1]:+.3f} "
              f"km1={res['L2_build_S_2ndorder']['kmeans_single']['kill'][-1]:+.3f} "
              f"kmX={res['L2_build_S_2ndorder']['kmeans_multi']['kill'][-1]:+.3f}",
              file=sys.stderr, flush=True)

    def mean(xs): xs = [x for x in xs if x == x]; return st.mean(xs) if xs else float("nan")
    summary = {rn: {a: {"kill_mean": mean(res[rn][a]["kill"]), "kill_lo_mean": mean(res[rn][a]["kill_lo"])}
                    for a in arms} for rn in reps}
    # the two decisive checks
    def gap(rn): return summary[rn]["kwta"]["kill_mean"] - summary[rn]["kmeans_multi"]["kill_mean"]
    km_matches_kwta_on_buildS = abs(gap("L2_build_S_2ndorder")) <= 0.05            # 127 replication
    km_anomaly_is_L1_specific = gap("L1_sppmi_1storder") > 0.10 and abs(gap("L2_build_S_2ndorder")) <= 0.05
    out = {
        "experiment": "71_operator_repr_diagnostic (settles exp70-verification red flags #1/#2; offline-only; NOT graduation)",
        "config": {**vars(args), "V": V, "windows": len(windows), "k_sym": k_sym,
                   "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
        "anchor": {"spec": a_spec, "kq": anchor["king_queen_cos"], "calib_ok": calib_ok},
        "metric": "kill = para_cos - within-set-shuffle_cos (hubness-immune B-KILL)",
        "by_representation": summary,
        "breadth": {rn: {a: brd[rn][a] for a in brd[rn]} for rn in brd},
        "kwta_minus_kmeansmulti": {"L1_1storder": gap("L1_sppmi_1storder"), "L2_build_S": gap("L2_build_S_2ndorder")},
        "kmeans_reproduces_kwta_on_buildS_127replication": km_matches_kwta_on_buildS,
        "kmeans_anomaly_is_1storder_representation_specific": km_anomaly_is_L1_specific,
    }
    print(json.dumps(out, indent=2, default=float))
    print(f"\n[DIAG] km~=kwta on build_S (127 repl)={km_matches_kwta_on_buildS}; "
          f"k-means anomaly is 1st-order-specific={km_anomaly_is_L1_specific}; "
          f"L1 kwta-kmX gap={gap('L1_sppmi_1storder'):+.3f}, L2 gap={gap('L2_build_S_2ndorder'):+.3f}",
          file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
