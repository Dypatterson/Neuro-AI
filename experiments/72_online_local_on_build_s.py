"""experiments/72 — The decisive control: online-LOCAL bounded writer on the CORRECT operator (build_S).

The exp70 verification (precommit §7) NARROWED the screen positive: exp70's online arms partitioned the
1st-order l2rows(sppmi), but the precommit/127 operator is the 2nd-order build_S = rownorm(sppmi@sppmi^T)
(the SAME operator as the grow_G ~0 floor + the NMF ceiling). exp71 confirmed on build_S: k-means (+0.19)
~= k-WTA (+0.23) [127 replication], grow_G ~0. THIS experiment re-runs the online arms ON build_S to
settle the one genuinely-new open question:

  Does an ONLINE, bounded-INPUT (decaying Pi, never holds the full operator), LEARNED k-WTA writer reach
  the build_S partition ceiling (~+0.20) AND beat its frozen-random control — ON THE CORRECT 2nd-order
  operator — where the linear local read (grow_G) gets ~0?

  YES -> locality+online is ~free FOR THE PARTITION on the right operator (a buildable local mechanism);
         the bounded≈converged gap + the single-pass control (follow-up) then characterize how local.
  NO  -> the online-local win in exp70 was a 1st-order-representation artifact; on the real operator the
         local writer falls short -> the global pass is load-bearing (replay/TEM re-enter as gap-closers).

Online writer (operator-aware, tractable): per epoch, stream window chunks updating Pi = decay*Pi + Cb,
then compute the 2nd-order operator L = l2rows(build_S(sppmi(Pi))) ONCE for that epoch and take `inner`
competitive steps carrying W. decay<1 = genuinely recency-bounded input; decay=1 = accumulating (offline-
with-warmup). learn=False = frozen-random (no Hebbian) = the 127 incompetent control. RUNG-1, substrate-
free, NOT graduation; 3 seeds (n=10 is the follow-up). Reuses exp61/65/68/70 verbatim. Floor untouched.
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
exp71 = _load("exp71", "71_operator_repr_diagnostic.py")

from energy_memory.phase2.corpus import build_vocabulary, encode_texts, make_windows  # noqa: E402
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402
EPS = 1e-12; F64 = torch.float64
l2rows = exp70.l2rows


def online_build_s(windows, V, special, k, cap, k_sym, chunk, epochs, inner, seed, decay=1.0, learn=True):
    """Online k-WTA on the 2nd-order build_S operator. Per epoch: stream chunks (Pi = decay*Pi + Cb),
    then L = l2rows(build_S(sppmi(Pi))) (the SAME operator class as offline/grow_G), `inner` competitive
    steps carrying W. decay<1 -> recency-bounded Pi (never the full operator). learn=False -> frozen W."""
    W = exp70._init_W(V, k, seed)
    Y = torch.zeros((V, k), dtype=F64)
    Pi = torch.zeros((V, V), dtype=F64)
    n = len(windows)
    for _ep in range(epochs):
        for s in range(0, n, chunk):
            Cb, _u, _t = exp61.build_cooccurrence(windows[s:s + chunk], V, special, device="cpu")
            Pi = decay * Pi + Cb
        uni = Pi.sum(dim=1)
        sppmi = exp61.build_sppmi(Pi, uni, float(uni.sum().item()), k_sym)
        L = l2rows(exp61.build_S(sppmi))                       # the 2nd-order operator from current Pi
        for _ in range(inner):
            Y, W_new = exp70._kwta_step(L, W, cap)
            if learn:
                W = W_new
    return l2rows(Y)


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
    ap.add_argument("--online-epochs", type=int, default=20, dest="online_epochs")
    ap.add_argument("--inner", type=int, default=6, dest="inner")
    ap.add_argument("--online-decay", type=float, default=0.7, dest="online_decay")
    ap.add_argument("--chunk", type=int, default=30000)
    ap.add_argument("--floor-epochs", type=int, default=20, dest="floor_epochs")
    ap.add_argument("--kmeans-restarts", type=int, default=10, dest="kmeans_restarts")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--seeds", type=int, default=10)
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
    build_S = exp61.build_S(sppmi)
    L2 = l2rows(build_S)                                       # the CORRECT partition representation
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)
    shuf = exp68.derange_partners(para, args.boot_seed)
    anchor = exp63.raw_sppmi_svd_anchor(sppmi, para, rand, collo, kq, args.svd_rank, args.n_boot, args.boot_seed)
    a_spec = anchor["specificity_para_minus_random"]["mean"]
    calib_ok = bool(0.082 <= a_spec <= 0.137 and abs(anchor["king_queen_cos"] - 0.222) <= 0.04)
    print(f"V={V} windows={len(windows)} k_sym={k_sym} n_para={len(para)} [anchor] spec={a_spec:+.4f} "
          f"kq={anchor['king_queen_cos']:.3f} calib_ok={calib_ok}", file=sys.stderr, flush=True)

    acc = {n: {"kill": [], "lo": []} for n in
           ("offline_kwta", "kmeans_multi", "nmf", "floor", "online_converged", "online_bounded", "online_frozen")}

    def rec(name, Y, bseed):
        # INDEPENDENT bootstrap seed per model-seed (verification control #3: decouple from a fixed
        # boot_seed so across-seed model variance is visible).
        kl = exp68.read_specificity(Y, para, rand, collo, shuf, args.n_boot, bseed)["kill_para_minus_shuf"]
        acc[name]["kill"].append(kl[0]); acc[name]["lo"].append(kl[1]); return kl

    for seed in range(args.seeds):
        bseed = args.boot_seed + seed * 101
        rec("offline_kwta", exp70.kwta_offline(L2, args.k, args.cap, args.offline_epochs, seed), bseed)
        rec("kmeans_multi", exp71.kmeans_multi(L2, args.k, seed, args.kmeans_restarts), bseed)
        rec("nmf", exp65.nmf_slots(build_S, args.k, seed=seed), bseed)
        sub = TorchFHRR(dim=args.D, seed=seed, device="cpu", alpha_anti=1.0)
        G0 = sub.random_vectors(V); deff0 = exp61.d_eff(sub, G0)
        cooc_x = torch.log1p(exp61.cooc_counts_for_pairs(C, torch.tensor(para, dtype=torch.long))) if para else None
        fr = exp65.faithful_read(sub, G0, deff0, build_S, para, rand, cooc_x, [0.0, 0.1],
                                 args.floor_epochs, 0.3, 0.9, args.n_boot, bseed)
        acc["floor"]["kill"].append(fr["best"]["spec"] if fr["best"] else float("nan")); acc["floor"]["lo"].append(float("nan"))
        rec("online_converged", online_build_s(windows, V, special, args.k, args.cap, k_sym, args.chunk,
                                                args.online_epochs, args.inner, seed, decay=1.0, learn=True), bseed)
        ob = rec("online_bounded", online_build_s(windows, V, special, args.k, args.cap, k_sym, args.chunk,
                                                  args.online_epochs, args.inner, seed, decay=args.online_decay, learn=True), bseed)
        rec("online_frozen", online_build_s(windows, V, special, args.k, args.cap, k_sym, args.chunk,
                                            args.online_epochs, args.inner, seed, decay=args.online_decay, learn=False), bseed)
        print(f"  [seed {seed}] off_kwta={acc['offline_kwta']['kill'][-1]:+.3f} "
              f"kmX={acc['kmeans_multi']['kill'][-1]:+.3f} nmf={acc['nmf']['kill'][-1]:+.3f} "
              f"floor={acc['floor']['kill'][-1]:+.3f} | converged={acc['online_converged']['kill'][-1]:+.3f} "
              f"BOUNDED={ob[0]:+.3f} (lo {ob[1]:+.3f}) frozen={acc['online_frozen']['kill'][-1]:+.3f}",
              file=sys.stderr, flush=True)

    def m(n): xs = [x for x in acc[n]["kill"] if x == x]; return st.mean(xs) if xs else float("nan")

    def tci(vals):  # across-seed bootstrap CI over per-seed point estimates -> (mean, lo, hi)
        t = torch.tensor([v for v in vals if v == v], dtype=F64)
        return exp61.flat_bootstrap_ci(t, 4000, 0)

    ceiling = max(m("offline_kwta"), m("kmeans_multi"))
    b_mean, b_lo, b_hi = tci(acc["online_bounded"]["kill"])
    bounded_ratio = b_mean / ceiling if ceiling > 1e-9 else float("nan")
    conv_ratio = m("online_converged") / ceiling if ceiling > 1e-9 else float("nan")
    bf = [b - f for b, f in zip(acc["online_bounded"]["kill"], acc["online_frozen"]["kill"])]   # learning margin
    bf_mean, bf_lo, bf_hi = tci(bf)
    cb = [c - b for c, b in zip(acc["online_converged"]["kill"], acc["online_bounded"]["kill"])]  # the locality cost
    cb_mean, cb_lo, cb_hi = tci(cb)
    learning_matters = bf_lo > 0                              # (bounded-frozen) across-seed CI-lo > 0
    locality_cost_robust = cb_lo > 0                         # (converged-bounded) across-seed CI-lo > 0
    bounded_reaches = (bounded_ratio == bounded_ratio and bounded_ratio >= 0.8 and b_lo > 0)
    if not calib_ok:
        verdict = "INVALID — anchor missed +0.109/0.222."
    elif bounded_reaches and learning_matters:
        verdict = (f"LOCALITY ~FREE (n={args.seeds}) — UNEXPECTED vs the n=3 screen, SCRUTINIZE: bounded "
                   f"reaches the build_S ceiling (ratio {bounded_ratio:.2f}, across-seed B-KILL CI "
                   f"[{b_lo:+.3f},{b_hi:+.3f}]) and beats frozen (bounded-frozen across-seed CI-lo {bf_lo:+.3f}).")
    elif locality_cost_robust:
        verdict = (f"LOCALITY COST BANKED (n={args.seeds}) — converged-MINUS-bounded across-seed CI "
                   f"[{cb_lo:+.3f},{cb_hi:+.3f}] (lo>0 ⇒ the locality cost is ROBUST across seeds): on the "
                   f"CORRECT operator build_S the genuinely bounded-memory LOCAL writer falls short of the "
                   f"global/accumulated writer. bounded ratio {bounded_ratio:.2f}, across-seed B-KILL CI "
                   f"[{b_lo:+.3f},{b_hi:+.3f}] (reaches ceiling={bounded_reaches}); learning_matters "
                   f"(bounded-frozen CI-lo {bf_lo:+.3f}). → On the right operator the GLOBAL pass is "
                   f"LOAD-BEARING; a local bounded-memory writer needs help → replay-reweight / Oracle-E TEM "
                   f"writer are the gap-closers (replay re-enters with a MEASURED gap). exp70's 'locality free' "
                   f"= a 1st-order-representation artifact, banked retracted at n={args.seeds}.")
    else:
        verdict = (f"INCONCLUSIVE (n={args.seeds}) — converged-minus-bounded across-seed CI "
                   f"[{cb_lo:+.3f},{cb_hi:+.3f}] straddles 0; locality cost not robustly established.")
    out = {
        "experiment": "72_online_local_on_build_s (decisive control on the CORRECT 2nd-order operator; rung-1; NOT graduation)",
        "precommit": "notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md (§7)",
        "config": {**vars(args), "V": V, "windows": len(windows), "k_sym": k_sym,
                   "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
        "anchor": {"spec": a_spec, "kq": anchor["king_queen_cos"], "calib_ok": calib_ok},
        "metric": "kill = para_cos - within-set-shuffle_cos (hubness-immune B-KILL), partition input = build_S (2nd-order)",
        "results": {n: {"kill_mean": m(n), "per_seed_kill": acc[n]["kill"], "per_seed_lo": acc[n]["lo"]} for n in acc},
        "build_S_ceiling": ceiling,
        "across_seed": {"bounded_BKILL_ci": [b_mean, b_lo, b_hi], "bounded_over_ceiling_ratio": bounded_ratio,
                        "converged_over_ceiling_ratio": conv_ratio,
                        "bounded_minus_frozen_ci": [bf_mean, bf_lo, bf_hi],
                        "converged_minus_bounded_ci": [cb_mean, cb_lo, cb_hi]},
        "learning_matters": learning_matters, "locality_cost_robust": locality_cost_robust,
        "bounded_reaches_ceiling": bounded_reaches, "VERDICT": verdict,
    }
    print(json.dumps(out, indent=2, default=float))
    print(f"\n[VERDICT] {verdict}", file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
