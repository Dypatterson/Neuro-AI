"""experiments/70 — WikiText partition head-to-head: online-LOCAL k-WTA vs offline-GLOBAL k-means.

Context: notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md (§7 run-log).
After 3 probes (PM-7b prototype + rung-0 rehearsals #1/#2) converged that replay-reweight is a WEAK lever
and the nonlinear PARTITION does the work, the live question shifted to: does a LOCAL/ONLINE partition
reach the paradigmatic structure on REAL data that the OFFLINE/GLOBAL partition (k-means, the 127
competent control) reaches? This is the head-to-head 127 §0.5 + the HANDOFF originally prescribed.
Rung-1 SUBSTRATE-FREE (no build-gate). NOT graduation. Floor (055-058) untouched.

REPLAY IS DEFERRED, NOT DISREGARDED (user steer, 2026-06-01): the replay-reweight arms are dropped HERE
because 3 probes deflate them for a flat-code partition writer — but replay is parked as (a) a GAP-CLOSER
if online-local falls short of offline here, and (b) the likely lever in the HIERARCHICAL/Abstraction
regime (CONTEXT.md §2 sense 3). It is not killed.

THE ONE QUESTION: online-local k-WTA B-KILL vs offline-global k-means B-KILL on real WikiText.
  - online-local ≈ offline-global  -> locality+online is ~FREE; a LOCAL partition reaches paradigmatic
    structure on real data (a buildable local mechanism; big — points past the 121-127 linear bound).
  - online-local << offline-global  -> there IS a gap; the global pass is load-bearing -> replay / the
    Oracle-E TEM local writer become the candidates to CLOSE that gap (replay re-enters here).

Reads: hubness-immune within-set LABEL-SHUFFLE B-KILL (para_cos - partner-shuffled_cos), the headline
that caught the 119/126/127 false positives. NMF (subdominant) = "signal exists" ceiling; grow_G linear
= the +0.021 floor; raw-SPPMI-SVD anchor = validity (+0.109/0.222 or INVALID). Reuses exp61/62/63/65/68.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics as st
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
exp65 = _load("exp65", "65_escape_route_triage.py")
exp68 = _load("exp68", "68_nonlinear_competition_kill_test.py")

from energy_memory.phase2.corpus import build_vocabulary, encode_texts, make_windows  # noqa: E402
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402

EPS = 1e-12
F64 = torch.float64


def l2rows(M):
    return M / M.norm(dim=1, keepdim=True).clamp(min=EPS)


def _kwta_step(L, W, cap):
    a = (L @ W.T).clamp(min=0.0)
    cap = min(cap, a.shape[1])
    thr = torch.topk(a, cap, dim=1).values[:, -1:].clamp(min=EPS)
    mask = (a >= thr).to(F64)
    Y = a * mask
    won = mask.T @ L
    cnt = mask.sum(0).clamp(min=1.0).unsqueeze(1)
    return Y, l2rows(W + 0.5 * (won / cnt - W))


def _init_W(V, k, seed):
    g = torch.Generator().manual_seed(seed * 104729 + 7)
    return l2rows(torch.rand((k, V), generator=g, dtype=F64))


def kmeans_onehot(L, k, seed, iters=50):
    """OFFLINE-GLOBAL competent control (127's): Lloyd k-means on the rows of L (token context profiles),
    one-hot codes (V x k). Sees ALL profiles at once -> the global partition the online-local must match."""
    V = L.shape[0]
    g = torch.Generator().manual_seed(seed * 2654435761 % (2**31))
    cent = L[torch.randperm(V, generator=g)[:k]].clone()                  # random init centroids
    assign = torch.zeros(V, dtype=torch.long)
    for _ in range(iters):
        d = torch.cdist(L, cent)                                          # (V,k)
        assign = d.argmin(dim=1)
        for c in range(k):
            m = assign == c
            if bool(m.any()):
                cent[c] = L[m].mean(dim=0)
    Y = torch.zeros((V, k), dtype=F64)
    Y[torch.arange(V), assign] = 1.0
    return l2rows(Y)


def kwta_offline(L, k, cap, epochs, seed):
    W = _init_W(L.shape[0], k, seed)
    Y = None
    for _ in range(epochs):
        Y, W = _kwta_step(L, W, cap)
    return l2rows(Y)


def online_local_kwta(windows, V, special, k, cap, k_sym, chunk, epochs, seed, decay=1.0, learn=True):
    """ONLINE streaming k-WTA: stream windows in corpus order (chunked), update co-occurrence
    Pi = decay*Pi + Cb, and after EACH chunk take ONE competitive step on the CURRENT SPPMI profiles
    (incremental W). decay=1.0 -> ACCUMULATING (by end-of-epoch Pi = the FULL operator, so this is
    offline-with-warmup; the gap to offline is just step-count). decay<1.0 -> BOUNDED-MEMORY / genuinely
    LOCAL (recency-weighted Pi NEVER equals the full global operator) — the real locality test."""
    W = _init_W(V, k, seed)
    Y = torch.zeros((V, k), dtype=F64)
    Pi = torch.zeros((V, V), dtype=F64)
    n = len(windows)
    for _ep in range(epochs):
        for s in range(0, n, chunk):
            Cb, _u, _t = exp61.build_cooccurrence(windows[s:s + chunk], V, special, device="cpu")
            Pi = decay * Pi + Cb
            uni = Pi.sum(dim=1)
            L = l2rows(exp61.build_sppmi(Pi, uni, float(uni.sum().item()), k_sym))
            Y, W_new = _kwta_step(L, W, cap)
            if learn:                                    # learn=False -> frozen-random partition (the 127 incompetent control)
                W = W_new
    return l2rows(Y)


def _kill(Y, para, rand, collo, shuf, n_boot, seed):
    r = exp68.read_specificity(Y, para, rand, collo, shuf, n_boot, seed)
    return r["kill_para_minus_shuf"], r["spec_para_minus_rand"]   # each (mean, lo, hi)


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
    ap.add_argument("--online-decay", type=float, default=0.7, dest="online_decay")
    ap.add_argument("--chunk", type=int, default=30000)
    ap.add_argument("--floor-epochs", type=int, default=20, dest="floor_epochs")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--planted-seed", type=int, default=0, dest="planted_seed")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # ---- corpus / operator (seed-independent, built once) ----
    splits = exp61.load_corpus(args.corpus_source, args, args.planted_seed)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    V = len(vocab.id_to_token)
    special = {vocab.unk_id, vocab.mask_id}
    windows = make_windows(encode_texts(splits["train"], vocab), args.W)
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    k_sym, dens, _ = exp61.pick_k_by_density(C, uni, total, 0.3, 0.5)
    k_sym = k_sym or 1
    sppmi = exp61.build_sppmi(C, uni, total, k_sym)
    L_full = l2rows(sppmi)                                       # 1st-order context profiles (the partition input)
    build_S = exp61.build_S(sppmi)                              # 2nd-order operator (NMF ceiling, 127 reference)
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)
    shuf = exp68.derange_partners(para, args.boot_seed)
    anchor = exp63.raw_sppmi_svd_anchor(sppmi, para, rand, collo, kq, args.svd_rank, args.n_boot, args.boot_seed)
    a_spec = anchor["specificity_para_minus_random"]["mean"]
    a_lo, a_hi = anchor["specificity_para_minus_random"]["ci"]
    hw = (a_hi - a_lo) / 2 if a_hi == a_hi else 0.0
    calib_ok = bool((0.082 - hw) <= a_spec <= (0.137 + hw) and abs(anchor["king_queen_cos"] - 0.222) <= 0.04)
    print(f"V={V} windows={len(windows)} k_sym={k_sym} dens={dens:.3f} n_para={len(para)} "
          f"n_shuf={len(shuf)} [anchor] spec={a_spec:+.4f} kq={anchor['king_queen_cos']:.3f} "
          f"calib_ok={calib_ok}", file=sys.stderr, flush=True)

    acc = {n: {"kill": [], "kill_lo": [], "spec": []} for n in
           ("nmf", "floor", "offline_kmeans", "offline_kwta", "online_converged", "online_bounded",
            "online_bounded_frozen")}

    def rec(name, Y, seed):
        kl, sp = _kill(Y, para, rand, collo, shuf, args.n_boot, args.boot_seed)
        acc[name]["kill"].append(kl[0]); acc[name]["kill_lo"].append(kl[1]); acc[name]["spec"].append(sp[0])
        return kl

    for seed in range(args.seeds):
        # NMF ceiling (subdominant modes) on build_S
        rec("nmf", exp65.nmf_slots(build_S, args.k, seed=seed), seed)
        # linear grow_G floor on build_S
        sub = TorchFHRR(dim=args.D, seed=seed, device="cpu", alpha_anti=1.0)
        G0 = sub.random_vectors(V); deff0 = exp61.d_eff(sub, G0)
        cooc_x = torch.log1p(exp61.cooc_counts_for_pairs(C, torch.tensor(para, dtype=torch.long))) if para else None
        fr = exp65.faithful_read(sub, G0, deff0, build_S, para, rand, cooc_x, [0.0, 0.1],
                                 args.floor_epochs, 0.3, 0.9, args.n_boot, args.boot_seed)
        fb = fr["best"]["spec"] if fr["best"] else float("nan")
        acc["floor"]["kill"].append(fb); acc["floor"]["kill_lo"].append(float("nan")); acc["floor"]["spec"].append(fb)
        # OFFLINE-GLOBAL competent controls (the ceiling the online-local must match)
        rec("offline_kmeans", kmeans_onehot(L_full, args.k, seed), seed)
        rec("offline_kwta", kwta_offline(L_full, args.k, args.cap, args.offline_epochs, seed), seed)
        # ONLINE-LOCAL k-WTA (the question)
        # CONVERGED accumulating (decay=1.0, many epochs) — confound-confirm: should ~= offline
        rec("online_converged",
            online_local_kwta(windows, V, special, args.k, args.cap, k_sym,
                              args.chunk, args.online_epochs, seed, decay=1.0), seed)
        # BOUNDED-MEMORY / genuinely LOCAL (decay<1, never sees the full operator) — the real locality test
        ob = rec("online_bounded",
                 online_local_kwta(windows, V, special, args.k, args.cap, k_sym,
                                   args.chunk, args.online_epochs, seed, decay=args.online_decay), seed)
        # FROZEN-RANDOM bounded (no Hebbian learning) — the 127 incompetent control: does LEARNING matter?
        ofz = rec("online_bounded_frozen",
                  online_local_kwta(windows, V, special, args.k, args.cap, k_sym,
                                    args.chunk, args.online_epochs, seed, decay=args.online_decay, learn=False), seed)
        print(f"  [seed {seed}] nmf={acc['nmf']['kill'][-1]:+.3f} floor={fb:+.3f} "
              f"off_kwta={acc['offline_kwta']['kill'][-1]:+.3f} converged={acc['online_converged']['kill'][-1]:+.3f} "
              f"BOUNDED_LOCAL={ob[0]:+.3f} (lo {ob[1]:+.3f}) frozen_rand={ofz[0]:+.3f}",
              file=sys.stderr, flush=True)

    def m(name, key="kill"):
        xs = [x for x in acc[name][key] if x == x]
        return st.mean(xs) if xs else float("nan")
    off = max(m("offline_kmeans"), m("offline_kwta"))
    conv_ratio = m("online_converged") / off if off > 1e-9 else float("nan")     # confound-confirm (~=1 expected)
    bounded = m("online_bounded")
    bounded_ratio = bounded / off if off > 1e-9 else float("nan")                 # the REAL locality test
    bounded_lo_pass = sum(1 for x in acc["online_bounded"]["kill_lo"] if x == x and x > 0)
    confound_confirmed = conv_ratio == conv_ratio and conv_ratio >= 0.8           # converged ~= offline
    frozen_m = m("online_bounded_frozen")
    learning_matters = (bounded - frozen_m) >= 0.05                                # bounded-LEARNED beats frozen-random
    bounded_strong = bounded_ratio == bounded_ratio and bounded_ratio >= 0.8 and bounded_lo_pass >= max(1, args.seeds - 1)
    if not calib_ok:
        verdict = "INVALID — calibration anchor missed +0.109/0.222."
    elif bounded_strong and not learning_matters:
        verdict = (f"DEFLATED — the bounded-local B-KILL ({bounded:+.3f}) is REPRODUCED by a FROZEN-RANDOM "
                   f"no-learning partition ({frozen_m:+.3f}) → the win is partition/co-membership INFLATION, "
                   f"NOT learned local structure (the 127 incompetent-control trap, inverted). Do NOT claim a "
                   f"learned-local-mechanism result.")
    elif bounded_strong and learning_matters:
        verdict = (f"LOCALITY ~FREE (LEARNING-DRIVEN) — the bounded-memory LEARNED local writer matches "
                   f"offline-global (ratio {bounded_ratio:.2f}, B-KILL lo>0 {bounded_lo_pass}/{args.seeds}), "
                   f"BEATS the frozen-random control ({bounded:+.3f} vs {frozen_m:+.3f}), converged confound-"
                   f"confirm {conv_ratio:.2f}. A LOCAL, ONLINE, bounded-memory, LEARNED nonlinear-partition "
                   f"writer reaches paradigmatic structure on real WikiText where the linear local read gets ~0 "
                   f"→ a buildable local mechanism past the 121-127 LINEAR bound. RUNG-1 SCREEN, 3 seeds — NOT a "
                   f"graduation; full multi-seed + adversarial verification is the next gate. (Replay parked as "
                   f"the hierarchical-regime lever; magnitudes are partition-inflated — NOT 'beats SVD/NMF'.)")
    else:
        verdict = (f"LOCALITY COST — bounded-memory local writer falls short of offline-global (ratio "
                   f"{bounded_ratio:.2f}; lo>0 {bounded_lo_pass}/{args.seeds}; converged {conv_ratio:.2f}, frozen "
                   f"{frozen_m:+.3f}) → replay-reweight / Oracle-E TEM writer are the gap-closers (replay re-enters).")
    out = {
        "experiment": "70_wikitext_partition_headtohead (rung-1 substrate-free; NOT graduation; replay deferred-not-killed)",
        "precommit": "notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md (§7)",
        "config": {**vars(args), "V": V, "windows": len(windows), "k_sym": k_sym,
                   "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
        "anchor": {"spec": a_spec, "kq": anchor["king_queen_cos"], "calib_ok": calib_ok},
        "metric": "kill = para_cos - within-set-shuffle_cos (hubness-immune B-KILL)",
        "results": {n: {"kill_mean": m(n), "kill_lo_mean": m(n, "kill_lo"), "spec_mean": m(n, "spec")}
                    for n in acc},
        "bounded_local_over_offline_ratio": bounded_ratio,
        "converged_over_offline_ratio_confound_confirm": conv_ratio,
        "bounded_minus_frozen_random": bounded - frozen_m, "learning_matters": learning_matters,
        "VERDICT": verdict,
    }
    print(json.dumps(out, indent=2, default=float))
    print(f"\n[VERDICT] {verdict}", file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
