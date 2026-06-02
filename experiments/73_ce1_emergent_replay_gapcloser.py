"""experiments/73 — CE-1 emergent-replay GAP-CLOSER on the correct operator build_S.

Precommit: notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md
(§2 arms, §3 gate, §3.5 RE-ANCHORED gate [user-approved 2026-06-01], §4 disposition, §4.5 fidelity ladder).

THE MEASURED TARGET (exp72 §7 banking run, n=10, on build_S):
  bounded-memory LOCAL writer (decay 0.7) = Arm A = +0.179 [CI +0.174,+0.184]
  global/offline partition ceiling          = +0.224          (ratio 0.80; gap +0.045)
  → locality is NOT free; the global pass is load-bearing.

THE QUESTION (CE-1, now well-motivated by a number): can an EMERGENT, LOCAL replay schedule make the
bounded-memory writer close +0.179 -> +0.224 on build_S — WITHOUT becoming the global pass in disguise?

MECHANISM (faithful, anti-homunculus — precommit §1).
  The bounded writer's operator is sppmi(S_decayed): a RECENCY-BIASED co-occurrence (decay forgets the
  early corpus). SPPMI cancels per-ROW frequency reweights (rehearsal #1's lesson), so replay must
  re-present specific PAIRS (episodes), not tokens. Arm B re-adds OBSERVED co-occurrence pairs, weighted
  by a LOCAL per-pair priority recomputed EACH epoch against the CURRENT codes:
    priority(i,j) = novelty(i)*novelty(j) * surprise(i,j)        [multiplicative, grill Q1]
      novelty(t)   = code-spread (1 - peak assembly fraction)  * assembly-balance (pattern-separation,
                     grill Q2: down-weight tokens in crowded dominant assemblies = dissimilarity-spacing)
      surprise(ij) = 1 - cos(code_i, code_j)  (settling residual: the partition has NOT yet bound i,j)
  NO supervisor reads "is this king/queen?"; NO if-X-then-Y; the boost is a smooth function of a local
  scalar over OBSERVED pairs only (no hallucinated episodes). Two-timescale: early codes immature ->
  broad boost; mature codes -> boost concentrates on the still-unresolved (paradigmatic-context) pairs.

ARMS (precommit §2; everything else held fixed = exp72 defaults):
  A  online_bounded      = exp72.online_build_s(decay 0.7)            [the banked +0.179 baseline]
  B  replay(priority)     = this writer, emergent codes-derived pair boost (the bet)
  gauge replay(random)    = same boost MASS/distribution, pair-assignment SHUFFLED (local signal destroyed)
  static replay(one-shot) = boost frozen at first computed priority (isolates the two-timescale LOOP)
  ceiling                 = offline kwta + multi-restart k-means on build_S (127 replication, ~+0.224)
  converged/frozen        = exp72.online_build_s(decay 1.0) / (decay 0.7, learn=False)
  floor / nmf / anchor    = grow_G (+0.021) / NMF (+0.19) / raw-SPPMI-SVD (+0.109/0.222 or INVALID)

GATE (re-anchored §3.5, across-seed CIs; B-KILL headline + g-gauge UNCHANGED): PASS iff ALL of
  g-headline : ArmB within-set label-shuffle B-KILL CI-lo>0 in >=4/5 seeds  (magnitude-immune verdict)
  g-close    : (B-A)/(ceiling-A) >= 0.5                                      (closes >=half the measured gap)
  g-dab      : (B-A) across-seed CI-lo > 0
  g-static   : (B-static) across-seed CI-lo > 0                             (the LOOP, not a one-shot reweight)
  g-gauge    : (gauge-A) across-seed CI-lo NOT > 0                          (random reorder does NOT close it)
  g4         : d_eff(B)/d_eff(A) >= 0.5                                     (no collapse)
  calib      : SVD anchor +0.109/0.222
Disposition (§4): SCREEN-PASS / NULL-BUT-RISING (trajectory B rises rel A -> ONE escalation) / NULL
(replay-invariant: global pass irreducible) / INVALID. RUNG-1 substrate-free; NOT a graduation.

Fast path: the base co-occurrence stream is seed- AND arm-independent and obeys Pi_E = decay^m*Pi_{E-1}+S
(S = one-epoch decayed stream). Stream the corpus ONCE -> every arm/seed/epoch is cheap matmuls, and the
mode="none" path reproduces exp72.online_build_s EXACTLY (asserted in --smoke). Reuses exp61/63/65/68/70/72.
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
exp63 = _load("exp63", "63_directional_successor_oracle.py")
exp65 = _load("exp65", "65_escape_route_triage.py")
exp68 = _load("exp68", "68_nonlinear_competition_kill_test.py")
exp70 = _load("exp70", "70_wikitext_partition_headtohead.py")
exp71 = _load("exp71", "71_operator_repr_diagnostic.py")
exp72 = _load("exp72", "72_online_local_on_build_s.py")

from energy_memory.phase2.corpus import build_vocabulary, encode_texts, make_windows  # noqa: E402
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402
EPS = 1e-12; F64 = torch.float64
l2rows = exp70.l2rows


def stream_S(windows, V, special, chunk, decay):
    """One epoch of exp72's decayed chunk stream from Pi=0 -> (S, m). Pi_E = decay^m*Pi_{E-1} + S exactly
    reproduces exp72.online_build_s's cross-epoch Pi sequence (asserted in --smoke)."""
    Pi = torch.zeros((V, V), dtype=F64); m = 0
    for s in range(0, len(windows), chunk):
        Cb, _u, _t = exp61.build_cooccurrence(windows[s:s + chunk], V, special, device="cpu")
        Pi = decay * Pi + Cb; m += 1
    return Pi, m


def pair_priority(Y, obs, mode, gen, k, pair_counts=None):
    """LOCAL per-pair replay priority from the CURRENT codes (None if codes not yet formed).
    priority(i,j) = novelty(i)*novelty(j)*surprise(i,j), max-normalized to [0,1], OBSERVED pairs only.
    mode='freq' = the codes-INDEPENDENT deflationary control: pure inverse-co-occurrence-frequency targeting
    (is "surprise" doing more than up-weighting rare pairs?). Needs pair_counts (= C_obs)."""
    if mode == "freq":
        P = (1.0 / pair_counts.clamp(min=1.0)) * obs                # rarer observed pair -> higher priority
        P.fill_diagonal_(0.0)
        mx = float(P.max())
        return P / mx if mx > EPS else None
    Yn = l2rows(Y)
    pk = Yn.abs().max(dim=1)
    nov = (1.0 - pk.values).clamp(min=0.0)                          # code-spread: under-committed token
    counts = torch.bincount(pk.indices, minlength=k).to(F64)        # dominant-assembly populations
    nov = nov * (1.0 / counts[pk.indices].clamp(min=1.0).sqrt())    # pattern-separation: space crowded assemblies
    surprise = (1.0 - (Yn @ Yn.T).clamp(-1.0, 1.0)).clamp(min=0.0)  # settling residual: codes disagree on i,j
    P = (nov.unsqueeze(1) * nov.unsqueeze(0)) * surprise
    P = P * obs
    P.fill_diagonal_(0.0)
    mx = float(P.max())
    if mx <= EPS:
        return None
    P = P / mx
    if mode == "gauge":                                            # destroy local signal, preserve mass+distribution
        ij = obs.nonzero(as_tuple=False)
        ij = ij[ij[:, 0] != ij[:, 1]]
        vals = P[ij[:, 0], ij[:, 1]]
        Pg = torch.zeros_like(P)
        Pg[ij[:, 0], ij[:, 1]] = vals[torch.randperm(vals.shape[0], generator=gen)]
        P = Pg
    return P


def point_bkill(Yn, para, shuf):
    """Cheap (no-bootstrap) within-set label-shuffle B-KILL point estimate for the per-epoch trajectory."""
    def mc(pairs):
        if not pairs:
            return 0.0
        idx = torch.tensor(pairs, dtype=torch.long)
        return float((Yn[idx[:, 0]] * Yn[idx[:, 1]]).sum(dim=1).mean())
    return mc(para) - mc(shuf)


def replay_writer(S, d_epoch, C_obs, V, k, cap, k_sym, epochs, inner, seed, mode,
                  replay_strength, learn=True, para=None, shuf=None, return_op=False):
    """Online k-WTA on build_S with an emergent replay boost. mode in {none,priority,gauge,static}.
    mode='none' reproduces exp72.online_build_s; replay re-adds OBSERVED pairs weighted by current-codes
    priority (recomputed each epoch = two-timescale; 'static' freezes it at first compute).
    return_op=True also returns the FINAL-epoch SPPMI (V×V 1st-order context profiles) — for the rung-2
    FHRR context-bundle port (exp74; build_S emerges as the FHRR Gram). Default off → byte-identical 2-tuple
    return (the Report-128 n=10 path is unaffected)."""
    W = exp70._init_W(V, k, seed)
    Y = torch.zeros((V, k), dtype=F64)
    Pi = torch.zeros((V, V), dtype=F64)
    obs = (C_obs > 0).to(F64)
    P_static = None
    traj = []
    sppmi = None
    for E in range(epochs):
        Pi = d_epoch * Pi + S                                      # base bounded stream (== exp72)
        Pi_eff = Pi
        if mode != "none" and float(l2rows(Y).abs().sum()) > EPS:   # codes exist (epoch >= 1)
            if mode == "static" and P_static is not None:
                P = P_static
            else:
                gen = torch.Generator().manual_seed(seed * 7919 + E * 131 + 17)
                P = pair_priority(Y, obs, mode, gen, k, pair_counts=C_obs)
                if mode == "static":
                    P_static = P
            if P is not None:
                Pi_eff = Pi + replay_strength * (C_obs * P)         # re-present surprising OBSERVED pairs
        uni = Pi_eff.sum(dim=1)
        sppmi = exp61.build_sppmi(Pi_eff, uni, float(uni.sum().item()), k_sym)
        L = l2rows(exp61.build_S(sppmi))
        for _ in range(inner):
            Y, W_new = exp70._kwta_step(L, W, cap)
            if learn:
                W = W_new
        if para is not None:
            traj.append(point_bkill(l2rows(Y), para, shuf))
    if return_op:
        return l2rows(Y), traj, sppmi
    return l2rows(Y), traj


def deff_codes(Y):
    s = torch.linalg.svdvals(l2rows(Y).to(F64))
    return float((s.sum() ** 2) / (s * s).sum().clamp(min=EPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source", default="wikitext", dest="corpus_source")
    ap.add_argument("--corpus-docs", type=int, default=40000, dest="corpus_docs")   # Phase-B 2nd-corpus slice
    ap.add_argument("--fast-arms", action="store_true", dest="fast_arms")            # route baselines via fast path (large corpora)
    # Per-corpus calibration band (§8: each corpus derives its OWN anchor; defaults = the WikiText band).
    ap.add_argument("--calib-lo", type=float, default=0.082, dest="calib_lo")
    ap.add_argument("--calib-hi", type=float, default=0.137, dest="calib_hi")
    ap.add_argument("--calib-kq", type=float, default=0.222, dest="calib_kq")
    ap.add_argument("--calib-kq-tol", type=float, default=0.04, dest="calib_kq_tol")
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
    ap.add_argument("--replay-strength", type=float, default=1.0, dest="replay_strength")  # FROZEN by principle
    ap.add_argument("--chunk", type=int, default=30000)
    ap.add_argument("--floor-epochs", type=int, default=20, dest="floor_epochs")
    ap.add_argument("--kmeans-restarts", type=int, default=10, dest="kmeans_restarts")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--planted-seed", type=int, default=0, dest="planted_seed")
    ap.add_argument("--smoke", action="store_true", help="fast apparatus check (asserts mode=none == exp72)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = min(args.seeds, 2); args.online_epochs = min(args.online_epochs, 6)

    splits = exp61.load_corpus(args.corpus_source, args, args.planted_seed)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    V = len(vocab.id_to_token); special = {vocab.unk_id, vocab.mask_id}
    windows = make_windows(encode_texts(splits["train"], vocab), args.W)
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    k_sym, dens, _ = exp61.pick_k_by_density(C, uni, total, 0.3, 0.5); k_sym = k_sym or 1
    sppmi = exp61.build_sppmi(C, uni, total, k_sym)
    build_S = exp61.build_S(sppmi)
    L2 = l2rows(build_S)
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)
    shuf = exp68.derange_partners(para, args.boot_seed)
    anchor = exp63.raw_sppmi_svd_anchor(sppmi, para, rand, collo, kq, args.svd_rank, args.n_boot, args.boot_seed)
    a_spec = anchor["specificity_para_minus_random"]["mean"]
    calib_ok = bool(args.calib_lo <= a_spec <= args.calib_hi and
                    (args.calib_kq < 0 or abs(anchor["king_queen_cos"] - args.calib_kq) <= args.calib_kq_tol))
    # base decayed stream computed ONCE (seed/arm-independent); fast path for every arm.
    S_dec, m_chunks = stream_S(windows, V, special, args.chunk, args.online_decay)
    d_epoch = args.online_decay ** m_chunks
    print(f"V={V} windows={len(windows)} chunks={m_chunks} d_epoch={d_epoch:.2e} k_sym={k_sym} "
          f"n_para={len(para)} [anchor] spec={a_spec:+.4f} kq={anchor['king_queen_cos']:.3f} "
          f"calib_ok={calib_ok}", file=sys.stderr, flush=True)

    arms = ("offline_kwta", "kmeans_multi", "nmf", "floor",
            "online_converged", "armA_bounded", "online_frozen", "armB_priority", "gauge", "static", "freq")
    acc = {n: {"kill": [], "lo": []} for n in arms}
    trajA, trajB = [], []
    deff_ratio = []

    def rec(name, Y, bseed):
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
        # ceiling + baselines. Default = the banked exp72 path (Arm A == the +0.179 WikiText baseline, exactly).
        # --fast-arms routes them through the proven-equivalent fast path (replay_writer mode=none, <1e-9 vs
        # exp72 in --smoke) — needed on large corpora (TinyStories) where exp72's per-epoch re-stream is slow.
        if args.fast_arms:
            conv, _ = replay_writer(C, 1.0, C, V, args.k, args.cap, k_sym, args.online_epochs, args.inner,
                                    seed, "none", args.replay_strength)               # decay=1 → S=C, accumulating
            YA, _ = replay_writer(S_dec, d_epoch, C, V, args.k, args.cap, k_sym, args.online_epochs, args.inner,
                                  seed, "none", args.replay_strength)
            frz, _ = replay_writer(S_dec, d_epoch, C, V, args.k, args.cap, k_sym, args.online_epochs, args.inner,
                                   seed, "none", args.replay_strength, learn=False)
            rec("online_converged", conv, bseed); rec("armA_bounded", YA, bseed); rec("online_frozen", frz, bseed)
        else:
            rec("online_converged", exp72.online_build_s(windows, V, special, args.k, args.cap, k_sym, args.chunk,
                                                         args.online_epochs, args.inner, seed, decay=1.0, learn=True), bseed)
            YA = exp72.online_build_s(windows, V, special, args.k, args.cap, k_sym, args.chunk,
                                      args.online_epochs, args.inner, seed, decay=args.online_decay, learn=True)
            rec("armA_bounded", YA, bseed)
            rec("online_frozen", exp72.online_build_s(windows, V, special, args.k, args.cap, k_sym, args.chunk,
                                                      args.online_epochs, args.inner, seed, decay=args.online_decay, learn=False), bseed)
        # replay arms via the fast precomputed-S writer
        if args.smoke:  # validate the fast path reproduces the banked writer before trusting the trajectory
            YA_fast, _ = replay_writer(S_dec, d_epoch, C, V, args.k, args.cap, k_sym, args.online_epochs,
                                       args.inner, seed, "none", args.replay_strength)
            diff = float((l2rows(YA) - YA_fast).abs().max())
            print(f"  [smoke seed {seed}] |armA_exp72 - armA_fast|_max = {diff:.2e}", file=sys.stderr, flush=True)
            assert diff < 1e-9, f"fast path diverges from exp72 ({diff:.2e}) — trajectory untrustworthy"
        YB, tB = replay_writer(S_dec, d_epoch, C, V, args.k, args.cap, k_sym, args.online_epochs, args.inner,
                               seed, "priority", args.replay_strength, para=para, shuf=shuf)
        rec("armB_priority", YB, bseed)
        rec("gauge", replay_writer(S_dec, d_epoch, C, V, args.k, args.cap, k_sym, args.online_epochs,
                                   args.inner, seed, "gauge", args.replay_strength)[0], bseed)
        rec("static", replay_writer(S_dec, d_epoch, C, V, args.k, args.cap, k_sym, args.online_epochs,
                                    args.inner, seed, "static", args.replay_strength)[0], bseed)
        rec("freq", replay_writer(S_dec, d_epoch, C, V, args.k, args.cap, k_sym, args.online_epochs,
                                  args.inner, seed, "freq", args.replay_strength)[0], bseed)   # inverse-freq control
        _, tA = replay_writer(S_dec, d_epoch, C, V, args.k, args.cap, k_sym, args.online_epochs, args.inner,
                              seed, "none", args.replay_strength, para=para, shuf=shuf)
        trajA.append(tA); trajB.append(tB)
        deff_ratio.append(deff_codes(YB) / max(deff_codes(YA), EPS))
        print(f"  [seed {seed}] off_kwta={acc['offline_kwta']['kill'][-1]:+.3f} km={acc['kmeans_multi']['kill'][-1]:+.3f} "
              f"conv={acc['online_converged']['kill'][-1]:+.3f} | A={acc['armA_bounded']['kill'][-1]:+.3f} "
              f"B={acc['armB_priority']['kill'][-1]:+.3f}(lo {acc['armB_priority']['lo'][-1]:+.3f}) "
              f"gauge={acc['gauge']['kill'][-1]:+.3f} static={acc['static']['kill'][-1]:+.3f} "
              f"freq={acc['freq']['kill'][-1]:+.3f} deff_r={deff_ratio[-1]:.2f}", file=sys.stderr, flush=True)

    def m(n):
        xs = [x for x in acc[n]["kill"] if x == x]; return st.mean(xs) if xs else float("nan")

    def tci(vals):
        t = torch.tensor([v for v in vals if v == v], dtype=F64)
        return exp61.flat_bootstrap_ci(t, 4000, 0) if t.numel() else (float("nan"),) * 3

    ceiling = max(m("offline_kwta"), m("kmeans_multi"))
    mA, mB = m("armA_bounded"), m("armB_priority")
    gap = ceiling - mA
    close_frac = (mB - mA) / gap if abs(gap) > 1e-9 else float("nan")
    ba = [b - a for b, a in zip(acc["armB_priority"]["kill"], acc["armA_bounded"]["kill"])]
    bs = [b - s for b, s in zip(acc["armB_priority"]["kill"], acc["static"]["kill"])]
    ga = [g - a for g, a in zip(acc["gauge"]["kill"], acc["armA_bounded"]["kill"])]
    bf = [b - f for b, f in zip(acc["armB_priority"]["kill"], acc["freq"]["kill"])]    # B vs inverse-freq control
    fa = [f - a for f, a in zip(acc["freq"]["kill"], acc["armA_bounded"]["kill"])]     # does freq alone close the gap?
    ba_ci, bs_ci, ga_ci, bf_ci, fa_ci = tci(ba), tci(bs), tci(ga), tci(bf), tci(fa)
    b_lo_pass = sum(1 for x in acc["armB_priority"]["lo"] if x == x and x > 0)
    need = max(4, -(-4 * args.seeds // 5))  # >=4/5 (ceil(0.8*seeds), min 4)
    mtA = [st.mean([trajA[s][e] for s in range(len(trajA))]) for e in range(args.online_epochs)]
    mtB = [st.mean([trajB[s][e] for s in range(len(trajB))]) for e in range(args.online_epochs)]
    diff_traj = [b - a for a, b in zip(mtA, mtB)]
    rising = bool(diff_traj[-1] > diff_traj[0] + 1e-4 and diff_traj[-1] > 0)
    deff_ok = bool(st.mean(deff_ratio) >= 0.5)

    g_head = b_lo_pass >= need
    g_close = close_frac == close_frac and close_frac >= 0.5
    g_dab = ba_ci[1] > 0
    g_static = bs_ci[1] > 0
    g_gauge = not (ga_ci[1] > 0)
    if not calib_ok:
        verdict = "INVALID — anchor missed +0.109/0.222; fix harness, re-run."
    elif g_head and g_close and g_dab and g_static and g_gauge and deff_ok:
        verdict = (f"SCREEN-PASS (rung-1, n={args.seeds}) — emergent replay CLOSES the locality gap: "
                   f"ArmB {mB:+.3f} vs ArmA {mA:+.3f} (closes {close_frac*100:.0f}% of the +{gap:.3f} gap), "
                   f"B-KILL lo>0 {b_lo_pass}/{args.seeds}, beats static ({bs_ci[1]:+.3f} lo>0) AND the random-"
                   f"reorder gauge does NOT ({ga_ci[1]:+.3f} lo). The emergent priority (not a hand schedule) "
                   f"is operative. NOT a graduation — licenses escalation up the fidelity ladder (§4.5).")
    elif rising:
        verdict = (f"NULL-BUT-RISING (rung-1, n={args.seeds}) — ArmB fails the frozen gate (close {close_frac*100:.0f}%, "
                   f"B-KILL lo>0 {b_lo_pass}/{args.seeds}) BUT the trajectory rises in B rel A "
                   f"(diff {diff_traj[0]:+.3f}->{diff_traj[-1]:+.3f}). Per §4: ONE pre-registered escalation "
                   f"(heavier/longer budget, 2nd dataset), then commit.")
    else:
        why = []
        if ga_ci[1] > 0: why.append("the random-reorder GAUGE reproduces ArmB (reordering artifact)")
        if bs_ci[1] <= 0: why.append("ArmB does NOT beat the static one-shot reweight (no two-timescale loop)")
        if not g_close: why.append(f"ArmB closes only {close_frac*100:.0f}% of the gap (<50%)")
        if b_lo_pass < need: why.append(f"B-KILL lo>0 only {b_lo_pass}/{args.seeds} (<{need})")
        verdict = (f"NULL (rung-1, n={args.seeds}) — emergent replay does NOT close the locality gap "
                   f"[{'; '.join(why) or 'ArmB ~ ArmA, flat trajectory'}]. The 121-127 bound is banked as "
                   f"capability-level AND REPLAY-INVARIANT: the global/offline nature of the partition is "
                   f"IRREDUCIBLE (you must see all the data at once; emergent local replay cannot substitute). "
                   f"→ next move is the Abstraction-node decision (hierarchical/latent code; Oracle-E TEM "
                   f"local writer = short-list #2), NOT another flat-code oracle (§4).")
    out = {
        "experiment": "73_ce1_emergent_replay_gapcloser (rung-1 substrate-free; gap-closer on build_S; NOT graduation)",
        "precommit": "notes/emergent-codebook/phase-3-ce1x127-replay-nonlinear-partition-precommit.md (§2/§3/§3.5/§4)",
        "config": {**vars(args), "V": V, "windows": len(windows), "chunks": m_chunks, "d_epoch": d_epoch,
                   "k_sym": k_sym, "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
        "anchor": {"spec": a_spec, "kq": anchor["king_queen_cos"], "calib_ok": calib_ok},
        "metric": "kill = para_cos - within-set-shuffle_cos (hubness-immune B-KILL), operator = build_S (2nd-order)",
        "results": {n: {"kill_mean": m(n), "per_seed_kill": acc[n]["kill"], "per_seed_lo": acc[n]["lo"]} for n in arms},
        "ceiling": ceiling, "gap": gap, "close_fraction": close_frac,
        "across_seed": {"B_minus_A_ci": list(ba_ci), "B_minus_static_ci": list(bs_ci),
                        "gauge_minus_A_ci": list(ga_ci), "B_minus_freq_ci": list(bf_ci),
                        "freq_minus_A_ci": list(fa_ci), "deff_ratio_mean": st.mean(deff_ratio)},
        "trajectory": {"armA_mean_by_epoch": mtA, "armB_mean_by_epoch": mtB, "B_minus_A_by_epoch": diff_traj,
                       "rising": rising},
        "gate": {"g_headline": g_head, "g_close": g_close, "g_dab": g_dab, "g_static": g_static,
                 "g_gauge": g_gauge, "g4_no_collapse": deff_ok, "b_lo_pass": b_lo_pass, "need": need},
        "VERDICT": verdict,
    }
    print(json.dumps(out, indent=2, default=float))
    print(f"\n[VERDICT] {verdict}", file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
