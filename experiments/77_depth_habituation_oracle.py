"""experiments/77 — Depth × local-temporal-write (habituation-deflation) oracle. DRILL-DOWN, NOT graduation.

Frozen pre-commit: notes/emergent-codebook/phase-3-depth-habituation-precommit.md.

The culmination of the 121-131 arc + the user's two reframes (not-flat → DEPTH holds the residual;
homunculus-dissolution → TEMPORAL HABITUATION does the peeling). A multi-LAYER stack where each layer
captures a nonlinear partition, then a LOCAL habituation (divisive-activity) deflation suppresses the
captured (dominant) mode so the residual — HELD as a distinct layer — exposes the next mode to the next
layer. The escape from the single-projection bound (channel-invariant per 130, capacity-invariant per
131) IFF the residual is held distinctly AND depth strictly beats L=1.

Per layer ℓ over M₀=build_S:
  CAPTURE  Yℓ = kwta(M_{ℓ-1}, k)              (nonlinear partition = local assignment; linear arm = predicted-null)
  DEFLATE  āℓ[i]=‖Yℓ[i,:]‖₁;  Mℓ[i,j] = M_{ℓ-1}[i,j] / ((1+κāℓ[i])(1+κāℓ[j]))   (divisive habituation; HELD as distinct V×V)
  HOLD     stack [Y₁;…;Y_L] (distinct blocks, never summed)
  READ     static B-KILL on the stacked codes (NOT grow_G)

THE critical gate (precommit §4): build_sppmi already divides by marginals, so L=1 habituation ≈ a 2nd
inverse-frequency pass. A PASS must BEAT a codes-independent inverse-frequency-deflation stack by ≥+0.02
→ the escape MUST come from L≥2. Headline = within-para-set label-shuffle B-KILL (sole arbiter) ∧ spec_ci_lo>0.
g5_depth: L>1 strictly beats L=1 ∧ per-layer para_centroid_rank rises. CEILING reach-not-exceed (NMF/SVD).
Reuses exp61/63/65/68/76 verbatim. Substrate-free; FHRR build = fence.
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
exp63 = _load("exp63", "63_directional_successor_oracle.py")
exp65 = _load("exp65", "65_escape_route_triage.py")
exp68 = _load("exp68", "68_nonlinear_competition_kill_test.py")
exp76 = _load("exp76", "76_tem_local_reachability_oracle.py")

from energy_memory.phase2.corpus import build_vocabulary, encode_texts, make_windows  # noqa: E402
from energy_memory.substrate.torch_fhrr import TorchFHRR  # noqa: E402

EPS = 1e-12


def l2rows(M):
    return M / M.norm(dim=1, keepdim=True).clamp(min=EPS)


# =====================================================================
# The new mechanism: habituation deflation + the multi-layer write (capture→deflate→HOLD)
# =====================================================================

def deflate_by_habituation(M, trace, kappa):
    """Divisive-normalization deflation by a per-token activity trace (Carandini-Heeger; fixed local
    statistic). Strongly-active (dominant-mode) tokens FADE; the residual exposes the next mode."""
    denom = (1.0 + kappa * trace).clamp(min=EPS)
    return M / (denom[:, None] * denom[None, :])


def capture_kwta(M, k, seed):
    return exp68.kwta_features(M.clamp(min=0.0), k, seed=seed)            # nonlinear partition (live arm)


def capture_linear(M, k, seed):
    V = M.shape[0]
    return exp76.tem_frozen_write(M, exp68.random_nonneg(V, k, seed))     # frozen-random projection (predicted-null arm)


def multilayer_write(M0, capture, L, k, kappa, seed, deflate, gfreq, para, rand):
    """capture→deflate→HOLD. deflate ∈ {'activity','freq','none'}. Returns stacked codes + per-layer
    norms + per-layer para_centroid_rank (the g5_depth rise diagnostic)."""
    M = M0.clone()
    Ys, norms, cents = [], [], []
    for ell in range(L):
        Y = capture(M, k, seed * 97 + ell)
        Ys.append(Y)
        norms.append(float(M.norm()))
        cents.append(exp76.spectral_diagnosis(Y, para, rand).get("para_centroid_rank", float("nan")))
        if ell < L - 1 and deflate != "none":
            # habituation trace must measure DOMINANCE (non-uniform): the CURRENT operator's residual
            # row-mass (codes/depth-dependent — at L=1 ~marginal, at L>=2 the un-captured residual mass)
            # vs the static-marginal control (codes-independent, same fade-frequent direction). The smoke
            # showed l2-normalized ||Y||_1 is ~uniform => inert; row-mass is the faithful non-inert trace.
            tr = M.abs().sum(dim=1) if deflate == "activity" else gfreq
            tr = tr / tr.max().clamp(min=EPS)            # normalize => kappa is a scale-invariant deflation strength
            M = deflate_by_habituation(M, tr, kappa)
    return l2rows(torch.cat(Ys, dim=1)), norms, cents


# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-source", default="wikitext", dest="corpus_source")
    ap.add_argument("--wikitext-name", default="wikitext-2-raw-v1", dest="wikitext_name")
    ap.add_argument("--W", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-vocab", type=int, default=2000, dest="max_vocab")
    ap.add_argument("--paradigmatic-max-cooc", type=int, default=2, dest="paradigmatic_max_cooc")
    ap.add_argument("--simlex-min-sim", type=float, default=5.0, dest="simlex_min_sim")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--L-grid", default="1,2,4", dest="L_grid")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--D-floor", type=int, default=2048, dest="D_floor")
    ap.add_argument("--svd-rank", type=int, default=300, dest="svd_rank")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--bkill-min", type=int, default=8, dest="bkill_min")
    ap.add_argument("--n-boot", type=int, default=4000, dest="n_boot")
    ap.add_argument("--boot-seed", type=int, default=7, dest="boot_seed")
    ap.add_argument("--planted-seed", type=int, default=0, dest="planted_seed")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    L_grid = [int(x) for x in args.L_grid.split(",") if x != ""]
    Lmax = max(L_grid)

    # ---- corpus / build_S / pairs / anchor ----
    splits = exp61.load_corpus(args.corpus_source, args, args.planted_seed)
    vocab = build_vocabulary(splits["train"], max_vocab=args.max_vocab)
    V = len(vocab.id_to_token)
    special = {vocab.unk_id, vocab.mask_id}
    windows = make_windows(encode_texts(splits["train"], vocab), args.W)
    C, uni, total = exp61.build_cooccurrence(windows, V, special, device="cpu")
    k_sym, _, _ = exp61.pick_k_by_density(C, uni, total, 0.3, 0.5)
    if k_sym is None:
        k_sym = 1
    sppmi = exp61.build_sppmi(C, uni, total, k_sym)
    M0 = exp61.build_S(sppmi)
    para, collo, rand, kq, sha, src = exp63.select_pairs(args, vocab, C, V, special)
    shuf = exp68.derange_partners([(a, b) for (a, b) in para], args.boot_seed)
    gfreq = uni.to(torch.float64)                                         # codes-independent static-marginal trace (same fade-frequent direction as the activity trace; the inverse-freq control)

    anchor = exp63.raw_sppmi_svd_anchor(sppmi, para, rand, collo, kq, args.svd_rank, args.n_boot, args.boot_seed)
    a_spec = anchor["specificity_para_minus_random"]["mean"]
    a_lo, a_hi = anchor["specificity_para_minus_random"]["ci"]
    hw = (a_hi - a_lo) / 2 if a_hi == a_hi else 0.0
    calib_ok = bool((0.082 - hw) <= a_spec <= (0.137 + hw) and abs(anchor["king_queen_cos"] - 0.222) <= 0.04)

    def rd(Y, seed):
        return exp68.read_specificity(Y, para, rand, collo, shuf, args.n_boot, seed)

    # ---- ceilings (NMF at k and at Lmax*k) + grow_G floor (apparatus) ----
    nmf_k = st.mean([rd(exp65.nmf_slots(M0, args.k, seed=s), s)["spec_para_minus_rand"][0] for s in range(3)])
    nmf_Lk = st.mean([rd(exp65.nmf_slots(M0, args.k * Lmax, seed=s), s)["spec_para_minus_rand"][0] for s in range(3)])
    ceiling = max(nmf_k, nmf_Lk, a_spec)
    subf = TorchFHRR(dim=args.D_floor, seed=0, device="cpu", alpha_anti=1.0)
    G0 = subf.random_vectors(V)
    floor = exp65.faithful_read(subf, G0, exp61.d_eff(subf, G0), M0, para, rand, None,
                                [0.0], 20, 0.3, 0.9, args.n_boot, args.boot_seed)
    floor_spec = floor["best"]["spec"] if floor["best"] else 0.0
    print(f"V={V} n_para={len(para)} src={src} [anchor] spec={a_spec:+.4f} kq={anchor['king_queen_cos']:.3f} "
          f"calib_ok={calib_ok} | nmf_k={nmf_k:+.4f} nmf_Lk={nmf_Lk:+.4f} ceiling={ceiling:+.4f} floor={floor_spec:+.4f}",
          file=sys.stderr, flush=True)

    # ---- arms ----
    ARMS = {
        "kwta_habituation": (capture_kwta, "activity"),     # the live candidate
        "kwta_freqdeflate": (capture_kwta, "freq"),         # THE inverse-frequency control (must be beaten)
        "kwta_nodeflate":   (capture_kwta, "none"),         # Control C: no deflation (depth/concat only)
        "linear_habituation": (capture_linear, "activity"), # predicted-NULL arm (linear depth = ordered PCA)
    }
    results = {}
    for arm, (cap, defl) in ARMS.items():
        per_L = {}
        for L in L_grid:
            specs, cilos, kills, cent_rise = [], [], [], []
            for seed in range(args.seeds):
                Y, norms, cents = multilayer_write(M0, cap, L, args.k, args.kappa, seed, defl, gfreq, para, rand)
                r = rd(Y, seed)
                specs.append(r["spec_para_minus_rand"][0]); cilos.append(r["spec_para_minus_rand"][1])
                kills.append(r["kill_para_minus_shuf"][1])
                cval = [c for c in cents if c == c]
                cent_rise.append((cval[-1] - cval[0]) if len(cval) >= 2 else float("nan"))
            per_L[L] = {"spec_mean": st.mean(specs), "spec_ci_lo": st.mean(cilos),
                        "bkill_seeds": sum(1 for x in kills if x > 0),
                        "centroid_rise_mean": st.mean([c for c in cent_rise if c == c] or [float("nan")])}
            print(f"[{arm} L={L}] spec={per_L[L]['spec_mean']:+.4f}(CIlo {per_L[L]['spec_ci_lo']:+.4f}) "
                  f"B-KILL={per_L[L]['bkill_seeds']}/{args.seeds} centroid_rise={per_L[L]['centroid_rise_mean']:+.3f}",
                  file=sys.stderr, flush=True)
        results[arm] = per_L

    # Control D: frozen-random deep stack (Lmax blocks)
    randstack = []
    for seed in range(args.seeds):
        Yr = l2rows(torch.cat([exp68.random_nonneg(V, args.k, seed * 13 + j) for j in range(Lmax)], dim=1))
        randstack.append(rd(Yr, seed)["spec_para_minus_rand"][0])
    randstack_spec = st.mean(randstack)

    # ---- gates on the live arm (kwta_habituation) ----
    hab = results["kwta_habituation"]; freqd = results["kwta_freqdeflate"]; nod = results["kwta_nodeflate"]
    L1 = min(L_grid); bestL = max(L_grid, key=lambda L: hab[L]["spec_mean"])
    H, F1 = hab[bestL], freqd[bestL]
    g1 = H["spec_ci_lo"] > 0.04
    g2 = (H["spec_mean"] - floor_spec) >= 0.02
    g4 = (H["spec_mean"] - randstack_spec) >= 0.02
    g5_depth = bestL > L1 and (hab[bestL]["spec_mean"] - hab[L1]["spec_mean"]) >= 0.02 and H["centroid_rise_mean"] > 0
    g_bkill = H["bkill_seeds"] >= args.bkill_min and H["spec_ci_lo"] > 0.0
    g_beats_freq = (H["spec_mean"] - F1["spec_mean"]) >= 0.02
    g_ceiling = H["spec_mean"] <= ceiling + 1e-9
    g_beats_nodeflate = (hab[bestL]["spec_mean"] - nod[bestL]["spec_mean"]) >= 0.02
    PASS = bool(calib_ok and g1 and g2 and g4 and g5_depth and g_bkill and g_beats_freq and g_ceiling and g_beats_nodeflate)
    gates = {"g1": g1, "g2_beats_floor": g2, "g4_beats_randstack": g4, "g5_depth": g5_depth,
             "g_bkill": g_bkill, "g_beats_inverse_freq": g_beats_freq, "g_ceiling_reach_not_exceed": g_ceiling,
             "g_beats_nodeflate": g_beats_nodeflate, "best_L": bestL, "PASS": PASS}

    if not calib_ok:
        verdict = "INVALID — anchor missed +0.109/0.222."
    elif hab[L1]["spec_ci_lo"] > 0.04 and hab[L1]["bkill_seeds"] >= args.bkill_min:
        verdict = "INVALID — L=1 already PASSES → a top-down/global signal leaked in (apparatus check, precommit §3)."
    elif PASS:
        verdict = ("PASS — depth+habituation reaches the subdominant structure: L>1 strictly beats L=1 (depth "
                   "load-bearing), centroid rises, beats the inverse-frequency stack AND no-deflate Control, "
                   "reach-not-exceed ceiling. FIRST local route past the single-LAYER bound. Escalate to rung-2 "
                   "FHRR port BEFORE banking; 'local route to the global code', NOT 'beats SVD'.")
    else:
        verdict = ("NULL — sub-case from gates: (a) no-composition if g5_depth False (L>1≈L=1 / centroid flat) → "
                   "the bound survives genuine depth → Codebook-growth⇄Replay or substrate change; "
                   "(deflates-to-PMI) if g_beats_inverse_freq False (≈ the freqdeflate stack) → the 5th narrowing; "
                   "(dim-inflation) if g_beats_nodeflate False → depth/concat is the lever, not deflation; "
                   "linear arm predicted-NULL (banked sub-result).")

    out = {"experiment": "77_depth_habituation_oracle (DRILL-DOWN, NOT graduation)",
           "precommit": "notes/emergent-codebook/phase-3-depth-habituation-precommit.md",
           "config": {"V": V, "k": args.k, "L_grid": L_grid, "kappa": args.kappa, "seeds": args.seeds,
                      "corpus_source": args.corpus_source, "simlex": f"{src}; sha={sha[:16] if sha and sha != 'PLANTED' else sha}"},
           "anchor": {"spec": a_spec, "kq": anchor["king_queen_cos"], "calib_ok": calib_ok},
           "ceiling": {"nmf_k": nmf_k, "nmf_Lk": nmf_Lk, "ceiling": ceiling, "floor": floor_spec,
                       "randstack": randstack_spec},
           "results": results, "gates": gates, "VERDICT": verdict}
    print(json.dumps(out, indent=2, default=float))
    print(f"[VERDICT] {verdict[:130]}", file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
